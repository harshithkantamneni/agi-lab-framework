/* MetalBridge.swift -- Swift-Metal bridge for GPU-accelerated compute.
 *
 * Provides C-callable functions for GPU matrix multiplication.
 * Uses unified memory (MTLStorageModeShared) for zero-copy CPU<->GPU
 * data sharing on Apple Silicon.
 *
 * Pipeline states are cached on first use. The Metal device and
 * command queue are created once during metal_init().
 *
 * Thread safety: all public functions use a serial lock to ensure
 * safe concurrent access (future-proofing for multi-threaded training).
 */

import Foundation
import Metal

// MARK: - Metal Context (singleton)

/// Holds all Metal state: device, queue, pipeline states.
/// Created by metal_init(), destroyed by metal_cleanup().
private final class MetalContext {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let library: MTLLibrary
    var pipelineMatmul: MTLComputePipelineState?

    init?() {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            fputs("MetalBridge: no Metal device found\n", stderr)
            return nil
        }
        self.device = dev

        guard let q = dev.makeCommandQueue() else {
            fputs("MetalBridge: failed to create command queue\n", stderr)
            return nil
        }
        self.queue = q

        // Load the default metallib from the build directory.
        // Search paths: next to the executable, then ../build/, then build/
        let execPath = ProcessInfo.processInfo.arguments[0]
        let execDir = (execPath as NSString).deletingLastPathComponent

        let searchPaths = [
            "\(execDir)/default.metallib",
            "\(execDir)/../default.metallib",
            "\(execDir)/../build/default.metallib",
            "build/default.metallib",
            "default.metallib",
        ]

        var loadedLib: MTLLibrary? = nil
        for path in searchPaths {
            let url = URL(fileURLWithPath: path)
            if FileManager.default.fileExists(atPath: url.path) {
                do {
                    loadedLib = try dev.makeLibrary(URL: url)
                    break
                } catch {
                    // Try next path
                }
            }
        }

        guard let lib = loadedLib else {
            fputs("MetalBridge: failed to load default.metallib from any search path\n", stderr)
            fputs("  Searched: \(searchPaths)\n", stderr)
            return nil
        }
        self.library = lib
    }

    /// Build and cache the matmul_fp32 pipeline state.
    func getMatmulPipeline() -> MTLComputePipelineState? {
        if let cached = pipelineMatmul {
            return cached
        }

        guard let fn = library.makeFunction(name: "matmul_fp32") else {
            fputs("MetalBridge: function 'matmul_fp32' not found in metallib\n", stderr)
            return nil
        }

        do {
            let pipeline = try device.makeComputePipelineState(function: fn)
            self.pipelineMatmul = pipeline
            return pipeline
        } catch {
            fputs("MetalBridge: failed to create matmul pipeline: \(error)\n", stderr)
            return nil
        }
    }
}

// MARK: - Global State

/// Serial lock for thread safety.
private let metalLock = NSLock()

/// The singleton Metal context, created by metal_init().
private var ctx: MetalContext? = nil

// MARK: - Threshold for GPU vs CPU

/// Minimum total FLOPs (2*M*N*K) before we dispatch to GPU.
/// Below this threshold, the overhead of GPU dispatch exceeds the compute savings.
/// Tuned for M3 Pro: GPU dispatch overhead ~50-100us, AMX cblas is very fast for small sizes.
private let GPU_FLOP_THRESHOLD: Int = 2 * 128 * 128 * 128  // ~4M FLOPs

// MARK: - C-Callable Functions

/// Initialize Metal device, command queue, and load shaders.
/// Returns 0 on success, -1 on failure.
@_cdecl("metal_init")
public func metal_init() -> Int32 {
    metalLock.lock()
    defer { metalLock.unlock() }

    // Already initialized
    if ctx != nil {
        return 0
    }

    guard let newCtx = MetalContext() else {
        return -1
    }

    // Pre-warm the matmul pipeline (compile shader on init, not first use)
    guard newCtx.getMatmulPipeline() != nil else {
        return -1
    }

    ctx = newCtx
    return 0
}

/// Release all Metal resources.
@_cdecl("metal_cleanup")
public func metal_cleanup() {
    metalLock.lock()
    defer { metalLock.unlock() }
    ctx = nil
}

/// Query whether Metal is initialized.
@_cdecl("metal_is_ready")
public func metal_is_ready() -> Int32 {
    metalLock.lock()
    defer { metalLock.unlock() }
    return ctx != nil ? 1 : 0
}

/// FP32 matrix multiplication on GPU: C = A @ B.
/// A: [M,K], B: [K,N], C: [M,N], all row-major FP32.
@_cdecl("metal_matmul")
public func metal_matmul(
    _ A: UnsafePointer<Float>,
    _ B: UnsafePointer<Float>,
    _ C: UnsafeMutablePointer<Float>,
    _ M: Int32, _ N: Int32, _ K: Int32
) {
    let m = Int(M)
    let n = Int(N)
    let k = Int(K)

    guard m > 0, n > 0, k > 0 else { return }

    metalLock.lock()
    guard let context = ctx else {
        metalLock.unlock()
        fputs("MetalBridge: metal_matmul called before metal_init\n", stderr)
        return
    }
    guard let pipeline = context.pipelineMatmul else {
        metalLock.unlock()
        fputs("MetalBridge: matmul pipeline not available\n", stderr)
        return
    }
    metalLock.unlock()

    autoreleasepool {
        let aSizeBytes = m * k * MemoryLayout<Float>.size
        let bSizeBytes = k * n * MemoryLayout<Float>.size
        let cSizeBytes = m * n * MemoryLayout<Float>.size

        let device = context.device

        guard let bufA = device.makeBuffer(bytes: A, length: aSizeBytes, options: .storageModeShared),
              let bufB = device.makeBuffer(bytes: B, length: bSizeBytes, options: .storageModeShared),
              let bufC = device.makeBuffer(length: cSizeBytes, options: .storageModeShared)
        else {
            fputs("MetalBridge: failed to create Metal buffers\n", stderr)
            return
        }

        var mVal = UInt32(m)
        var kVal = UInt32(k)
        var nVal = UInt32(n)

        guard let cmdBuf = context.queue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder()
        else {
            fputs("MetalBridge: failed to create command buffer/encoder\n", stderr)
            return
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(bufA, offset: 0, index: 0)
        encoder.setBuffer(bufB, offset: 0, index: 1)
        encoder.setBuffer(bufC, offset: 0, index: 2)
        encoder.setBytes(&mVal, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&kVal, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&nVal, length: MemoryLayout<UInt32>.size, index: 5)

        // Optimized kernel: 32x32 output tiles, 128 threads (4 simdgroups of 32)
        let tileN: Int = 32
        let tileM: Int = 32
        let gridX = (n + tileN - 1) / tileN
        let gridY = (m + tileM - 1) / tileM
        let threadgroupsPerGrid = MTLSize(width: gridX, height: gridY, depth: 1)
        let threadsPerGroup = MTLSize(width: 128, height: 1, depth: 1)

        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let resultPtr = bufC.contents().bindMemory(to: Float.self, capacity: m * n)
        C.update(from: resultPtr, count: m * n)
    }
}
