#!/usr/bin/env python3
"""Numerical Verification — Compare C implementations against SymPy ground truth."""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import tempfile
import numpy as np
from sympy import *
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
BUILD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build")


def compile_c_function(c_code, func_name, extra_flags=None):
    """Compile a C function to a shared library and load it."""
    with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
        f.write(c_code)
        c_path = f.name

    so_path = c_path.replace(".c", ".dylib")
    cmd = [
        "clang", "-shared", "-fPIC", "-O2", "-mcpu=apple-m3",
        "-framework", "Accelerate",
        c_path, "-o", so_path,
    ]
    if extra_flags:
        cmd.extend(extra_flags)

    result = subprocess.run(cmd, capture_output=True, text=True)
    os.unlink(c_path)

    if result.returncode != 0:
        print(f"Compilation failed:\n{result.stderr}", file=sys.stderr)
        return None, None

    lib = ctypes.CDLL(so_path)
    return lib, so_path


def verify_scalar_function(c_code, func_name, sympy_expr_str, var="x",
                           test_range=(-10, 10), n_points=1000, tolerance=1e-5):
    """Verify a C scalar function against a SymPy expression."""
    # Parse symbolic expression
    v = Symbol(var)
    expr = parse_expr(sympy_expr_str, transformations=TRANSFORMS)
    sym_fn = lambdify(v, expr, modules=["numpy"])

    # Compile C function
    lib, so_path = compile_c_function(c_code, func_name)
    if lib is None:
        return {"passed": False, "error": "compilation failed"}

    # Set up C function signature (float -> float)
    c_fn = getattr(lib, func_name)
    c_fn.restype = ctypes.c_float
    c_fn.argtypes = [ctypes.c_float]

    # Generate test points
    points = np.linspace(test_range[0], test_range[1], n_points, dtype=np.float32)

    # Compare
    max_error = 0.0
    failures = []
    for x_val in points:
        try:
            c_result = c_fn(ctypes.c_float(float(x_val)))
            sym_result = float(sym_fn(float(x_val)))

            if np.isnan(sym_result) or np.isinf(sym_result):
                continue

            error = abs(c_result - sym_result)
            rel_error = error / (abs(sym_result) + 1e-10)
            max_error = max(max_error, rel_error)

            if rel_error > tolerance:
                failures.append({
                    "x": float(x_val),
                    "c_result": float(c_result),
                    "expected": sym_result,
                    "rel_error": float(rel_error),
                })
        except Exception as e:
            failures.append({"x": float(x_val), "error": str(e)})

    os.unlink(so_path)

    passed = len(failures) == 0
    result = {
        "passed": passed,
        "func_name": func_name,
        "sympy_expr": sympy_expr_str,
        "n_points": n_points,
        "max_relative_error": float(max_error),
        "n_failures": len(failures),
        "tolerance": tolerance,
    }
    if failures:
        result["first_failures"] = failures[:5]

    return result


def verify_matrix_op(c_code, func_name, op="matmul", sizes=None, tolerance=1e-4):
    """Verify a C matrix operation against numpy."""
    if sizes is None:
        sizes = [4, 8, 16, 32]

    lib, so_path = compile_c_function(c_code, func_name)
    if lib is None:
        return {"passed": False, "error": "compilation failed"}

    results = []
    all_passed = True

    for n in sizes:
        a = np.random.randn(n, n).astype(np.float32)
        b = np.random.randn(n, n).astype(np.float32)
        c_out = np.zeros((n, n), dtype=np.float32)

        if op == "matmul":
            expected = a @ b
        elif op == "transpose":
            expected = a.T
        elif op == "add":
            expected = a + b
        else:
            expected = a @ b

        # Call C function
        c_fn = getattr(lib, func_name)
        c_fn.restype = None
        c_fn.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int
        ]
        c_fn(
            a.ctypes.data_as(ctypes.c_void_p),
            b.ctypes.data_as(ctypes.c_void_p),
            c_out.ctypes.data_as(ctypes.c_void_p),
            n
        )

        max_err = float(np.max(np.abs(c_out - expected)))
        passed = max_err < tolerance

        results.append({"size": n, "max_error": max_err, "passed": passed})
        if not passed:
            all_passed = False

    os.unlink(so_path)
    return {"passed": all_passed, "op": op, "results": results}


def main():
    parser = argparse.ArgumentParser(description="Numerical Verification Tool")
    sub = parser.add_subparsers(dest="command")

    sc = sub.add_parser("scalar", help="Verify scalar C function against SymPy")
    sc.add_argument("c_file", help="C source file")
    sc.add_argument("func_name", help="Function name to test")
    sc.add_argument("sympy_expr", help="SymPy expression (ground truth)")
    sc.add_argument("--var", default="x")
    sc.add_argument("--range", nargs=2, type=float, default=[-10, 10])
    sc.add_argument("--points", type=int, default=1000)
    sc.add_argument("--tolerance", type=float, default=1e-5)

    mx = sub.add_parser("matrix", help="Verify matrix C function against numpy")
    mx.add_argument("c_file", help="C source file")
    mx.add_argument("func_name", help="Function name to test")
    mx.add_argument("--op", default="matmul", choices=["matmul", "transpose", "add"])
    mx.add_argument("--sizes", nargs="+", type=int, default=[4, 8, 16, 32])
    mx.add_argument("--tolerance", type=float, default=1e-4)

    args = parser.parse_args()

    if args.command == "scalar":
        with open(args.c_file) as f:
            c_code = f.read()
        result = verify_scalar_function(
            c_code, args.func_name, args.sympy_expr,
            var=args.var, test_range=args.range,
            n_points=args.points, tolerance=args.tolerance
        )
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["passed"] else 1)

    elif args.command == "matrix":
        with open(args.c_file) as f:
            c_code = f.read()
        result = verify_matrix_op(
            c_code, args.func_name, op=args.op,
            sizes=args.sizes, tolerance=args.tolerance
        )
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["passed"] else 1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
