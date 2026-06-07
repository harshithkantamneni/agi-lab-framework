# AGI Project — Master Build System
# Targets: C core, Metal shaders, Swift bridge, Python tools

SHELL := /bin/bash
.DEFAULT_GOAL := help

# === Directories ===
SRC_DIR     := src
BUILD_DIR   := build
SHADER_DIR  := src/metal
SWIFT_DIR   := src/swift
TOOLS_DIR   := tools
DATA_DIR    := data
CHECKPOINTS := data/checkpoints

# === Compiler Settings (M3 Pro optimized) ===
CC          := clang
INCLUDES    := -Isrc/core -Isrc/model -Isrc/training
CFLAGS      := -Wall -Wextra -Werror -std=c17 -O2 -mcpu=apple-m3 -fPIC -DACCELERATE_NEW_LAPACK $(INCLUDES)
CFLAGS_DBG  := -Wall -Wextra -std=c17 -g -O0 -mcpu=apple-m3 -fsanitize=address,undefined -fPIC -DACCELERATE_NEW_LAPACK $(INCLUDES)
LDFLAGS     := -framework Metal -framework MetalPerformanceShaders -framework Foundation -framework Accelerate
METAL       := xcrun -sdk macosx metal
METALLIB    := xcrun -sdk macosx metallib
SWIFT       := swiftc
SWIFTFLAGS  := -O -target arm64-apple-macosx14.0

# === Source Discovery ===
# Exclude files with main() from the library sources
C_SRCS_ALL  := $(wildcard $(SRC_DIR)/**/*.c $(SRC_DIR)/*.c)
C_SRCS      := $(filter-out $(SRC_DIR)/training/micro_experiment.c $(SRC_DIR)/training/scale_experiment.c $(SRC_DIR)/training/train_tokenizer.c $(SRC_DIR)/eval/eval_model.c,$(C_SRCS_ALL))
C_OBJS      := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(C_SRCS))
METAL_SRCS  := $(wildcard $(SHADER_DIR)/*.metal)
METAL_AIR   := $(patsubst $(SHADER_DIR)/%.metal,$(BUILD_DIR)/metal/%.air,$(METAL_SRCS))
METAL_LIB   := $(BUILD_DIR)/default.metallib
SWIFT_SRCS  := $(wildcard $(SWIFT_DIR)/*.swift)

# === Test Discovery ===
# Default suite: tests/test_*.c, EXCLUDING tests/quarantine/*.
# Quarantined tests are built + run via `make test-quarantine` only.
# See tests/quarantine/README.md for the policy and D-097 P2.
# Note: tests/test_config_drift.c (P-CONFIG-DRIFT-DETECTION, D-121) is in-band
# in the default `test` target via this glob. Its impl source
# src/training/config_drift.c is also in-band via the C_SRCS glob above.
TEST_SRCS   := $(wildcard tests/test_*.c)
TEST_BINS   := $(patsubst tests/test_%.c,$(BUILD_DIR)/tests/test_%,$(TEST_SRCS))

# Quarantined tests (explicit list; not globbed so any future quarantined
# test requires an intentional addition to this line).
QUARANTINE_SRCS := $(wildcard tests/quarantine/test_*.c)
QUARANTINE_BINS := $(patsubst tests/quarantine/test_%.c,$(BUILD_DIR)/tests/quarantine/test_%,$(QUARANTINE_SRCS))

# === Targets ===

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: all
all: dirs core shaders swift-bridge ## Build everything

.PHONY: dirs
dirs: ## Create build directories
	@mkdir -p $(BUILD_DIR)/metal $(BUILD_DIR)/core $(BUILD_DIR)/swift $(BUILD_DIR)/tests $(BUILD_DIR)/training $(BUILD_DIR)/eval
	@mkdir -p $(DATA_DIR)/papers $(DATA_DIR)/training/active $(CHECKPOINTS)
	@mkdir -p src/core src/metal src/swift src/tests src/eval

.PHONY: core
core: dirs $(C_OBJS) ## Build C core library
	@echo "=== C Core built ==="

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: core-debug
core-debug: CFLAGS := $(CFLAGS_DBG)
core-debug: dirs $(C_OBJS) ## Build C core with debug + sanitizers

.PHONY: shaders
shaders: dirs $(METAL_LIB) ## Compile Metal shaders
	@echo "=== Metal shaders built ==="

$(BUILD_DIR)/metal/%.air: $(SHADER_DIR)/%.metal
	$(METAL) -c $< -o $@

$(METAL_LIB): $(METAL_AIR)
	@if [ -n "$(METAL_AIR)" ]; then $(METALLIB) $(METAL_AIR) -o $@; fi

.PHONY: compile-commands
compile-commands: dirs ## Generate compile_commands.json for clangd
	@echo "[" > compile_commands.json
	@first=true; for src in $(C_SRCS); do \
		if [ "$$first" = true ]; then first=false; else echo "," >> compile_commands.json; fi; \
		echo "  {" >> compile_commands.json; \
		echo "    \"directory\": \"$(CURDIR)\"," >> compile_commands.json; \
		echo "    \"command\": \"$(CC) $(CFLAGS) -c $$src\"," >> compile_commands.json; \
		echo "    \"file\": \"$$src\"" >> compile_commands.json; \
		echo -n "  }" >> compile_commands.json; \
	done; \
	echo "" >> compile_commands.json
	@echo "]" >> compile_commands.json
	@echo "=== compile_commands.json generated ==="

# === Swift Bridge ===
SWIFT_LIB    := $(BUILD_DIR)/swift/libMetalBridge.dylib
SWIFT_HEADER := $(BUILD_DIR)/swift/MetalBridge-Swift.h

.PHONY: swift-bridge
swift-bridge: dirs shaders $(SWIFT_LIB) ## Build Swift-Metal bridge library
	@echo "=== Swift-Metal bridge built ==="

$(SWIFT_LIB): $(SWIFT_DIR)/MetalBridge.swift
	@mkdir -p $(BUILD_DIR)/swift
	$(SWIFT) $(SWIFTFLAGS) \
		-emit-library \
		-emit-objc-header -emit-objc-header-path $(SWIFT_HEADER) \
		-module-name MetalBridge \
		-o $@ $<

# === Test build rules ===
# Each test binary compiles the C sources directly with debug flags + sanitizers.
# This ensures tests always run with ASan/UBSan regardless of core build state.

# Special rule for Metal matmul test (QUARANTINED — see tests/quarantine/README.md).
# Links against Swift bridge + Metal. Not part of the default `make test` suite.
$(BUILD_DIR)/tests/quarantine/test_metal_matmul: tests/quarantine/test_metal_matmul.c $(SWIFT_LIB)
	@mkdir -p $(dir $@)
	$(CC) -std=c17 -O2 -mcpu=apple-m3 -DACCELERATE_NEW_LAPACK \
		-Isrc/swift -Isrc/core -Isrc/tests \
		$< -o $@ \
		-L$(BUILD_DIR)/swift -lMetalBridge \
		-Xlinker -rpath -Xlinker @executable_path/../swift \
		-framework Accelerate -framework Metal -framework Foundation

# Special rule for eval test — needs -Isrc/eval (eval_utils.c is already in C_SRCS)
$(BUILD_DIR)/tests/test_eval: tests/test_eval.c $(C_SRCS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS_DBG) -Isrc/core -Isrc/model -Isrc/training -Isrc/eval -Isrc/tests $< $(C_SRCS) -o $@ $(LDFLAGS)

# Special rule for score-per-choice integration test — subprocess-only, no C_SRCS link.
# Passes fixture paths as defines so tests can find the built binary + checkpoint.
# EVAL_MODEL_DBG_BIN: the sanitizer-instrumented SUT binary (D-325 FLAG-3).
# run_eval_model() in the test prefers this binary when the file is present,
# falling back to EVAL_MODEL_BIN when eval-debug has not been built.
$(BUILD_DIR)/tests/test_eval_model_score_per_choice: tests/test_eval_model_score_per_choice.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS_DBG) -Isrc/tests \
		-DEVAL_MODEL_BIN='"build/eval_model"' \
		-DEVAL_MODEL_DBG_BIN='"build/eval_model_dbg"' \
		-DFIXTURE_CHECKPOINT='"data/checkpoints/phase3_factorial/A42/final.ckpt"' \
		-DFIXTURE_TOKENIZER='"data/training/tokenizer_32k.bin"' \
		$< -o $@

# Default rule for all other test binaries (C-only, with sanitizers)
$(BUILD_DIR)/tests/test_%: tests/test_%.c $(C_SRCS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS_DBG) -Isrc/core -Isrc/model -Isrc/training -Isrc/tests $< $(C_SRCS) -o $@ $(LDFLAGS)

.PHONY: test-build
test-build: dirs $(TEST_BINS) ## Build all test binaries

.PHONY: test
test: test-build ## Run all tests
	@echo "=== Running C tests ==="
	@for t in $(TEST_BINS); do \
		echo "Running $$t..."; \
		MTL_DEBUG_LAYER=1 $$t || exit 1; \
	done
	@echo "=== Running Python tests ==="
	@source .venv/bin/activate && python -m pytest tests/ -v
	@echo "=== All tests passed ==="

.PHONY: test-python
test-python: ## Run Python tests only
	@source .venv/bin/activate && python -m pytest tests/ --ignore=tests/quarantine -v

.PHONY: test-quarantine-build
test-quarantine-build: dirs $(SWIFT_LIB) $(QUARANTINE_BINS) ## Build quarantined tests only

.PHONY: test-quarantine
test-quarantine: test-quarantine-build ## Run quarantined tests (NOT part of `make test`)
	@echo "=== Running QUARANTINED C tests (see tests/quarantine/README.md) ==="
	@if [ -z "$(QUARANTINE_BINS)" ]; then echo "(no quarantined tests)"; exit 0; fi
	@fail=0; for t in $(QUARANTINE_BINS); do \
		echo "Running $$t..."; \
		MTL_DEBUG_LAYER=1 $$t || fail=1; \
	done; \
	if [ $$fail -ne 0 ]; then \
		echo "=== Quarantined tests FAILED (expected: see reinstate trigger) ==="; \
	else \
		echo "=== Quarantined tests all passed (consider reinstating to tests/) ==="; \
	fi

.PHONY: lint
lint: ## Run all linters
	@echo "=== C lint ==="
	@if command -v clang-format &>/dev/null; then \
		find $(SRC_DIR) -name '*.c' -o -name '*.h' | xargs clang-format --dry-run --Werror 2>&1 || true; \
	fi
	@echo "=== Python lint ==="
	@source .venv/bin/activate && python -m mypy $(TOOLS_DIR)/ --ignore-missing-imports 2>/dev/null || true
	@echo "=== Swift lint ==="
	@if command -v swiftlint &>/dev/null; then \
		swiftlint lint $(SWIFT_DIR)/ 2>/dev/null || true; \
	fi

.PHONY: format
format: ## Auto-format all code
	@if command -v clang-format &>/dev/null; then \
		find $(SRC_DIR) -name '*.c' -o -name '*.h' | xargs clang-format -i; \
	fi
	@if command -v swiftformat &>/dev/null; then \
		swiftformat $(SWIFT_DIR)/ 2>/dev/null || true; \
	fi

.PHONY: hwmon
hwmon: ## Show hardware status
	@source .venv/bin/activate && python $(TOOLS_DIR)/hwmon.py

.PHONY: preflight
preflight: ## Pre-training resource check
	@source .venv/bin/activate && python $(TOOLS_DIR)/preflight.py

.PHONY: benchmark
benchmark: ## Run benchmarks
	@source .venv/bin/activate && python $(TOOLS_DIR)/benchmark.py

.PHONY: experiments
experiments: ## Show experiment stats
	@source .venv/bin/activate && python $(TOOLS_DIR)/experiments.py stats

.PHONY: slack-bot-start
slack-bot-start: ## Start the Slack bot daemon in tmux (curated updates + bidirectional commands)
	@if tmux has-session -t agi-slack-bot 2>/dev/null; then \
		echo "Slack bot already running."; \
	else \
		ENV_FILE="$$HOME/.config/agi-slack-bot.env"; \
		tmux new-session -d -s agi-slack-bot \
			"caffeinate -di bash -lc 'cd $(PWD) && \
				if [ -f $$ENV_FILE ]; then set -a; . $$ENV_FILE; set +a; fi; \
				if [ -z \"\$$AGI_LAB_SLACK_BOT_TOKEN\" ] || [ -z \"\$$AGI_LAB_SLACK_APP_TOKEN\" ] || [ -z \"\$$AGI_LAB_SLACK_CHANNEL\" ]; then \
					echo \"ERROR: tokens missing. Set in ~/.config/agi-slack-bot.env or export before make.\"; \
					echo \"Required: AGI_LAB_SLACK_BOT_TOKEN, AGI_LAB_SLACK_APP_TOKEN, AGI_LAB_SLACK_CHANNEL\"; \
					exit 1; \
				fi; \
				source .venv/bin/activate && python3 tools/slack_bot.py 2>&1 | tee -a data/infra/slack_bot.log'"; \
		echo "=== AGI Lab Slack bot started in tmux session 'agi-slack-bot' ==="; \
		echo "Tokens loaded from: \$$AGI_LAB_SLACK_*  OR  ~/.config/agi-slack-bot.env (if exists)"; \
		echo "Log: data/infra/slack_bot.log"; \
		echo "Attach: tmux attach -t agi-slack-bot"; \
	fi

.PHONY: slack-bot-stop
slack-bot-stop: ## Stop the Slack bot daemon
	@tmux kill-session -t agi-slack-bot 2>/dev/null && echo "Slack bot stopped." || echo "Slack bot was not running."

.PHONY: slack-bot-log
slack-bot-log: ## Tail the Slack bot log
	@tail -f data/infra/slack_bot.log

.PHONY: lab-start
lab-start: ## Start the autonomous AGI lab in tmux (wrapped in caffeinate to survive rate-limit waits)
	@if tmux has-session -t agi-lab 2>/dev/null; then \
		echo "AGI lab is already running. Use 'make lab-attach' to view."; \
	else \
		tmux new-session -d -s agi-lab 'caffeinate -dis ./run_agi_lab.sh'; \
		echo "=== AGI Lab started in tmux session 'agi-lab' ==="; \
		echo "  (wrapped in 'caffeinate -dis' to prevent sleep during long rate-limit waits)"; \
		echo "Use 'make lab-attach' to view, 'make lab-status' for status"; \
	fi

.PHONY: lab-stop
lab-stop: ## Stop the autonomous AGI lab
	@tmux kill-session -t agi-lab 2>/dev/null && echo "AGI lab stopped." || echo "AGI lab is not running."

.PHONY: lab-attach
lab-attach: ## Attach to the AGI lab tmux session
	@tmux attach -t agi-lab

.PHONY: lab-status
lab-status: ## Show AGI lab status (non-interactive)
	@echo "=== Session State ==="
	@cat data/session_state.md 2>/dev/null || echo "(no session state — lab has not run yet)"
	@echo ""
	@echo "=== Benchmark Tracker ==="
	@cat data/benchmark_tracker.md 2>/dev/null || echo "(no tracker)"
	@echo ""
	@echo "=== Session Count ==="
	@ls data/infra/session_logs/ 2>/dev/null | wc -l | xargs -I{} echo "{} sessions completed"

.PHONY: lab-log
lab-log: ## Show the latest session log
	@ls -t data/infra/session_logs/*.log 2>/dev/null | head -1 | xargs cat 2>/dev/null || echo "No session logs yet."

.PHONY: lab-dashboard
lab-dashboard: ## Open live dashboard in browser (http://localhost:8420)
	@source .venv/bin/activate && python $(TOOLS_DIR)/dashboard.py &
	@sleep 1
	@open http://localhost:8420
	@echo "=== Dashboard at http://localhost:8420 (Ctrl+C in terminal to stop) ==="

# === Micro-experiment binary ===
MICRO_EXP   := $(BUILD_DIR)/micro_experiment

.PHONY: micro
micro: dirs $(MICRO_EXP) ## Build micro-experiment binary
	@echo "=== Micro-experiment built: $(MICRO_EXP) ==="

$(MICRO_EXP): $(SRC_DIR)/training/micro_experiment.c $(C_SRCS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $< $(C_SRCS) -o $@ $(LDFLAGS)

.PHONY: micro-run
micro-run: micro ## Build and run micro-experiment (500 steps, S=32, patterns)
	@echo "=== Running Micro-Experiment (iPC) ==="
	$(MICRO_EXP) 500 32 1

.PHONY: micro-backprop
micro-backprop: micro ## Build and run micro-experiment with backprop (500 steps)
	@echo "=== Running Micro-Experiment (Backprop) ==="
	$(MICRO_EXP) --backprop 500 32 1

# === Train-tokenizer binary ===
TRAIN_TOK   := $(BUILD_DIR)/train_tokenizer

.PHONY: train-tokenizer
train-tokenizer: dirs $(TRAIN_TOK) ## Build BPE tokenizer training utility
	@echo "=== train_tokenizer built: $(TRAIN_TOK) ==="

$(TRAIN_TOK): $(SRC_DIR)/training/train_tokenizer.c $(C_SRCS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $< $(C_SRCS) -o $@ $(LDFLAGS)

# === Scale-experiment binary ===
SCALE_EXP   := $(BUILD_DIR)/scale_experiment

.PHONY: scale
scale: dirs $(SCALE_EXP) ## Build scale-up experiment binary
	@echo "=== Scale experiment built: $(SCALE_EXP) ==="

$(SCALE_EXP): $(SRC_DIR)/training/scale_experiment.c $(C_SRCS) $(SWIFT_LIB)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -Isrc/swift $< $(C_SRCS) -o $@ $(LDFLAGS) \
		-L$(BUILD_DIR)/swift -lMetalBridge \
		-Xlinker -rpath -Xlinker @executable_path/../swift

.PHONY: scale-run
scale-run: scale ## Build and run scale-up experiment (10M, 2000 steps)
	@echo "=== Running Scale-Up Experiment ==="
	$(SCALE_EXP) --model small --steps 2000 --seq-len 128

# === Program 2 Phase 2 primary-run pair (Dense-A + MoE Rev-2) ===
# Session-spanning, session-detach-immune, auto-resume via tools/run_long.py.
# Serial `&&` enforces PC-3 AMX serialization (Cycle 20 PID 46031: 86%
# step_time inflation under concurrent AMX). Dense-A first as tripwire — if
# Dense-A diverges the MoE comparison is invalid and we save ~12.6h by not
# launching Rev-2. Flags signed into question.md (measurement_theorist +
# implementation_engineer_c); do not modify without PI sign-off.
# Spec: data/engineering/long_run_launcher_design.md (D-117, PI-approved).

.PHONY: run-phase2-pair
run-phase2-pair: scale ## Launch Dense-A then MoE Rev-2 (session-spanning, serial per PC-3)
	@mkdir -p data/checkpoints/phase2_dense_a data/checkpoints/phase2_moe_rev2 \
		data/runs/dense_a data/runs/moe_rev2
	python3 tools/run_long.py --run-id dense_a --cmd \
	  'build/scale_experiment --model dense50m --tokenizer data/training/tokenizer_32k.bin --backprop --stream --weight-seed 42 --steps 5000 \
	   --checkpoint-every 500 --checkpoint-dir data/checkpoints/phase2_dense_a' && \
	python3 tools/run_long.py --run-id moe_rev2 --cmd \
	  'build/scale_experiment --model medium --tokenizer data/training/tokenizer_32k.bin --backprop --stream --default-moe --entropy-penalty \
	   --temp-anneal --weight-seed 42 --steps 5000 --checkpoint-every 500 \
	   --checkpoint-dir data/checkpoints/phase2_moe_rev2'

# === Program 2 Phase 3 factorial 12-run launcher (D-193 P8) ===
# Drives the locked launch order A42 → D42 → B42 → C42 → A43 → D43 → B43 → C43 →
# A44 → D44 → B44 → C44 per phase3_p6_prereg.md §2.1 (D-192 LOCKED unanimous).
# Serial under PC-3 AMX-serialization (next never starts until previous exits).
# After run 8 (C43), invokes tools/b4_t1_evaluator.py per §8.1 to record T1 verdict.
# State machine (data/checkpoints/phase3_factorial/run_index.json) enables resume.
# Pre-flights (ALL must PASS before any cell launches):
#   - spec_invariants.yaml fingerprint snapshot/match (verdict-breaking on drift)
#   - per-cell config_drift prediction (12/12 must predict binary-side PASS)
#   - per-cell sentinel verification (P-CHECKPOINT-DIR-ARCHIVE-DISCIPLINE)
# Spec source-of-truth: phase3_p6_prereg.md §2.1 + §8 + §13.
# Apparatus design: data/engineering/d193_phase3_factorial_launch_design.md.
# IMPORTANT: factorial launch is BLOCKED until D-194 implementation_engineer_c
# follow-up lands (spec_invariants.yaml extension + config_drift.c LAB_CELL
# discriminator) — see d193 design §10.1-§10.3.

.PHONY: run-phase3-factorial
run-phase3-factorial: scale ## Launch the Phase-3 12-run factorial (locked order; serial; PC-3-safe; ~146h wall)
	@mkdir -p data/checkpoints/phase3_factorial data/runs
	@echo "=== Phase-3 factorial orchestration ==="
	@echo "Locked launch order: A42 -> D42 -> B42 -> C42 -> A43 -> D43 -> B43 -> C43 -> A44 -> D44 -> B44 -> C44"
	@echo "Wall budget: ~146.4h serialized (~6.1d) under PC-3 AMX-discipline"
	@echo "B4 T1 trigger eval: after run 8 (C43)"
	@echo "Apparatus design: data/engineering/d193_phase3_factorial_launch_design.md"
	@echo "Spec: programs/program_2_dense_vs_moe_sub100m/phase3_p6_prereg.md §2.1 + §8"
	python3 tools/run_phase3_factorial.py

.PHONY: run-phase3-factorial-dryrun
run-phase3-factorial-dryrun: scale ## Dry-run Phase-3 factorial: pre-flights + matrix print, no actual launches
	@echo "=== Phase-3 factorial DRY-RUN ==="
	python3 tools/run_phase3_factorial.py --dry-run

# === P11 benchmark scoring (apparatus.md §4.3) ===
EVAL_HARNESS_P11 := tools/eval_harness_p11.py
P11_OUTPUT_DIR   := data/eval/p11

.PHONY: eval-p11
eval-p11: $(EVAL_HARNESS_P11) ## Run P11 benchmark scoring (12 cells x 4 benchmarks; ~12-24h)
	@echo "P11 apparatus: lm-evaluation-harness commit 3fa4fd725c8a428710109f1d6c14eda37e95baea"
	@mkdir -p $(P11_OUTPUT_DIR)
	@source .venv/bin/activate && python3 $(EVAL_HARNESS_P11) \
	    --checkpoint-base data/checkpoints/phase3_factorial \
	    --output-dir $(P11_OUTPUT_DIR) \
	    --tokenizer data/training/tokenizer_32k.bin \
	    --output-json programs/program_2_dense_vs_moe_sub100m/benchmark_scores.json

.PHONY: eval-p11-smoke
eval-p11-smoke: ## Smoke test: single cell (A42) x 1 benchmark (MMLU, 100 items) for apparatus validation
	@source .venv/bin/activate && python3 $(EVAL_HARNESS_P11) \
	    --checkpoint data/checkpoints/phase3_factorial/A42/final.ckpt \
	    --output-dir $(P11_OUTPUT_DIR) \
	    --tokenizer data/training/tokenizer_32k.bin \
	    --task mmlu --limit 100 --smoke-test

# === Eval binary ===
EVAL_BIN    := $(BUILD_DIR)/eval_model

.PHONY: eval
eval: dirs $(EVAL_BIN) ## Build evaluation binary
	@echo "=== Eval model built: $(EVAL_BIN) ==="

$(EVAL_BIN): $(SRC_DIR)/eval/eval_model.c $(C_SRCS) $(SWIFT_LIB)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -Isrc/swift -Isrc/eval $< $(C_SRCS) -o $@ $(LDFLAGS) \
		-L$(BUILD_DIR)/swift -lMetalBridge \
		-Xlinker -rpath -Xlinker @executable_path/../swift

# === Eval debug binary (ASan + UBSan — D-325 FLAG-3) ===
# Builds build/eval_model_dbg with full sanitizer instrumentation so that
# ASan/UBSan run INSIDE the forked child process during integration tests.
# The iter-1 sizeof(prompt) bug slipped past 4 PASSING Unity tests because
# the child executed the non-sanitized release binary; this target closes
# that gap.  Run `make eval-debug && make test` to get sanitizer coverage
# on the SUT in addition to the test harness.
EVAL_DBG_BIN := $(BUILD_DIR)/eval_model_dbg

CFLAGS_EVAL_DBG := -Wall -Wextra -Werror -std=c17 -O0 -g -mcpu=apple-m3 \
                   -fsanitize=address,undefined -fPIC \
                   -DACCELERATE_NEW_LAPACK $(INCLUDES)

.PHONY: eval-debug
eval-debug: dirs $(EVAL_DBG_BIN) ## Build eval binary with ASan+UBSan (build/eval_model_dbg)
	@echo "=== Eval debug binary built: $(EVAL_DBG_BIN) (ASan+UBSan) ==="

$(EVAL_DBG_BIN): $(SRC_DIR)/eval/eval_model.c $(C_SRCS) $(SWIFT_LIB)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS_EVAL_DBG) -Isrc/swift -Isrc/eval $< $(C_SRCS) -o $@ $(LDFLAGS) \
		-L$(BUILD_DIR)/swift -lMetalBridge \
		-Xlinker -rpath -Xlinker @executable_path/../swift

.PHONY: clean
clean: ## Remove build artifacts
	rm -rf $(BUILD_DIR)
	rm -f compile_commands.json

.PHONY: clean-all
clean-all: clean ## Remove build + data caches
	rm -rf __pycache__ .pytest_cache
	find . -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

.PHONY: freeze
freeze: ## Lock Python dependencies
	@source .venv/bin/activate && pip freeze > requirements.txt
	@echo "=== requirements.txt updated ==="

# === Lab memory (tools/lab_memory.py) ===
# Always run lab_memory through the project venv — it needs sqlite-vec and
# sentence-transformers which are not available on system Python (PEP 668).
# Never invoke `python3 tools/lab_memory.py` directly; use these targets or
# `.venv/bin/python tools/lab_memory.py …`.

.PHONY: lab-memory-check
lab-memory-check: ## Smoke-test tools/lab_memory.py (sqlite_vec + schema + empty query)
	@echo "=== lab_memory smoke test (sqlite_vec + schema + empty-index query) ==="
	@.venv/bin/python -c "import sys, os, tempfile; sys.path.insert(0, 'tools'); \
import lab_memory; \
tmp = tempfile.mkdtemp(); db = os.path.join(tmp, 'smoke.db'); \
lm = lab_memory.LabMemory(db); lm.init(); \
hits = lm.search('empty index probe', top_k=3); \
assert hits == [], f'expected [], got {hits}'; \
print('OK: sqlite_vec loads, schema inits, empty search returns [].')"

.PHONY: lab-memory-test
lab-memory-test: ## Run tools/test_lab_memory.py test suite inside venv
	@.venv/bin/python -m pytest tools/test_lab_memory.py -v

.PHONY: lab-memory-install
lab-memory-install: ## Install sqlite-vec + sentence-transformers + rank_bm25 into project venv
	@if [ ! -x .venv/bin/python ]; then \
		echo "No .venv/ found. Creating one with python3.14 (Homebrew)."; \
		python3.14 -m venv .venv 2>/dev/null || python3 -m venv .venv; \
	fi
	@.venv/bin/python -m pip install --upgrade pip >/dev/null
	@.venv/bin/python -m pip install sqlite-vec sentence-transformers rank_bm25==0.2.2
	@echo "=== lab_memory deps installed in .venv/ ==="

# ---- Memory tool (tools/memory.py) ----
.PHONY: memory-audit memory-index memory-rotate-log memory-test

memory-audit:
	.venv/bin/python3 tools/memory.py audit

memory-index:
	.venv/bin/python3 tools/memory.py index

memory-rotate-log:
	.venv/bin/python3 tools/memory.py rotate-log --cap-kb 30

memory-test:
	.venv/bin/pytest tests/test_memory.py tests/test_migrate_to_memories.py -v

# ---- Shell-level CLI guardrail tests (P-GUARDRAIL-SHELL-TEST, D-120) ----
# Verifies binary end-to-end behavior of scale_experiment CLI guardrails and
# run_long.py launcher paths. NOT added to default `test` target yet —
# Director to promote after D-120 review.
.PHONY: test-shell-guardrails
test-shell-guardrails: ## Run shell-level CLI guardrail tests (see tests/shell/)
	bash tests/shell/run_all.sh


# ---- IEEE PDF build (Phase-15 paper output) ----
.PHONY: ieee-pdf

ieee-pdf: ## Build IEEE-conference PDF from a markdown paper. Usage: make ieee-pdf PAPER=path/to/paper.md
	@if [ -z "$(PAPER)" ]; then \
		echo "Usage: make ieee-pdf PAPER=programs/<dir>/paper_draft_v2.md [KEEP_TEX=1]"; \
		exit 1; \
	fi
	@.venv/bin/python3 tools/build_ieee_pdf.py "$(PAPER)" $(if $(KEEP_TEX),--keep-tex)

.PHONY: lab-report
lab-report: ## Generate the lab self-assessment report (data/eval/lab_assessment.{md,json})
	PYTHONPATH=. .venv/bin/python -m tools.lab_assessment --repo .
