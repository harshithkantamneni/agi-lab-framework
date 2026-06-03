#!/usr/bin/env bash
# Live progress bar for the Phase 2 MoE Rev-2 training run.
# Usage: bash tools/moe_progress.sh   (Ctrl-C to stop)

LOG=data/runs/moe_rev2/stdout.log
CKPT_DIR=data/checkpoints/phase2_moe_rev2
TOTAL=5000
BAR_LEN=40
PID=$(cat data/runs/moe_rev2/pid 2>/dev/null || echo 58700)

while true; do
    clear
    now=$(date "+%H:%M:%S")
    step=$(grep -E "\[entpen\] step=" "$LOG" 2>/dev/null | tail -1 | grep -oE "step=[0-9]+" | cut -d= -f2)
    step=${step:-0}
    pct=$((step * 100 / TOTAL))
    filled=$((step * BAR_LEN / TOTAL))
    bar=""
    for ((i=0; i<filled; i++)); do bar+="█"; done
    for ((i=filled; i<BAR_LEN; i++)); do bar+="░"; done

    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Phase 2 — MoE Rev-2 training                      $now  ║"
    echo "╠══════════════════════════════════════════════════════════════╣"
    printf "║  [%s] %3d%%            ║\n" "$bar" "$pct"
    printf "║  step %5d / %5d                                          ║\n" "$step" "$TOTAL"
    echo "╠══════════════════════════════════════════════════════════════╣"

    # Process status
    if ps -p "$PID" > /dev/null 2>&1; then
        ps_line=$(ps -p "$PID" -o etime=,stat=,%cpu=,rss= 2>/dev/null)
        etime=$(echo "$ps_line" | awk '{print $1}')
        stat=$(echo "$ps_line" | awk '{print $2}')
        cpu=$(echo "$ps_line" | awk '{print $3}')
        rss_kb=$(echo "$ps_line" | awk '{print $4}')
        rss_mb=$((rss_kb / 1024))
        printf "║  pid %-6s  %s  etime %8s  cpu %5s%%  mem %4d MB  ║\n" "$PID" "$stat" "$etime" "$cpu" "$rss_mb"
    else
        echo "║  PROCESS DEAD — check for resume via 'make run-phase2-pair'  ║"
    fi

    # Latest checkpoint
    latest_ckpt=$(ls -t "$CKPT_DIR"/step_*.ckpt 2>/dev/null | head -1 | xargs basename 2>/dev/null)
    latest_ckpt=${latest_ckpt:-none}
    ckpt_age=$(ls -t "$CKPT_DIR"/step_*.ckpt 2>/dev/null | head -1 | xargs -I{} stat -f "%Sm" -t "%H:%M" {} 2>/dev/null)
    ckpt_age=${ckpt_age:-""}
    printf "║  last ckpt: %-20s  @ %s%-20s  ║\n" "$latest_ckpt" "$ckpt_age" ""

    # ETA based on recent step times
    if [ "$step" -gt 0 ]; then
        remaining=$((TOTAL - step))
        # Recent step_time from log (last ~5 steps)
        step_time=$(grep -E "step_time=" "$LOG" 2>/dev/null | tail -5 | grep -oE "step_time=[0-9.]+" | awk -F= '{s+=$2; n++} END{if(n>0) print s/n; else print 2.6}')
        step_time=${step_time:-2.6}
        eta_seconds=$(awk "BEGIN{print int($remaining * $step_time)}")
        eta_min=$((eta_seconds / 60))
        eta_h=$((eta_min / 60))
        eta_m=$((eta_min % 60))
        printf "║  step_time ~%s s    ETA  %dh %02dm  (%d steps left)         ║\n" "$step_time" "$eta_h" "$eta_m" "$remaining"
    fi

    echo "╠══════════════════════════════════════════════════════════════╣"

    # Battery
    batt_line=$(pmset -g batt | grep -oE "[0-9]+%;[^;]*;[^ ]* remaining" | head -1)
    batt_pct=$(echo "$batt_line" | grep -oE "^[0-9]+%" | tr -d %)
    batt_state=$(pmset -g batt | head -1 | sed "s/.*'\(.*\)'.*/\1/")
    batt_remain=$(echo "$batt_line" | grep -oE "[0-9]+:[0-9]+ remaining")
    if [ -z "$batt_remain" ]; then
        batt_remain=$(pmset -g batt | grep -oE "[0-9]+:[0-9]+ remaining" | head -1)
    fi
    batt_remain=${batt_remain:-"(calculating)"}
    printf "║  battery: %3s%%  %-20s %-18s  ║\n" "${batt_pct:-??}" "${batt_state:-unknown}" "$batt_remain"

    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  Dense-A (done):  PPL 1.90, final.ckpt saved"
    echo "  Refresh every 5s. Ctrl-C to stop."

    sleep 5
done
