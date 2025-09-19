#!/usr/bin/env bash
# Usage:  bin/run_eval.sh <MODEL_DIR> <OUT_DIR>
set -euo pipefail

MODEL_DIR=${1:?first arg = fused-model path}
OUT_DIR=${2:?second arg = output dir for lighteval results}

MODEL_ARGS="model_name=$MODEL_DIR,dtype=float16,max_length=4096,batch_size=1,\
generation_parameters={max_new_tokens:2048,temperature:0.6,top_p:0.95}"

mkdir -p "$OUT_DIR"

run_task() {
    local task=$1
    lighteval accelerate $MODEL_ARGS "lighteval|$task|0|0" \
        --use-chat-template --output-dir "$OUT_DIR/$task"
}

# ---------- MATH-500 ----------
run_task math_500

# ---------- GPQA Diamond ----------
run_task gpqa:diamond
