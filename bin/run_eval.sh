#!/usr/bin/env bash
# Usage:  bin/run_eval.sh <MODEL_DIR> <OUT_DIR>
set -euo pipefail

export VLLM_WORKER_MULTIPROC_METHOD=spawn   # required by vLLM

MODEL_DIR=${1:?first arg = fused-model path}
OUT_DIR=${2:?second arg = output dir for lighteval results}

MODEL_ARGS="model_name=$MODEL_DIR,dtype=float16,max_model_length=4096,\
max_num_batched_tokens=4096,gpu_memory_utilization=0.8,\
generation_parameters={max_new_tokens:2048,temperature:0.6,top_p:0.85}"

mkdir -p "$OUT_DIR"


# ---------- MATH-500 ----------
TASK=math_500
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template --output-dir "$OUT_DIR/$TASK"

# ---------- GPQA Diamond ----------
TASK=gpqa:diamond
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template --output-dir "$OUT_DIR/$TASK"

# ---------- GSM8K ----------
TASK=gsm8k
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template --output-dir "$OUT_DIR/$TASK"

