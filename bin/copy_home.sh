SWEEP=results_run_20250926-083327
RUN=grpo_Qwen-Qwen2.5-3B-Instruct_b0__
r4
MODEL_KEY=Qwen-Qwen2.5-3B-Instruct
SCRATCH=/scratch/izar/rottman/simple-grpo/$SWEEP/$RUN/staging
DEST=$HOME/simple-grpo/outputs/$SWEEP/$MODEL_KEY/$RUN

mkdir -p "$DEST/lora_adapters" "$DEST/evaluation_results"
rsync -avh --delete "$SCRATCH/lora_adapters/" "$DEST/lora_adapters/"
rsync -avh --delete "$SCRATCH/evaluation_results/" "$DEST/evaluation_results/"
cp "$SCRATCH/run_metadata.json" "$DEST/run_metadata.json"
echo "copied $RUN back home"