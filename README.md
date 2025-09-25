You won’t see big VRAM changes with LoRA, but you can still expose the KL cost clearly by measuring time and utilization. I added two upgrades to make this easy:

GPU utilization/power logging

StepSystemStatsCallback now logs, per GPU: gpu/N/utilization_pct, gpu/N/memory_util_pct, gpu/N/power_w, gpu/N/temp_c (if pynvml is available). These appear alongside step/duration_sec and memory stats.
Optional dependency: pip install nvidia-ml-py3
KL timing breakdown

KLProfilerCallback now reports kl_profiler/policy_forward_ms, kl_profiler/ref_forward_ms, and kl_profiler/softmax_kl_ms, plus kl_profiler/duration_ms. This isolates the extra ref forward and KL compute time.
How to show the KL cost with PEFT

Compare two matched runs (same seed, batch, lengths):

Baseline: GRPO_BETA=0, KL_PROFILER=0
KL on: GRPO_BETA=1e-6 (or 1e-5), KL_PROFILER=0
Inspect median step/duration_sec after warmup (ignore first ~20 steps). The KL run will be slower per step; that delta is the cost. Memory won’t change much with LoRA.
Visualize utilization and power (optional):

Install nvidia-ml-py3, rerun, then compare gpu/0/utilization_pct and gpu/0/power_w across the same two runs. KL should raise average utilization/power over time even if peak memory is flat.
Break down KL overhead independently:

Enable the profiler briefly: KL_PROFILER=1 KL_FREQ=20 KL_BATCH=2 KL_MAX_TOKENS=256
Watch kl_profiler/ref_forward_ms vs kl_profiler/policy_forward_ms and kl_profiler/softmax_kl_ms. That shows what KL adds on top of the main forward/backward.
Make the comparison fair

Align generation and accumulation so the baseline doesn’t pay extra compute: ensure gradient_accumulation_steps % (steps_per_generation * num_iterations) == 0.
Keep max_prompt_length, max_completion_length, num_generations identical between runs.
Use the same model dtype and attention implementation.
If you want an obvious VRAM delta just for illustration, do a very short run with PEFT disabled (no LoRA) and beta>0; TRL will load a separate ref model and you’ll see VRAM jump. For normal LoRA training though, focus on time/utilization/power — that’s where the KL cost shows.




Modified TRL package:
@ 
.venv/lib64/python3.13/site-packages/trl/trainer/grpo_trainer.py:420

Original: 
self.beta = args.beta
if self.beta == 0.0:
    self.ref_model = None
elif is_peft_model(model):
    self.ref_model = None
else:
    # build ref model from model_id ...


Modified to force ref when peft is loaded:
self.beta = args.beta
force_ref = os.getenv("TRL_FORCE_REF_MODEL", "0") in {"1","true","yes"}
if self.beta == 0.0:
    self.ref_model = None
elif is_peft_model(model) and not force_ref:
    self.ref_model = None
else:
    # build ref model from model_id ...
