# train_grpo.py
import re
import gc
import sys
from pathlib import Path
from typing import Callable, List

import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
import os
import logging
import wandb
import weave


_THIS_FILE = Path(__file__).resolve()
for parent in _THIS_FILE.parents:
    bin_dir = parent / "bin"
    candidate = bin_dir / "training_monitor.py"
    if candidate.exists():
        if str(parent) not in sys.path:
            sys.path.append(str(parent))
        if str(bin_dir) not in sys.path:
            sys.path.append(str(bin_dir))
        break
else:  # fallback to immediate parent to preserve previous behaviour
    fallback_root = _THIS_FILE.parents[1]
    if str(fallback_root) not in sys.path:
        sys.path.append(str(fallback_root))
    fb_bin = fallback_root / "bin"
    if fb_bin.exists() and str(fb_bin) not in sys.path:
        sys.path.append(str(fb_bin))

from training_monitor import StepSystemStatsCallback

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import TransformersModel, TransformersModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

# Load HF token from .env file 
from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid float for {name}: {raw}") from exc

def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid int for {name}: {raw}") from exc

# ============================================================================
# Validate environment variables
# ============================================================================

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
if not MODEL_NAME:
    raise ValueError("MODEL_NAME environment variable is not set.")

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs/LoRA-GRPO")
if not OUTPUT_DIR:
    raise ValueError("OUTPUT_DIR environment variable is not set.")

RUN_NAME = os.getenv("RUN_NAME", "LoRA-GRPO-gsm8k")
if not RUN_NAME:
    raise ValueError("RUN_NAME environment variable is not set.")

EVAL_OUTPUT_DIR = os.getenv("EVAL_OUTPUT_DIR", f"{OUTPUT_DIR}/evaluation_results")

EVAL = os.getenv("EVAL", "NONE")


# ============================================================================
# Load model and tokenizer
# ============================================================================

try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        device_map="auto"
    ).to("cuda")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# PEFT config (optional)
LORA_R = _get_env_int("LORA_R", 16)

peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_R,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    # task_type="CAUSAL_LM",
    lora_dropout=0.05,
)



# # ============================================================================
# # Reward function and dataset configuration
# # ============================================================================
def extract_hash_answer(text):
    if "####" not in text: return None
    return text.split("####")[1].strip()

"""We now create a system prompt which can be customized. We add 4 extra symbols for working out or thinking / reasoning sections and a final answer:"""

reasoning_start = "<start_working_out>"
reasoning_end   = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

match_format = re.compile(
    rf"^[\s]{{0,}}"\
    rf"{reasoning_start}.+?{reasoning_end}.*?"\
    rf"{solution_start}(.+?){solution_end}"\
    rf"[\s]{{0,}}$",
    flags = re.MULTILINE | re.DOTALL
)

system_prompt = \
f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""


def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        strict_match = match_format.search(response) is not None
        has_reasoning = reasoning_start in response and reasoning_end in response
        has_solution = solution_start in response and solution_end in response

        if strict_match:
            scores.append(2.0)
        elif has_reasoning and has_solution:
            # Near-perfect shaping: correct sections but spacing may be off
            scores.append(0.4)
        elif has_reasoning or has_solution:
            scores.append(0.15)
        else:
            scores.append(-0.4)
    return scores

"""If it fails, we want to reward the model if it at least follows the format partially, by counting each symbol:"""

def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        total = 0.0
        for token in (reasoning_start, reasoning_end, solution_start, solution_end):
            count = response.count(token)
            if count == 1:
                total += 0.5
            elif count > 1:
                total += max(0.2 - 0.1 * (count - 2), -0.4)
            else:
                total -= 0.5

        if solution_end in response:
            tail = response.split(solution_end, 1)[-1]
            if tail.strip():
                total -= min(len(tail.strip()) * 0.002, 0.4)

        scores.append(total)
    return scores

"""Finally, we want to extract the generated answer, and reward or penalize it! We also reward it based on how close the answer is to the true one via ratios:"""

def check_answer(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    def _extract_candidate(text: str) -> str:
        match = match_format.search(text)
        if match is not None:
            return match.group(1).strip()

        alt = extract_hash_answer(text)
        if alt:
            return alt.strip()

        if solution_end in text:
            tail = text.split(solution_end, 1)[-1]
            tail = tail.strip()
            if tail:
                return tail.splitlines()[0].strip()

        return text.rsplit("####", 1)[-1].strip() if "####" in text else text.rsplit("\n", 1)[-1].strip()

    def _normalized(text: str) -> str:
        return re.sub(r"\s+", "", text).lower().strip(".,;:!?")

    def _parse_float(text: str):
        cleaned = text.replace(",", "").replace("%", "").strip()
        match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return None
        return None

    scores: List[float] = []
    for response, true_answer in zip(responses, answer):
        candidate = _extract_candidate(response)
        target_clean = (true_answer or "").strip()

        if not candidate:
            scores.append(-0.2)
            continue

        guess_norm = _normalized(candidate)
        target_norm = _normalized(target_clean)

        if not target_norm:
            scores.append(0.0)
            continue

        if guess_norm == target_norm:
            scores.append(3.0)
            continue

        if target_norm in guess_norm or guess_norm in target_norm:
            scores.append(2.0)
            continue

        guess_float = _parse_float(candidate)
        target_float = _parse_float(target_clean)
        if guess_float is not None and target_float is not None:
            diff = abs(guess_float - target_float)
            denom = max(1.0, abs(target_float))
            rel_err = diff / denom
            if rel_err <= 0.02:
                scores.append(1.8)
                continue
            if rel_err <= 0.05:
                scores.append(1.2)
                continue
            if rel_err <= 0.15:
                scores.append(0.6)
                continue

        digits_guess = re.sub(r"\D", "", candidate)
        digits_target = re.sub(r"\D", "", target_clean)
        if digits_guess and digits_target and digits_guess == digits_target:
            scores.append(0.5)
            continue

        if target_clean and target_clean.lower() in candidate.lower():
            scores.append(0.3)
            continue

        scores.append(-0.3)

    return scores

"""Also sometimes it might not be 1 number as the answer, but like a sentence for example "The solution is $20" -> we extract 20.

We also remove possible commas for example as in 123,456
"""
match_numbers = re.compile(
    solution_start + r".*?([\d\.\,]{1,})",
    flags = re.MULTILINE | re.DOTALL
)

global PRINTED_TIMES
PRINTED_TIMES = 0
global PRINT_EVERY_STEPS
PRINT_EVERY_STEPS = 5

def check_numbers(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1)
        if (guess := match_numbers.search(r)) is not None else None \
        for r in responses
    ]

    scores = []
    # Print only every few steps
    global PRINTED_TIMES
    global PRINT_EVERY_STEPS
    if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
        print('*'*20, f"Question:\n{question}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    PRINTED_TIMES += 1

    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(-0.2)
            continue
        try:
            target = float(true_answer.strip().replace(",", ""))
            value = float(guess.strip().replace(",", ""))
            diff = abs(value - target)
            denom = max(1.0, abs(target))
            rel_err = diff / denom
            if rel_err <= 0.02:
                scores.append(1.5)
            elif rel_err <= 0.08:
                scores.append(1.0)
            elif rel_err <= 0.2:
                scores.append(0.5)
            else:
                scores.append(-0.5)
        except Exception:
            scores.append(0.0)
    return scores


def scale_reward(reward_fn: Callable, factor: float) -> Callable:
    def wrapper(*args, **kwargs):
        base = reward_fn(*args, **kwargs)
        return [factor * value for value in base]

    return wrapper


match_format_exactly_weighted = scale_reward(match_format_exactly, 0.6)
check_answer_emphasized = scale_reward(check_answer, 1.3)

"""Get the maximum prompt length so we don't accidentally truncate it!"""

dataset = load_dataset("openai/gsm8k", "main", split = "train")

dataset = dataset.map(lambda x: {
    "prompt" : [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": x["question"]},
    ],
    "answer": extract_hash_answer(x["answer"]),
})


max(dataset.map(
    lambda x: {"tokens" : tokenizer.apply_chat_template(x["prompt"], add_generation_prompt = True, tokenize = True)},
    batched = True,
).map(lambda x: {"length" : len(x["tokens"])})["length"])

# ============================================================================
# Training configuration
# ============================================================================

# Training config
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    run_name=RUN_NAME,
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=False,
    fp16=True,
    per_device_train_batch_size=4,  
    gradient_accumulation_steps=8,
    num_generations=16,  # Reduced from 16
    max_prompt_length=256,
    max_completion_length=786,
    #num_train_epochs=1, # comment out or overriden by setting max_steps 
    max_steps=300,  # Set max_steps for quicker testing
    save_steps=100, # changed for testing
    max_grad_norm=1.0,
    report_to="wandb",
    log_on_each_node=False,
    beta=_get_env_float("GRPO_BETA", 0.05),
    temperature=0.6,
    top_p=0.85,
    top_k=20,
    repetition_penalty=1.05,
)


# Trainer setup
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs = [
        match_format_exactly_weighted,
        match_format_approximately,
        check_answer_emphasized,
        check_numbers,
    ],
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
    callbacks=[StepSystemStatsCallback()],
)

# ============================================================================
# Training model 
# ============================================================================

try:
    trainer.train()
except Exception as e:
    logger.error(f"Training failed: {e}")
    raise

# ====================================================================
# Evaluation after training
# ====================================================================
logger.info("Starting evaluation...")

# Merge LoRA weights back into base model for evaluation
logger.info("Merging LoRA weights into base model...")
merged_model = trainer.model.merge_and_unload()

# ======================================================
# Save merged model to launch from cli
# =====================================================
if EVAL == "CLI":
    logger.info(f"Saving merged model to {OUTPUT_DIR}/merged_model")
    merged_model.save_pretrained(f"{OUTPUT_DIR}/merged_model")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/merged_model")

elif EVAL == "API":

    # ======================================================
    # Launch evaluation from api 
    # =====================================================
    print("Launching evaluation from API...")

    tasks = ["lighteval|math_500|0|0","lighteval|gpqa:diamond|0|0"]
    for task in tasks:
        logger.info(f"Evaluating task: {task}")
        trainer.accelerator.free_memory()   # releases gradient shards/optim state
        del trainer.model, trainer.optimizer, trainer.lr_scheduler
        gc.collect()
        torch.cuda.empty_cache()

        evaluation_tracker = EvaluationTracker(output_dir="./evaluation_resultsls")

        pipeline_params = PipelineParameters(
            launcher_type=ParallelismManager.NONE,
            use_chat_template=True
        )

        config = TransformersModelConfig(model_name=MODEL_NAME, batch_size=1,generation_parameters={"max_new_tokens":512}, max_length=512)

        lighteval_model = TransformersModel.from_model(merged_model, config)


        pipeline = Pipeline(
            model=lighteval_model,
            pipeline_parameters=pipeline_params,
            evaluation_tracker=evaluation_tracker,
            tasks=task,  
        )

        results = pipeline.evaluate()
        pipeline.show_results()
        detailed_results = pipeline.get_results()

else:
    print("EVAL variable not set correctly, skipping evaluation.")
    
