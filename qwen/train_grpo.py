# train_grpo.py
import re
import torch
import gc
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
import os
import logging
import wandb
import weave

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
# Reward function and dataset configuration
# ============================================================================

# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format:

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    """Extracts the answer from XML-formatted text."""
    try:
        answer = text.split("<answer>")[-1].split("</answer>")[0].strip()
        return answer
    except IndexError:
        logger.warning("Failed to extract answer from XML format.")
        return ""

def extract_hash_answer(text: str) -> str | None:
    """Extracts the answer from a hash-formatted string."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def _to_number(candidate: str | None) -> float | None:
    """Lenient numeric parsing used by rewards to reduce sparsity."""
    if candidate is None:
        return None
    s = candidate.strip().replace(",", "")
    if not s:
        return None
    # Simple fraction support like 3/4
    if "/" in s:
        a, b = s.split("/", 1)
        try:
            return float(a) / float(b)
        except Exception:
            pass
    try:
        return float(s)
    except Exception:
        return None



# Configurable one-shot prompting
def get_gsm8k_questions(split="train", use_one_shot=False) -> Dataset:
    """Loads and prepares the GSM8K dataset with optional one-shot prompting."""
    try:
        data = load_dataset('openai/gsm8k', 'main')[split]
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    def format_example(x):
        prompt = [{'role': 'system', 'content': SYSTEM_PROMPT}]
        if use_one_shot:
            prompt.extend([
                {'role': 'user', 'content': 'What is the largest single-digit prime number?'},
                {'role': 'assistant', 'content': XML_COT_FORMAT.format(
                    reasoning="9 is divisible by 3 and 8 is divisible by 2, but 7 is prime.",
                    answer="7"
                )}
            ])
        prompt.append({'role': 'user', 'content': x['question']})
        return {'prompt': prompt, 'answer': extract_hash_answer(x['answer'])}

    return data.map(format_example)

dataset = get_gsm8k_questions(use_one_shot=True)

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward exact matches, with smooth fallback on numeric closeness.

    Keeps the name and intent of the existing reward while reducing sparsity
    for cases where the predicted number is near the gold answer.
    """
    golds = answer if isinstance(answer, list) else [answer] * len(completions)
    preds = [extract_xml_answer(c[0]['content']) for c in completions]

    rewards: list[float] = []
    for pred, gold in zip(preds, golds):
        # Exact string match first
        if pred.strip() == (gold or "").strip():
            rewards.append(2.0)
            continue
        # Numeric partial credit
        p = _to_number(pred)
        g = _to_number(gold)
        if p is None or g is None:
            rewards.append(0.0)
            continue
        rel_err = abs(p - g) / max(1.0, abs(g))
        # full credit at 0 err -> 2.0; 0 at >= 50% rel error
        shaped = max(0.0, 1.0 - (rel_err / 0.5)) * 2.0
        rewards.append(shaped)
    return rewards

def int_reward_func(completions, **kwargs) -> list[float]:
    """Reward if the extracted response parses as a number (lenient)."""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    scores = []
    for r in extracted_responses:
        val = _to_number(r)
        scores.append(0.5 if val is not None else 0.0)
    return scores

def format_reward_func(completions, strict=False, **kwargs) -> list[float]:
    """Reward well-formed outputs with partial credit when both sections exist.

    - 0.5 if matches the canonical strict multi-line block
    - 0.25 if both <reasoning>…</reasoning> and <answer>…</answer> are present (any spacing)
    - 0.0 otherwise
    """
    strict_pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    loose_has_both = r"<reasoning>.*?</reasoning>.*?<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    scores = []
    for r in responses:
        if re.match(strict_pattern, r, flags=re.DOTALL):
            scores.append(0.5)
        elif re.search(loose_has_both, r, flags=re.DOTALL):
            scores.append(0.25)
        else:
            scores.append(0.0)
    return scores

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """Calculates reward based on XML tag counts."""
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def count_xml(text) -> float:
    """Counts XML tags and penalizes extra content."""
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count

# ============================================================================
# Model and training configuation
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
peft_config = LoraConfig(
    r=_get_env_int("LORA_R", 16),
    lora_alpha=_get_env_int("LORA_ALPHA", 16),
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    # task_type="CAUSAL_LM",
    lora_dropout=0.05,
)

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
    max_steps=1000,  # Set max_steps for quicker testing
    save_steps=100, # changed for testing
    max_grad_norm=1.0,
    report_to="wandb",
    log_on_each_node=False,
    beta=_get_env_float("GRPO_BETA", 0.0),
    temperature=0.6,
    top_p=0.85,
    top_k=20,
    repetition_penalty=1.05,
)

# Trainer setup
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        format_reward_func,  # No need for lambda, just pass the function
        int_reward_func,
        correctness_reward_func
    ],
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config  # Uncomment if PEFT is working for you
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
    
