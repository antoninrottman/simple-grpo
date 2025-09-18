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
    """Calculates reward based on correctness of the response."""
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    logger.info(f"Question:\n{q}\nAnswer:\n{answer[0]}\nResponse:\n{responses[0]}\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    """Calculates reward if the extracted response is a digit."""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def format_reward_func(completions, strict=False, **kwargs) -> list[float]:
    """Calculates reward based on XML formatting."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$" if strict else r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

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
    r=16,
    lora_alpha=64,
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
    per_device_train_batch_size=8,  
    gradient_accumulation_steps=4,
    num_generations=16,  # Reduced from 16
    max_prompt_length=256,
    max_completion_length=786,
    #num_train_epochs=1, # comment out or overriden by setting max_steps 
    max_steps=1,  # Set max_steps for quicker testing
    save_steps=1, # changed for testing
    max_grad_norm=1.0,
    report_to="wandb",
    log_on_each_node=False,
    beta=0.0,
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

trainer.accelerator.free_memory()   # releases gradient shards/optim state
del trainer.model, trainer.optimizer, trainer.lr_scheduler
gc.collect()
torch.cuda.empty_cache()

evaluation_tracker = EvaluationTracker(output_dir="./results")

pipeline_params = PipelineParameters(
    launcher_type=ParallelismManager.NONE,
    use_chat_template=True
)

config = TransformersModelConfig(model_name=MODEL_NAME, batch_size=1,generation_parameters={"max_new_tokens":512}, max_length=512)

lighteval_model = TransformersModel.from_model(merged_model, config)

# Run the exact tasks you listed
tasks = "lighteval|math_500|0|0,lighteval|gpqa:diamond|0|0"


pipeline = Pipeline(
    model=lighteval_model,
    pipeline_parameters=pipeline_params,
    evaluation_tracker=evaluation_tracker,
    tasks=tasks,  # list of well-formed task strings
)

results = pipeline.evaluate()
pipeline.show_results()
detailed_results = pipeline.get_results()


# Log evaluation results to wandb
import wandb
if wandb.run is not None:
    for task_name, task_results in detailed_results.items():
        if isinstance(task_results, dict):
            for metric_name, metric_value in task_results.items():
                if isinstance(metric_value, (int, float)):
                    wandb.log({f"eval/{task_name}/{metric_name}": metric_value})
    logger.info("Evaluation results logged to wandb")
