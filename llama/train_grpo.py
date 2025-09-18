# train_grpo.py
import re
import gc
import torch
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
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
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
        score = 0
        response = completion[0]["content"]
        # Match if format is seen exactly!
        if match_format.search(response) is not None: score += 3.0
        scores.append(score)
    return scores

"""If it fails, we want to reward the model if it at least follows the format partially, by counting each symbol:"""

def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Count how many keywords are seen - we penalize if too many!
        # If we see 1, then plus some points!
        score += 0.5 if response.count(reasoning_start) == 1 else -1.0
        score += 0.5 if response.count(reasoning_end)   == 1 else -1.0
        score += 0.5 if response.count(solution_start)  == 1 else -1.0
        score += 0.5 if response.count(solution_end)    == 1 else -1.0
        scores.append(score)
    return scores

"""Finally, we want to extract the generated answer, and reward or penalize it! We also reward it based on how close the answer is to the true one via ratios:"""

def check_answer(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1)
        if (guess := match_format.search(r)) is not None else None \
        for r in responses
    ]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        if guess is None:
            scores.append(0)
            continue
        # Correct answer gets 3 points!
        if guess == true_answer:
            score += 3.0
        # Match if spaces are seen, but less reward
        elif guess.strip() == true_answer.strip():
            score += 1.5
        else:
            # We also reward it if the answer is close via ratios!
            # Ie if the answer is within some range, reward it!
            try:
                ratio = float(guess) / float(true_answer)
                if   ratio >= 0.9 and ratio <= 1.1: score += 1.0
                elif ratio >= 0.8 and ratio <= 1.2: score += 0.5
                else: score -= 1.5 # Penalize wrong answers
            except:
                score -= 1.5 # Penalize
        scores.append(score)
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
            scores.append(0)
            continue
        # Convert to numbers
        try:
            true_answer = float(true_answer.strip())
            # Remove commas like in 123,456
            guess       = float(guess.strip().replace(",", ""))
            scores.append(1.5 if guess == true_answer else -0.5)
        except:
            scores.append(0)
            continue
    return scores

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
    learning_rate=1e-5,
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
    max_steps=500,  # Set max_steps for quicker testing
    save_steps=100, # changed for testing
    max_grad_norm=1.0,
    report_to="wandb",
    log_on_each_node=False,
    beta=0.0,
)


# Trainer setup
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs = [
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
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
