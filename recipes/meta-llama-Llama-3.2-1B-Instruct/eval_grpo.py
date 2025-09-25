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

MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct")
if not MODEL_NAME:
    raise ValueError("MODEL_NAME environment variable is not set.")

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs/LoRA-GRPO")
if not OUTPUT_DIR:
    raise ValueError("OUTPUT_DIR environment variable is not set.")

RUN_NAME = os.getenv("RUN_NAME", "LoRA-GRPO-gsm8k")
if not RUN_NAME:
    raise ValueError("RUN_NAME environment variable is not set.")

EVAL_OUTPUT_DIR = os.getenv("EVAL_OUTPUT_DIR", f"{OUTPUT_DIR}/evaluation_results")

evaluation_tracker = EvaluationTracker(output_dir="./results")

pipeline_params = PipelineParameters(
    launcher_type=ParallelismManager.NONE,
    use_chat_template=True
)


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

config = TransformersModelConfig(model_name=MODEL_NAME, batch_size=1,generation_parameters={"max_new_tokens":512}, max_length=512)

lighteval_model = TransformersModel.from_model(model, config)

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
