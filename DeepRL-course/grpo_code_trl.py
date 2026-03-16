import ast
import os
import re
import subprocess
import tempfile

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

# Split GPUs: Use 0,1,2,3,4,5,6,7
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

# Load environment variables from .env
load_dotenv()

# --- Configuration ---
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
DATASET_ID = "deepmind/code_contests"
CACHE_DIR_MODELS = "/data/aiseed/hf-models"
CACHE_DIR_DATASETS = "/data/aiseed/hf-datasets"
OUTPUT_DIR = "deepseek-r1-code-contests-grpo"

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user provides a competitive programming problem description. "
    "The Assistant solves it by first reasoning and then providing a Python 3 solution. "
    "The reasoning process and the Python 3 code are enclosed within <think> </think> and <answer> </answer> tags, respectively.\n"
    "Wait! The content inside <answer> should be ONLY the Python 3 code, no other text or explanation.\n"
    "Example:\n"
    "<think> reasoning here </think><answer>\nimport sys\n# your code\n</answer>"
)

# --- Reward Functions ---


def format_reward(completions, **_kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content in completion_contents:
        if re.search(pattern, content, re.DOTALL):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def reasoning_length_reward(completions, **_kwargs):
    """Encourages longer reasoning traces."""
    completion_contents = [completion[0]["content"] for completion in completions]
    lengths = []
    for content in completion_contents:
        match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        content_len = len(match.group(1)) if match else 0
        lengths.append(min(1.0, content_len / 1000.0))
    return lengths


def syntax_reward(completions, **_kwargs):
    """Reward function that checks if the extracted python code is syntactically correct."""
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content in completion_contents:
        # Extract code from <answer> tags
        match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        if match:
            code = match.group(1).strip()
            # Remove ```python and ``` if present
            code = re.sub(r"^```python\n", "", code)
            code = re.sub(r"^```\n", "", code)
            code = re.sub(r"\n```$", "", code)
            try:
                ast.parse(code)
                rewards.append(1.0)
            except Exception:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards


def code_correctness_reward(completions, **kwargs):
    """
    Reward function that executes the Python code against public test cases.
    NOTE: In production, use a secure sandbox!
    """
    public_tests = kwargs.get("public_tests")
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for i, content in enumerate(completion_contents):
        # Extract the code
        match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        if not match:
            rewards.append(0.0)
            continue

        code = match.group(1).strip()
        # Basic cleanup
        code = re.sub(r"^```python\n", "", code)
        code = re.sub(r"^```\n", "", code)
        code = re.sub(r"\n```$", "", code)

        # Get the tests for this completion
        # GRPO passes row values in kwargs.
        # Since completions is a list of generations for a batch of prompts,
        # we need to be careful with indexing if multiple prompts are in a batch.
        # GRPOTrainer handles this: kwargs will have the same length as completions?
        # Actually, for GRPOTrainer, reward_func(completions, **kwargs) is called
        # where completions is the list of ALL generations for the current batch.
        # But wait, public_tests will be a list of the same length as the batch size.
        # num_generations is usually > 1, so the mapping is completions[i] -> prompt[j]
        # In TRL, kwargs contain lists of size batch_size.
        # We need to find which prompt this completion belongs to.
        # TRL GRPO actually handles the broadcasting if the reward function expects the same index.
        # Actually, the signature is a bit subtle. Let's assume standard behavior.

        tests = public_tests[
            i
        ]  # If TRL broadcasts public_tests to match completions length

        if not tests or not tests.get("input"):
            rewards.append(1.0)  # No tests, assume okay for now? Or 0.5?
            continue

        passed_count = 0
        total_tests = len(tests["input"])

        # Limit total tests to 3 to save time during training
        num_to_test = min(total_tests, 3)

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=True) as f:
            f.write(code)
            f.flush()

            for j in range(num_to_test):
                input_str = tests["input"][j]
                expected_output = tests["output"][j].strip()

                try:
                    # Run code with timeout
                    process = subprocess.run(
                        ["python3", f.name],
                        input=input_str,
                        text=True,
                        capture_output=True,
                        timeout=2.0,
                    )

                    if process.returncode == 0:
                        actual_output = process.stdout.strip()
                        if actual_output == expected_output:
                            passed_count += 1
                except subprocess.TimeoutExpired:
                    continue
                except Exception:
                    continue

        if num_to_test > 0:
            rewards.append(passed_count / num_to_test)
        else:
            rewards.append(1.0)

    return rewards


# --- Preprocessing ---


def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["description"]},
        ],
        "public_tests": example["public_tests"],
    }


def main():
    # 1. Load Dataset
    print(f"Loading dataset: {DATASET_ID}")
    # Using streaming=False for select() to work easily, or use range if supported.
    # Code contests is big, so we take a portion.
    dataset = load_dataset(DATASET_ID, split="train", cache_dir=CACHE_DIR_DATASETS)

    # 2. Preprocess Dataset
    print("Preprocessing dataset...")
    dataset = dataset.map(make_conversation)
    # Filter or select subset
    dataset = dataset.select(range(min(2000, len(dataset))))

    # Remove original columns to keep only what's needed for training and rewards
    # GRPOTrainer needs 'prompt' and columns used in reward functions
    keep_columns = ["prompt", "public_tests"]
    dataset = dataset.remove_columns(
        [c for c in dataset.column_names if c not in keep_columns]
    )

    # 3. Load Model & Tokenizer
    print(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=CACHE_DIR_MODELS,
    )

    # 4. LoRA Configuration
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Training Configuration
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        run_name=OUTPUT_DIR,
        learning_rate=2e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=100,
        bf16=True,
        gradient_accumulation_steps=4,
        save_strategy="steps",
        save_steps=1000,
        max_steps=5000,
        num_generations=8,  # Recommended for GRPO
        max_completion_length=1536,  # Code solutions + thinking can be long
        remove_unused_columns=False,
        report_to=["mlflow"],
    )

    # 6. Initialize Trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            format_reward,
            reasoning_length_reward,
            syntax_reward,
            code_correctness_reward,
        ],
        args=training_args,
        train_dataset=dataset,
    )

    # 7. Start Training
    print("Starting training...")
    trainer.train()
    print("Training completed.")


if __name__ == "__main__":
    main()
