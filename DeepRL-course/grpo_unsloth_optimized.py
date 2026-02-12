import re
import os

# Split GPUs: Use 4,5,6,7 for Unsloth experiment
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from unsloth import FastLanguageModel
from dotenv import load_dotenv
from datasets import load_dataset
from math_verify import LatexExtractionConfig, parse, verify
from trl import GRPOConfig, GRPOTrainer


load_dotenv()

# --- Configuration ---
# Using a 7B model for faster experimentation, but you can change back to 235B if your hardware supports it.
MODEL_ID = "unsloth/DeepSeek-R1-Distill-Qwen-32B"
DATASET_ID = "AI-MO/NuminaMath-TIR"
CACHE_DIR_MODELS = "/data/aiseed/hf-models"
CACHE_DIR_DATASETS = "/data/aiseed/hf-datasets"
OUTPUT_DIR = "unsloth-deepseek-r1-grpo-32b"

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

# --- Reward Functions ---


def format_reward(completions, **_kwargs):
    """Reward function that checks if the completion has <think> and <answer> tags."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def accuracy_reward(completions, **kwargs):
    """Reward function that checks correctness against ground truth."""
    solutions = kwargs["solution"]
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, solution in zip(completion_contents, solutions):
        gold_parsed = parse(
            solution,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        answer_parsed = parse(
            content,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )

        if len(gold_parsed) != 0:
            try:
                rewards.append(float(verify(answer_parsed, gold_parsed)))
            except Exception:
                rewards.append(0.0)
        else:
            rewards.append(1.0)  # Fallback if gold is missing
    return rewards


def reasoning_length_reward(completions, **_kwargs):
    """Encourages longer reasoning traces (the 'Aha' moment)."""
    completion_contents = [completion[0]["content"] for completion in completions]
    lengths = []
    for content in completion_contents:
        # Measure content inside <think> tags
        match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        content_len = len(match.group(1)) if match else 0
        lengths.append(min(1.0, content_len / 1000.0))  # Scaled reward up to 1000 chars
    return lengths


# --- Preprocessing ---


def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }


def main():
    # 1. Load Model & Tokenizer
    print(f"Loading {MODEL_ID} with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=2048,  # Increased for longer reasoning
        load_in_4bit=True,
        cache_dir=CACHE_DIR_MODELS,
        # fast_inference=True,
        device_map="auto",
    )

    # 2. Add LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,  # Slightly higher rank for better reasoning
        target_modules=[
            "q_proj",
            "k_proj",
            # "v_proj",
            # "o_proj",
            # "gate_proj",
            # "up_proj",
            # "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # 3. Load & Preprocess Dataset
    print(f"Loading dataset: {DATASET_ID}")
    dataset = load_dataset(DATASET_ID, split="train", cache_dir=CACHE_DIR_DATASETS)
    # Take a small subset for testing
    dataset = dataset.select(range(1000)).map(make_conversation)
    dataset = dataset.remove_columns(["messages", "problem"])

    # 4. Training Arguments
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        run_name=OUTPUT_DIR,
        learning_rate=5e-6,  # Lower LR often helps RL stability
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        bf16=True,
        gradient_accumulation_steps=8,  # Match TRL setting
        save_strategy="steps",
        save_steps=50,
        max_steps=500,  # Set a cap for testing
        num_generations=4,
        max_completion_length=1024,  # Room for "thinking"
        max_prompt_length=None,  # error 'UnslothGRPOTrainer' object has no attribute 'image_token_id'
        remove_unused_columns=False,
        report_to=["mlflow"],
    )

    # 5. Initialize Trainer
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=[format_reward, accuracy_reward, reasoning_length_reward],
        args=training_args,
        train_dataset=dataset,
    )

    # 6. Train
    print("Starting Optimized GRPO Training...")
    trainer.train()

    # 7. Save
    model.save_pretrained_merged(OUTPUT_DIR, tokenizer, save_method="lora")
    print(f"Completed! Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
