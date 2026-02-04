import re
import torch
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from math_verify import LatexExtractionConfig, parse, verify
from trl import GRPOConfig, GRPOTrainer

# Load environment variables from .env
load_dotenv()

# --- Configuration ---
MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DATASET_ID = "AI-MO/NuminaMath-TIR"
CACHE_DIR_MODELS = "/data/aiseed/hf-models"
CACHE_DIR_DATASETS = "/data/aiseed/hf-datasets"
OUTPUT_DIR = "Qwen3-30B-A3B-GRPO-test"

# MLflow settings (loaded from .env via load_dotenv)

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

# --- Reward Functions ---


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    return rewards_list


def accuracy_reward(completions, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
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
            rewards.append(1.0)
    return rewards


# --- Preprocessing ---


def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }


def main():
    # 1. Load Dataset
    print(f"Loading dataset: {DATASET_ID}")
    train_dataset, test_dataset = load_dataset(
        DATASET_ID, split=["train[:5%]", "test[:5%]"], cache_dir=CACHE_DIR_DATASETS
    )

    # 2. Preprocess Dataset
    train_dataset = train_dataset.map(make_conversation)
    test_dataset = test_dataset.map(make_conversation)

    # Remove unnecessary columns
    train_dataset = train_dataset.remove_columns(["messages", "problem"])

    # 3. Load Model
    print(f"Loading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype="auto", device_map="auto", cache_dir=CACHE_DIR_MODELS
    )

    # 4. LoRA Configuration
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Training Configuration
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        run_name=OUTPUT_DIR,
        learning_rate=1e-5,
        remove_unused_columns=False,  # to access the solution column in accuracy_reward
        gradient_accumulation_steps=16,
        num_train_epochs=5,
        bf16=True,
        # Parameters that control the data preprocessing
        max_completion_length=64,
        num_generations=4,
        # Parameters related to reporting and saving
        report_to=["mlflow"],
        logging_steps=100,
        push_to_hub=False,
        save_strategy="steps",
        save_steps=100,
    )

    # 6. Initialize Trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward, accuracy_reward],
        args=training_args,
        train_dataset=train_dataset,
    )

    # 7. Start Training
    print("Starting training...")
    trainer.train()
    print("Training completed.")


if __name__ == "__main__":
    main()
