"""Basic script for fine-tuning a language model with Hugging Face Transformers.

This example assumes you have a dataset in JSON format with fields 'text' or a pair of 'instruction' and 'response'.
"""

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments


def main(dataset_path: str, model_name: str = "gpt2-medium", output_dir: str = "./model"):
    # Load dataset
    data = load_dataset("json", data_files=dataset_path)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

def tokenize_function(example):
-    # original single‐field tokenization
-    return tokenizer(example["text"], truncation=True)
+    # build prompt from instruction/response pair if available
+    if "instruction" in example and "response" in example:
+        prompt = example["instruction"] + tokenizer.eos_token + example["response"]
+    else:
+        # fallback to a single text field
+        prompt = example.get("text", "")
+    return tokenizer(prompt, truncation=True)

    tokenized_datasets = data.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        num_train_epochs=3,
        evaluation_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune a model for car repair knowledge")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSON file")
    parser.add_argument("--model_name", default="gpt2-medium", help="Base model name")
    parser.add_argument("--output_dir", default="./model", help="Directory to save the fine-tuned model")

    args = parser.parse_args()
    main(args.dataset, args.model_name, args.output_dir)
