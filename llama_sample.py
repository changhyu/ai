"""Simple example of running a LLaMA model with transformers."""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def main():
    # Replace with your local path or Hugging Face model hub name
    model_name = "path/to/llama"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

    prompt = "Introduce yourself in one sentence."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
        )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
