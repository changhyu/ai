# AI

This repository contains a simple example of how to load and run a LLaMA model using Hugging Face's `transformers` library.

## Requirements

- Python 3.10+
- PyTorch
- transformers >= 4.31

## Example

The `llama_sample.py` script demonstrates how to load the base model and run inference with a prompt. You will need access to the LLaMA model weights from Meta. Update the `model_name` in the script to the path where you've stored the model.

```bash
pip install torch transformers
python llama_sample.py
```
