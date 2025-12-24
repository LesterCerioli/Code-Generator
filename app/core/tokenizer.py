from transformers import AutoTokenizer

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
