from __future__ import annotations
import torch, transformers
from ._hf_wrapper import HFWrapper
ID, _SEQ = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 128

def get_model(): return HFWrapper(transformers.AutoModel.from_pretrained(ID, dtype=torch.float32).eval())
def get_dummy_input():
    tok = transformers.AutoTokenizer.from_pretrained(ID)
    enc = tok(
        "Hello DeepSeek, I want to optimize you.",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=_SEQ,
    )
    return enc["input_ids"], enc["attention_mask"]

