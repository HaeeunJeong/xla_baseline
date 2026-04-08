from __future__ import annotations
import torch, transformers
from ._hf_wrapper import HFWrapper
ID, _SEQ = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 512

def get_model():
    config = transformers.Qwen2Config.from_pretrained(ID)
    model = transformers.Qwen2ForCausalLM.from_pretrained(ID, dtype=torch.float32, config=config)
    return HFWrapper(model.eval())
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

