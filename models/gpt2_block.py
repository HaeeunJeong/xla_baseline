from __future__ import annotations
import torch, transformers
from ._hf_wrapper import HFWrapper
ID, _SEQ = "gpt2", 512

def get_model():
    config = transformers.GPT2Config.from_pretrained(ID)
    config.return_dict = False
    model = transformers.GPT2Model.from_pretrained(ID, config=config)
    return HFWrapper(model.eval())
def get_dummy_input():
    tok = transformers.AutoTokenizer.from_pretrained(ID)
    tok.pad_token = tok.eos_token  # GPT2 has no pad token by default
    enc = tok(
        "Hello GPT2, I want to optimize you.",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=_SEQ,
    )
    return enc["input_ids"], enc["attention_mask"]

