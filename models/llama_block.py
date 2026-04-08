from __future__ import annotations
import torch, transformers
from ._hf_wrapper import HFWrapper
ID, _SEQ = "meta-llama/Llama-3.2-1B", 512

def get_model():
    config = transformers.LlamaConfig.from_pretrained(ID)
    config.return_dict = False
    model = transformers.LlamaModel.from_pretrained(ID, dtype=torch.float32, config=config)
    return HFWrapper(model.eval())
def get_dummy_input():
    tok = transformers.AutoTokenizer.from_pretrained(ID)
    tok.pad_token = tok.eos_token  # Llama has no pad token by default
    enc = tok(
        "Hello Llama, I want to optimize you.",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=_SEQ,
    )
    return enc["input_ids"], enc["attention_mask"]

