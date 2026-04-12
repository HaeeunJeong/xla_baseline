from __future__ import annotations
import torch, transformers
from ._hf_wrapper import HFWrapper
ID, _SEQ = "meta-llama/Llama-3.2-1B", 512

def get_model(): return HFWrapper(transformers.AutoModel.from_pretrained(ID, dtype=torch.float32).eval())
def get_dummy_input():
    input_ids = (torch.arange(_SEQ, dtype=torch.long) % 1000).unsqueeze(0)
    attention_mask = torch.ones((1, _SEQ), dtype=torch.long)
    return input_ids, attention_mask
