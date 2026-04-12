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
    input_ids = (torch.arange(_SEQ, dtype=torch.long) % 1000).unsqueeze(0)
    attention_mask = torch.ones((1, _SEQ), dtype=torch.long)
    return input_ids, attention_mask

