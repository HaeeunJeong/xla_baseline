from __future__ import annotations
import torch, transformers
from ._hf_wrapper import HFWrapper
ID, _SEQ = "meta-llama/Llama-3.2-1B", 512

def get_model():
    config = transformers.LlamaConfig.from_pretrained(ID)
    config.num_hidden_layers = 1
    config.return_dict = False
    model = transformers.LlamaModel(config).eval()
    return HFWrapper(model)

def get_dummy_input():
    # Use deterministic sequential token IDs without padding to produce a pure causal mask
    input_ids = (torch.arange(_SEQ, dtype=torch.long) % 1000).unsqueeze(0)
    attention_mask = torch.ones((1, _SEQ), dtype=torch.long)
    return input_ids, attention_mask