'''
HuggingFace Wrapper

HF models basically returns dictionary-like objects:
BaseModelOutput(
    last_hidden_state,
    pooler_output,
    hidden_states,
    attentions,
    ...
)

This wrapper simplifies this object: eliminate unused outputs and normalize
(input_ids, attention_mask)
    →
last_hidden_state [B,S,H]
'''


from __future__ import annotations
import torch

class HFWrapper(torch.nn.Module):
    """kwargs(Llama) → forward(ids, mask)"""
    def __init__(self, backbone: torch.nn.Module):
        super().__init__(); self.m = backbone
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self.m(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
