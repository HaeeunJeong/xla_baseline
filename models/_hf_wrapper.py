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
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor | None = None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        
        out = self.m(**kwargs)
        # If return_dict=False
        if isinstance(out, tuple):
            return out[0]
        # If return_dict=True
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        elif hasattr(out, "logits"):
            return out.logits
        return out
