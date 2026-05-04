from __future__ import annotations
import torch, transformers
from ._hf_wrapper import HFWrapper
from transformers import BertModel

_MODEL_ID = "bert-base-uncased"
_SEQ = 512

def get_model():
    config = transformers.BertConfig.from_pretrained(_MODEL_ID)
    config.return_dict = False

    # Model Loading
    model = transformers.BertModel.from_pretrained(_MODEL_ID)
    return HFWrapper(model.eval())

def get_dummy_input():
    input_ids = (torch.arange(_SEQ, dtype=torch.long) % 1000).unsqueeze(0)
    attention_mask = torch.ones((1, _SEQ), dtype=torch.long)
    token_type_ids = torch.zeros_like(input_ids)

    return input_ids, attention_mask, token_type_ids