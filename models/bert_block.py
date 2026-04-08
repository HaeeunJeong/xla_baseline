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
    # 1. Create BERT-base uncased: 12 layers, 768 hidden size, 12 attention heads, 110M parameters
    # 2. Download the model weights from Hugging Face and load them into the model
    # 3. Generate model graph consisted with PyTorch module
    model = transformers.BertModel(config)
    return HFWrapper(model.eval())

def get_dummy_input():
    tok = transformers.AutoTokenizer.from_pretrained(_MODEL_ID)
    enc = tok(
        "Hello BERT, I want to optimize you.",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=_SEQ,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    
    token_type_ids = enc.get("token_type_ids")
    if token_type_ids is None:
        token_type_ids = torch.zeros_like(input_ids)

    return input_ids, attention_mask, token_type_ids