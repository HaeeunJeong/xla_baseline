from __future__ import annotations
import torch, transformers
from ._hf_wrapper import HFWrapper

_MODEL_ID = "bert-base-uncased"
_SEQ = 32

def get_model():
    # Model Loading
    # 1. Create BERT-base uncased: 12 layers, 768 hidden size, 12 attention heads, 110M parameters
    # 2. Download the model weights from Hugging Face and load them into the model
    # 3. Generate model graph consisted with PyTorch module
    return HFWrapper(transformers.AutoModel.from_pretrained(_MODEL_ID).eval())

def get_dummy_input():
    tok = transformers.AutoTokenizer.from_pretrained(_MODEL_ID)
    enc = tok(
        "Hello BERT, I want to optimize you.",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=_SEQ,
    )
    return enc["input_ids"], enc["attention_mask"]

