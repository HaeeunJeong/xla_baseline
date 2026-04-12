from __future__ import annotations
import torch, transformers
from ._hf_wrapper import HFWrapper
ID, _SEQ = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 512

def get_model():
    config = transformers.Qwen2Config.from_pretrained(ID)
    model = transformers.Qwen2ForCausalLM.from_pretrained(ID, dtype=torch.float32, config=config)
    return HFWrapper(model.eval())
def get_dummy_input():
    input_ids = (torch.arange(_SEQ, dtype=torch.long) % 1000).unsqueeze(0)
    attention_mask = torch.ones((1, _SEQ), dtype=torch.long)
    return input_ids, attention_mask

