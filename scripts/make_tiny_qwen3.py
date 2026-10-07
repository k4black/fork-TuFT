"""Build a tiny random Qwen3 with the real Qwen3 tokenizer for CPU integration tests.

Usage: python scripts/make_tiny_qwen3.py OUT_DIR
"""

import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


BASE = "Qwen/Qwen3-0.6B"

out = sys.argv[1]
# head_dim 32 is the smallest head size the vLLM CPU attention backend supports.
cfg = AutoConfig.from_pretrained(
    BASE,
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    max_position_embeddings=4096,
)
cfg.layer_types = cfg.layer_types[:2]  # the base config lists all 28 layers
torch.manual_seed(0)
# float32: CPUs without bf16 units run the HF training backend far slower in bfloat16.
AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).save_pretrained(out)
AutoTokenizer.from_pretrained(BASE).save_pretrained(out)
