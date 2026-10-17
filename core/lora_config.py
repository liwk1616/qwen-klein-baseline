# Extracted from scripts/flintstones_klein/train.py; historical orchestration removed.

LORA_MODULES = r"^(?:transformer_blocks\.\d+\.attn\.(?:to_q|to_k|to_v|add_q_proj|add_k_proj|add_v_proj|to_add_out|to_out\.0)|single_transformer_blocks\.\d+\.attn\.(?:to_qkv_mlp_proj|to_out))$"
