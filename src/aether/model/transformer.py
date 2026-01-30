import mlx.core as mx
import mlx.nn as nn
import math
from typing import Optional, Tuple
from .configuration import AetherConfig
from .layers import RMSNorm, AetherAttention, SwiGLU

class AetherDecoderLayer(nn.Module):
    """
    A single Transformer Block.
    Flow:
    Input -> RMSNorm -> Attention -> + Residual
          -> RMSNorm -> SwiGLU    -> + Residual
    """
    def __init__(self, config: AetherConfig):
        super().__init__()
        self.self_attn = AetherAttention(config)
        self.mlp = SwiGLU(config)
        self.input_layernorm = RMSNorm(config.dim, eps=config.norm_eps)
        self.post_attention_layernorm = RMSNorm(config.dim, eps=config.norm_eps)

    def __call__(self, x, mask=None, cache=None):
        # Attention Block (Pre-Norm)
        r = self.input_layernorm(x)
        h, cache = self.self_attn(r, mask=mask, cache=cache)
        x = x + h # Residual Connection
        
        # FFN Block (Pre-Norm)
        r = self.post_attention_layernorm(x)
        x = x + self.mlp(r) # Residual Connection
        
        return x, cache

class AetherModel(nn.Module):
    """
    The Aether Titan Backbone.
    """
    def __init__(self, config: AetherConfig):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)
        self.layers = [
            AetherDecoderLayer(config) for _ in range(config.n_layers)
        ]
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        
    def __call__(self, inputs: mx.array, mask: Optional[mx.array] = None, cache=None):
        h = self.embed_tokens(inputs)
        
        # Iterate layers
        new_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            h, c = layer(h, mask=mask, cache=layer_cache)
            if cache is not None:
                new_cache.append(c)
                
        h = self.norm(h)
        
        return h, new_cache

    def apply_initialization(self):
        # Deprecated: Logic moved to src/aether/model/utils.py (SovereignInit)
        pass

class AetherForCausalLM(nn.Module):
    """
    End-to-End Language Model Head.
    Adds a linear projection to vocab size at the end.
    """
    def __init__(self, config: AetherConfig):
        super().__init__()
        self.model = AetherModel(config)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Tie weights if usually done (Llama doesn't mandate tied weights but it matches embedding often)
        # For simplicity/explicit structure, keeping separate or tied?
        # Typically SOTA models have separate or tied. Let's keep separate for flexibility unless specified.
        # But wait, weights sharing is good for size. Let's verify Titan specs? 
        # User didn't specify. Standard Llama 2 uses non-tied.
        
        # --- SURVIVAL TECH: Manual Weight Initialization (Phase 3.2) ---
        from .utils import SovereignInit
        SovereignInit.init_model(self, config.n_layers)
        # ---------------------------------------------------
        
    def __call__(self, inputs, mask=None, cache=None):
        out, new_cache = self.model(inputs, mask, cache)
        logits = self.lm_head(out)
        return logits, new_cache
