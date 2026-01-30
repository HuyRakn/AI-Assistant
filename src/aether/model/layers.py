import mlx.core as mx
import mlx.nn as nn
import math
from typing import Optional, Tuple
from .configuration import AetherConfig

class RMSNorm(nn.Module):
    """
    Production Solution 2: RMSNorm (Root Mean Square Normalization).
    More stable than LayerNorm for deep networks.
    x = x * w / sqrt(mean(x^2) + eps)
    """
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def _norm(self, x):
        return x * mx.rsqrt(x.square().mean(-1, keepdims=True) + self.eps)

    def __call__(self, x):
        output = self._norm(x.astype(mx.float32)).astype(x.dtype)
        return self.weight * output

class RoPE(nn.Module):
    """
    Production Solution 1: Rotary Positional Embeddings.
    Encodes relative position by rotating vectors in complex space.
    x' = x * e^(i * theta)
    """
    def __init__(self, dims: int, base: float = 10000.0):
        super().__init__()
        self.dims = dims
        self.base = base
        # Precompute frequency cis (cos + i*sin) is done dynamically usually or cached.
        
    def __call__(self, x, offset: int = 0):
        # Manual White-Box RoPE (Sovereign Math)
        # 1. Generate frequencies
        B, L, H, D = x.shape
        cos, sin = precompute_freqs_cis(D, L + offset, self.base)
        
        # Select the slice corresponding to current window
        cos = cos[offset : offset + L]
        sin = sin[offset : offset + L]
        
        # Reshape for broadcast: [L, D/2] -> [1, L, 1, D/2]
        cos = cos.reshape(1, L, 1, -1)
        sin = sin.reshape(1, L, 1, -1)
        
        return apply_rope(x, cos, sin)
        
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2)[: (dim // 2)].astype(mx.float32) / dim))
    t = mx.arange(end, dtype=mx.float32)
    freqs = mx.outer(t, freqs) # (SeqLen, Dim/2)
    # Return (cos, sin)
    return mx.cos(freqs), mx.sin(freqs)

def apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    # x: [B, H, L, D]
    # Split into real and imaginary parts (or pairs)
    # x = [x0, x1, x2, x3, ...]
    
    # Reshape to separate pairs: [B, H, L, D/2, 2]
    # Note: Optimization for [B, H, L, D] layout
    B, H, L, D = x.shape
    x_reshaped = x.reshape(B, H, L, -1, 2)
    
    x1 = x_reshaped[..., 0] # Real part
    x2 = x_reshaped[..., 1] # Imaginary part
    
    # Rotate:
    # x1' = x1 * cos - x2 * sin
    # x2' = x1 * sin + x2 * cos
    
    out_x1 = x1 * cos - x2 * sin
    out_x2 = x1 * sin + x2 * cos
    
    # Stack back and flatten
    out = mx.stack([out_x1, out_x2], axis=-1)
    return out.reshape(B, H, L, D)

class SwiGLU(nn.Module):
    """
    Production Solution 3: SwiGLU (Swish Gated Linear Unit).
    FFN = (Swish(xW_gate) * xW_val) W_out
    """
    def __init__(self, config: AetherConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.down_proj = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x):
        # Gate path (Cognition Gate)
        gate = nn.silu(self.gate_proj(x))
        # Value path (Information)
        value = self.up_proj(x)
        # Element-wise mult
        h = gate * value
        # Down project with Dropout
        output = self.down_proj(h)
        return self.dropout(output)

class AetherAttention(nn.Module):
    """
    MHA with RoPE and SDPA.
    """
    def __init__(self, config: AetherConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(config.dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.dim, bias=False)
        
        # USE MANUAL WHITE-BOX ROPE
        self.rope = RoPE(self.head_dim, base=config.rope_theta)
        
        # Manual Dropout Components
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache=None):
        B, L, D = x.shape
        
        # Projections
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        
        # Reshape for heads: [B, L, H, D_h] -> Transpose to [B, H, L, D_h]
        # This layout is optimal for Attention (H is batch-like)
        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Apply RoPE (Titan Positioning)
        # Optimization: Apply directly on [B, H, L, D] without Transpose
        # Now input is [B, H, L, D], RoPE will treat axis 2 (L) as sequence correctly
        
        if cache is not None:
             # If using KV cache (during generation), apply RoPE with offset
             # cache shape is [B, H, L, D] usually.
             offset = cache.offset if hasattr(cache, 'offset') else 0
             # Note: simple RoPE call handles offset logic
             queries = self.rope(queries, offset=offset)
             keys = self.rope(keys, offset=offset)
        else:
             queries = self.rope(queries)
             keys = self.rope(keys)
             
        # KV Cache management (Static Cache Optimization)
        if cache is not None:
            # cache is passed as the state object for this layer
            
            # Standardized StaticKVCache usage
            if hasattr(cache, 'update_and_fetch'): 
                # Duct-typing check is safer if import is tricky, but we should import.
                # Assuming cache object has the interface.
                keys, values = cache.update_and_fetch(keys, values)
            else:
                 # Check if it is a tuple (Legacy/Fallback)
                 if isinstance(cache, tuple) and len(cache) == 2:
                     key_cache, value_cache = cache
                     keys = mx.concatenate([key_cache, keys], axis=2)
                     values = mx.concatenate([value_cache, values], axis=2)
                 else:
                     # Unexpected cache type
                     pass
            
        # --- WHITE-BOX ATTENTION IMPLEMENTATION (NO FUSED KERNELS) ---
        # 1. Compute Scores: Q * K^T / sqrt(d)
        # queries: [B, H, L, D]
        # keys:    [B, H, L_kv, D] -> Transpose to [B, H, D, L_kv]
        # Result:  [B, H, L, L_kv]
        
        # Explicit Matrix Multiplication
        scores = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale
        
        # 2. Apply Causal Mask (Manual)
        if mask is not None:
            # mask should be additive (0 for keep, -inf for discard)
            scores = scores + mask
            
        # 3. Softmax (Probabilities) - The heart of attention
        probs = mx.softmax(scores, axis=-1)
        
        # 4. Dropout (Anti-Overfitting/Regularization)
        # Randomly zero out attention weights
        probs = self.attn_dropout(probs)
        
        # 5. Weighted Sum: Probs * Values
        output = probs @ values
        # output: [B, H, L, D]
        # -------------------------------------------------------------
        
        # Transpose back to [B, L, H, D] then reshape to [B, L, H*D]
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        
        # Final Output Projection with Dropout
        return self.resid_dropout(self.o_proj(output)), (keys, values)
