from dataclasses import dataclass

@dataclass
class AetherConfig:
    """
    Titan Configuration (The Genome).
    Defines the shape and capacity of the Aether Model.
    """
    vocab_size: int = 50000         # From Phase 1 Titan Tokenizer
    dim: int = 4096                 # Embedding dimension
    n_layers: int = 32              # Number of Transformer blocks
    n_heads: int = 32               # Number of Attention heads
    n_kv_heads: int = None          # Number of KV heads (for GQA, None = same as n_heads)
    head_dim: int = None            # Dimension of each head (default = dim // n_heads)
    hidden_dim: int = None          # FeedForward hidden dim (default = 4 * dim for standard, but SwiGLU uses custom logic)
    
    norm_eps: float = 1e-5          # Epsilon for RMSNorm
    rope_theta: float = 10000.0     # Base period for RoPE
    rope_traditional: bool = False  # False = Standard Llama RoPE
    dropout: float = 0.1            # Dropout probability (White-Box Requirement)
    
    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        
        if self.head_dim is None:
            self.head_dim = self.dim // self.n_heads
            
        if self.hidden_dim is None:
            # Llama-style SwiGLU hidden dim is usually: 2/3 * 4 * dim (approx)
            # But let's stick to standard configurable or safe default
            self.hidden_dim = 4 * self.dim
            self.hidden_dim = int(2 * self.hidden_dim / 3)
            # Ensure it's a multiple of 256 for efficiency
            self.hidden_dim = 256 * ((self.hidden_dim + 255) // 256)
