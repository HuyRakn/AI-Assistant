import mlx.core as mx
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import json
import os

class VectorDB:
    """
    The Knowledge Vault (Phase 3.5).
    A White-Box Vector Database implementation.
    
    Features:
    - Pure MLX/Numpy backend (No dependency on Faiss/Chroma).
    - Cosine Similarity Search.
    - Persistent Storage (JSON + Numpy specific binary).
    - Semantic Routing capabilities.
    """
    def __init__(self, dim: int, collection_name: str = "default_vault"):
        self.dim = dim
        self.collection_path = f"knowledge_vault/{collection_name}"
        
        # In-Memory Storage
        self.embeddings: Optional[mx.array] = None
        self.metadata: List[Dict] = []
        
        # Ensure storage directory
        if not os.path.exists(self.collection_path):
            os.makedirs(self.collection_path)
            
    def add(self, embeddings: Union[mx.array, np.ndarray, List[float]], meta: List[Dict]):
        """
        Add vectors to the vault.
        """
        if isinstance(embeddings, list):
             embeddings = mx.array(embeddings)
        elif isinstance(embeddings, np.ndarray):
             embeddings = mx.array(embeddings)
             
        # Normalize for Cosine Similarity
        # L2 Norm: x / ||x||
        norm = mx.linalg.norm(embeddings, axis=-1, keepdims=True)
        embeddings = embeddings / (norm + 1e-9)
        
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = mx.concatenate([self.embeddings, embeddings], axis=0)
            
        self.metadata.extend(meta)
        
    def search(self, query: Union[mx.array, List[float]], top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Semantic Search via Dot Product (Cosine Similarity since normalized).
        """
        if self.embeddings is None:
            return []
            
        if isinstance(query, list):
            query = mx.array(query)
            
        # Normalize Query
        q_norm = mx.linalg.norm(query, axis=-1, keepdims=True)
        query = query / (q_norm + 1e-9)
        
        # Dot Product: [1, D] @ [N, D].T -> [1, N]
        # or just query @ embeddings.T
        scores = query @ self.embeddings.T
        
        # Top K
        # MLX argpartition is efficient for TopK?
        # mx.topk returns values and indices
        k = min(top_k, self.embeddings.shape[0])
        
        # We want indices of top scores (largest first)
        # Note: mx.argpartition gets indices but unordered. mx.sort/argsort is safer for exact order.
        # For small K, just sort?
        # Default implementation:
        indices = mx.argsort(scores, axis=-1)[::-1][:k] # Sort descending
        
        results = []
        for idx in indices.tolist():
            results.append((self.metadata[idx], scores[idx].item()))
            
        return results

    def save(self):
        """Persist vault to disk."""
        if self.embeddings is not None:
             np.save(f"{self.collection_path}/embeddings.npy", np.array(self.embeddings))
        
        with open(f"{self.collection_path}/metadata.json", "w") as f:
            json.dump(self.metadata, f)
            
        print(f"✅ Knowledge Vault '{self.collection_path}' persisted.")

    def load(self):
        """Load vault from disk."""
        emb_path = f"{self.collection_path}/embeddings.npy"
        meta_path = f"{self.collection_path}/metadata.json"
        
        if os.path.exists(emb_path):
             self.embeddings = mx.array(np.load(emb_path))
             
        if os.path.exists(meta_path):
             with open(meta_path, "r") as f:
                 self.metadata = json.load(f)
                 
        print(f"✅ Knowledge Vault loaded (Size: {len(self.metadata)}).")
