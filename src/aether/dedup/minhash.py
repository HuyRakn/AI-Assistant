import mlx.core as mx
import numpy as np
from typing import List, Set, Tuple
import zlib

class MinHashLSH:
    """
    MinHash Locality Sensitive Hashing (LSH) for fuzzy deduplication.
    Accelerated by MLX for parallel signature computation.
    """
    def __init__(self, num_perm: int = 128, threshold: float = 0.8):
        """
        Args:
            num_perm: Number of permutations (hash functions).
            threshold: Jaccard similarity threshold.
        """
        self.num_perm = num_perm
        self.threshold = threshold
        # Generate random hashing parameters (a*x + b) % c
        # We use a large prime for c (Mersenne prime M_61)
        self.prime = (1 << 61) - 1
        
        # Initialize permutation parameters on GPU if available
        # But for simple hashing, standard numpy/python might be sufficient 
        # unless batch processing. Let's start with optimized Python for simplicity
        # and MLX implementation for the heavy lifting if we move to dense representations.
        
        # Consistent random seed for reproducibility
        rng = np.random.RandomState(42)
        self.perms_a = rng.randint(1, self.prime, size=num_perm, dtype=np.uint64)
        self.perms_b = rng.randint(0, self.prime, size=num_perm, dtype=np.uint64)

    def compute_signature(self, text: str, n_gram: int = 5) -> np.ndarray:
        """
        Compute MinHash signature for a text.
        """
        # 1. Shingling (n-grams)
        words = text.split()
        if len(words) < n_gram:
            # Fallback for short texts: character n-grams
            shingles = {text[i:i+n_gram] for i in range(len(text) - n_gram + 1)}
        else:
            shingles = {" ".join(words[i:i+n_gram]) for i in range(len(words) - n_gram + 1)}
            
        if not shingles:
            return np.full(self.num_perm, self.prime, dtype=np.uint64)

        # 2. Hash shingles to integers (32-bit CRC32 is fast and sufficient for shingle ID)
        shingle_hashes = np.array([zlib.crc32(s.encode('utf-8')) & 0xffffffff for s in shingles], dtype=np.uint64)
        
        # 3. Compute MinHash Signature (Titan Edition: Vectorized MLX)
        if len(shingle_hashes) == 0:
             return np.full(self.num_perm, self.prime, dtype=np.uint64)

        # Convert to MLX arrays (Move to GPU)
        # Shape: (num_shingles, )
        mx_shingles = mx.array(shingle_hashes.astype(np.uint64))
        
        # Permutation Params: (num_perm, 1)
        mx_a = mx.array(self.perms_a.astype(np.uint64)).reshape(-1, 1)
        mx_b = mx.array(self.perms_b.astype(np.uint64)).reshape(-1, 1)
        
        # Broadcasting:
        # (num_perm, 1) * (1, num_shingles) -> (num_perm, num_shingles)
        # This computes all hashes for all shingles in parallel
        
        # Note: MLX broadcasting works automatically
        # a * shingle + b
        # We need to reshape shingles to (1, num_shingles) first for correct broadcast
        mx_shingles = mx_shingles.reshape(1, -1)
        
        # Hash Matrix: [Num_Perms, Num_Shingles]
        # Calculation: ((a * x) + b) % prime
        # Since numbers are large (uint64), proceed with caution on overflow if not handled automatically.
        # MLX supports uint64.
        
        hashed_matrix = (mx_a * mx_shingles + mx_b) % self.prime
        
        # 4. Global Min Pooling
        # Find min hash value for each permutation across all shingles
        # Reduce along axis 1 (shingles) -> Result shape: (num_perm,)
        signature = mx.min(hashed_matrix, axis=1)
        
        # Convert back to numpy (CPU)
        return np.array(signature)

    def lsh_banding(self, signatures: List[np.ndarray], bands: int = 20) -> Set[Tuple[int, int]]:
        """
        Identify candidate pairs using LSH Banding.
        Divide signature into 'bands' of 'rows'.
        """
        rows_per_band = self.num_perm // bands
        candidates = set()
        
        # Dictionary mapping: Band_Index -> { Hash_Value -> [Doc_IDs] }
        buckets = [{} for _ in range(bands)]
        
        for doc_id, sig in enumerate(signatures):
            for band_idx in range(bands):
                start = band_idx * rows_per_band
                end = start + rows_per_band
                
                # Create a hashable segment for this band
                segment = tuple(sig[start:end])
                segment_hash = hash(segment)
                
                bucket = buckets[band_idx]
                if segment_hash in bucket:
                    # Found collision -> Candidates
                    for existing_doc_id in bucket[segment_hash]:
                        # Store pair (smaller, larger) to avoid duplicates
                        if existing_doc_id < doc_id:
                            candidates.add((existing_doc_id, doc_id))
                        else:
                            candidates.add((doc_id, existing_doc_id))
                    bucket[segment_hash].append(doc_id)
                else:
                    bucket[segment_hash] = [doc_id]
                    
        return candidates

    def compute_jaccard(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """
        Estimate Jaccard similarity from signatures.
        """
        return np.mean(sig1 == sig2)

