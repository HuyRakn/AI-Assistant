import mlx.core as mx
import numpy as np
from typing import List, Set, Tuple, Optional
import zlib
import redis

class MinHashLSH:
    """
    MinHash Locality Sensitive Hashing (LSH) for fuzzy deduplication.
    Accelerated by MLX for parallel signature computation.
    
    PRODUCTION UPGRADE (Redis Backend):
    Replaced in-memory dictionaries with Redis Sets for infinite horizontal scaling.
    """
    def __init__(self, num_perm: int = 128, threshold: float = 0.8, bands: int = 20, 
                 redis_host: str = 'localhost', redis_port: int = 6379, redis_db: int = 0):
        """
        Args:
            num_perm: Number of permutations (hash functions).
            threshold: Jaccard similarity threshold.
            bands: Number of bands for LSH.
            redis_host: Redis Host (default localhost).
            redis_port: Redis Port (default 6379).
            redis_db: Redis DB Index.
        """
        self.num_perm = num_perm
        self.threshold = threshold
        self.bands = bands
        self.rows_per_band = num_perm // bands
        
        # Generate random hashing parameters (a*x + b) % c
        self.prime = (1 << 61) - 1
        
        rng = np.random.RandomState(42)
        self.perms_a = rng.randint(1, self.prime, size=num_perm, dtype=np.uint64)
        self.perms_b = rng.randint(0, self.prime, size=num_perm, dtype=np.uint64)
        
        # Connect to Sovereign Storage (Redis)
        try:
            self.redis = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)
            self.redis.ping() # Check connection
            print(f"✅ Connected to Redis at {redis_host}:{redis_port} (DB={redis_db})")
        except redis.ConnectionError:
            print("❌ FATAL: Could not connect to Redis. Ensure Redis is running (brew services start redis).")
            # Fallback to crashing/raising error because this is Production mode
            raise

    def clear(self):
        """
        Reset the global index.
        WARNING: This flushes the dedicated Redis DB if we treat it as isolated, 
        or scans for keys. For safety, we use FlushDB on the current DB index.
        """
        self.redis.flushdb()
        print("⚠️ Redis DB Flushed. Index cleared.")

    def insert(self, signature: np.ndarray, doc_id: str):
        """
        Add a signature to the global LSH index (Redis).
        """
        # Pipeline for atomic/fast batch insertion
        pipe = self.redis.pipeline()
        
        for band_idx in range(self.bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            
            segment = tuple(signature[start:end])
            segment_hash = hash(segment)
            
            # Key: aether:lsh:{band_idx}:{segment_hash}
            # Efficient Key Design
            key = f"aether:lsh:{band_idx}:{segment_hash}"
            
            # Add doc_id to the set (0 for exists, or actual ID)
            # We just need to know IF it exists for Deduplication boolean check?
            # Or do we want to retrieve candidates?
            # For "Is Duplicate?", we just need existence.
            # But LSH usually returns Candidates. Even if collision, might not be duplicate.
            # However, for high-speed massive crawl, we often trust LSH collision as Dupe 
            # if bands/rows are tuned (False Positive vs False Negative).
            
            # Storing doc_id allows post-verification.
            pipe.sadd(key, doc_id)
            # Set TTL to auto-expire old keys? Optional.
            pipe.expire(key, 60*60*24*30) # 30 Days retention
            
        pipe.execute()

    def query(self, signature: np.ndarray) -> bool:
        """
        Check if signature matches any existing document in the index.
        Returns: True if candidate found (collision in at least one band).
        """
        # We need to check if ANY band bucket is non-empty.
        pipe = self.redis.pipeline()
        
        for band_idx in range(self.bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            
            segment = tuple(signature[start:end])
            segment_hash = hash(segment)
            
            key = f"aether:lsh:{band_idx}:{segment_hash}"
            pipe.exists(key)
            
        results = pipe.execute()
        # if any result is true (1)
        return any(results)

    def compute_signature(self, text: str, n_gram: int = 5) -> np.ndarray:
        """
        Compute MinHash signature for a text using MLX.
        """
        words = text.split()
        if len(words) < n_gram:
            shingles = {text[i:i+n_gram] for i in range(len(text) - n_gram + 1)}
        else:
            shingles = {" ".join(words[i:i+n_gram]) for i in range(len(words) - n_gram + 1)}
            
        if not shingles:
            return np.full(self.num_perm, self.prime, dtype=np.uint64)

        shingle_hashes = np.array([zlib.crc32(s.encode('utf-8')) & 0xffffffff for s in shingles], dtype=np.uint64)
        
        if len(shingle_hashes) == 0:
             return np.full(self.num_perm, self.prime, dtype=np.uint64)

        mx_shingles = mx.array(shingle_hashes.astype(np.uint64))
        mx_a = mx.array(self.perms_a.astype(np.uint64)).reshape(-1, 1)
        mx_b = mx.array(self.perms_b.astype(np.uint64)).reshape(-1, 1)
        
        mx_shingles = mx_shingles.reshape(1, -1)
        hashed_matrix = (mx_a * mx_shingles + mx_b) % self.prime
        signature = mx.min(hashed_matrix, axis=1)
        
        return np.array(signature)

    def compute_jaccard(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        return np.mean(sig1 == sig2)

