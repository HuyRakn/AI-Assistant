from typing import List, Iterator, Callable
import mlx.data as dx
from ..data.ingestion import create_pipeline as dx_pipeline
from .vietnamese import ViToneNormalizer
from .english import EnglishNormalizer
from .cleaning import DeepTextCleaner
from .segmentation import VietnameseSegmenter
from ..dedup.minhash import MinHashLSH
import numpy as np

class AetherDataFactory:
    """
    The Orchestrator.
    Constructs the end-to-end data processing assembly line.
    Step 1: Raw Parquet -> Step 2: Clean -> Step 3: Normalize -> Step 4: Tokenize
    Step 0: Deduplication (Optional Batch Process)
    """
    
    
    def __init__(self, tokenizer_model_path: str = None):
        self.cleaner = DeepTextCleaner()
        self.normalizer = ViToneNormalizer()
        self.segmenter = VietnameseSegmenter()
        self.eng_norm = EnglishNormalizer() # Disconnected Treasury Fixed
        
        # Load Tokenizer using Sovereign AetherBPE
        from ..tokenization.bpe import AetherBPE
        self.tokenizer = AetherBPE()
        
        if tokenizer_model_path:
             self.tokenizer.load(tokenizer_model_path)
        else:
             print("Warning: AetherDataFactory initialized without loaded tokenizer.")
             
        # Dedup Engine
        self.dedup_engine = MinHashLSH()

    def stream_deduplication(self, text_iterator: Iterator[str], batch_size: int = 10000) -> Iterator[str]:
        """
        Streaming Deduplication (Generator).
        Prevents RAM explosion by processing in batches while maintaining 
        a global LSH index in memory.
        
        Args:
            text_iterator: Generator or Iterator yielding strings.
            batch_size: Number of docs to process per LSH update.
            
        Yields:
            Unique documents.
        """
        batch = []
        # We need a way to track duplicates globally. 
        # MinHashLSH in this repo currently recalculates bands per call?
        # To support streaming, we need persistent buckets in dedup_engine.
        # Assuming dedup_engine (MinHashLSH) is persistent and we can just check against it?
        # The current MinHashLSH.lsh_banding takes a list of signatures.
        # We should probably adapt it or just keep "seen signatures" here if cheap.
        # But MinHash signatures are arrays.
        
        # Let's use a simplified approach for this Phase:
        # 1. Accumulate batch.
        # 2. Check collisions within batch.
        # 3. Check collisions against history (requires storage).
        
        # PRO IMPLEMENTATION: 
        # We will add signatures to the engine incrementally.
        # But `dedup_engine` might not have `add` method.
        # Let's implement a batch-process that maintains a lightweight history of "Band Hashes".
        
        # For this refactor, we focus on the Memory Safety first:
        # We assume the user accepts that we deduplicate *within* specific windows 
        # OR we maintain a set of seen checksums (Exact Dedup) + MinHash (Fuzzy)?
        
        # Let's Implement: Batch-Processing with Global Signature Tracking.
        # We will store `seen_band_hashes` in the factory or engine.
        
        # Note: Ideally this logic belongs in `MinHashLSH`, but we are patching Pipeline.
        # We will simply process batches and prevent the list explosion.
        
        print(f"Starting Streaming Deduplication (Batch Size: {batch_size})...")
        
        for text in text_iterator:
            batch.append(text)
            if len(batch) >= batch_size:
                # Process Batch
                unique_batch = self._deduplicate_batch(batch)
                for doc in unique_batch:
                    yield doc
                batch = [] # Reset
                
        # Final batch
        if batch:
            unique_batch = self._deduplicate_batch(batch)
            for doc in unique_batch:
                yield doc

    def _deduplicate_batch(self, batch: List[str]) -> List[str]:
        """Internal batch processor (Stateful)."""
        # 1. Compute Signatures
        # We process sequentially or parallel?
        # Parallel compute signatures first (MLX efficiency)
        signatures = [self.dedup_engine.compute_signature(t) for t in batch]
        
        unique_batch_docs = []
        
        # 2. Check against Global Index
        import hashlib
        
        for i, sig in enumerate(signatures):
            # Check if exists in history
            is_dupe = self.dedup_engine.query(sig)
            
            if not is_dupe:
                # Unique: Keep it and Add to Index
                unique_batch_docs.append(batch[i])
                
                # Generate a content-based ID for retrieval/debugging
                # Using MD5 for speed (cryptographic security not needed for ID generation here)
                doc_id = hashlib.md5(batch[i].encode('utf-8')).hexdigest()
                
                self.dedup_engine.insert(sig, doc_id)
            else:
                # Duplicate: Drop it
                # (Optional: We could verify Jaccard here if we stored signatures, 
                # but for memory efficiency we trust LSH bands)
                pass
                
        return unique_batch_docs

    def _process_text(self, text: str) -> str:
        """
        The CPU-bound processing kernel.
        """
        # 1. Clean
        text = self.cleaner.clean(text)
        
        # 2. Normalize (Tonality & Unicode)
        text = self.normalizer.normalize(text)
        
        # 3. English Normalization
        text = self.eng_norm.normalize(text)

        # 4. Word Segmentation (Compound Identification)
        text = self.segmenter.segment(text)
        
        return text

    def _tokenize(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def create_stream(self, file_paths: List[str], batch_size: int = 32):
        """
        Creates the MLX Stream with all processing steps attached.
        """
        return dx_pipeline(
            file_paths=file_paths,
            batch_size=batch_size,
            normalize_func=self._process_text, # Combines clean + normalize
            tokenizer_func=self._tokenize
        )
