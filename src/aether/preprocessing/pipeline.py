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
        
        # Load Tokenizer using Sovereign Tokenizer (Linked List BPE)
        from ..tokenization.core_bpe import SovereignTokenizer
        self.tokenizer = SovereignTokenizer()
        
        if tokenizer_model_path:
             self.tokenizer.load(tokenizer_model_path)
        else:
             print("Warning: AetherDataFactory initialized without loaded tokenizer.")
             
        # Dedup Engine
        self.dedup_engine = MinHashLSH()

    def stream_deduplication(self, text_iterator: Iterator[str], batch_size: int = 10000) -> Iterator[str]:
        """
        Streaming Deduplication (Generator).
        Uses Redis-backed MinHashLSH to enforce Global Uniqueness across all batches.
        
        Args:
            text_iterator: Generator or Iterator yielding strings.
            batch_size: Number of docs to process per LSH update.
            
        Yields:
            Unique documents.
        """
        batch = []
        print(f"Starting Streaming Deduplication (Batch Size: {batch_size})...")
        print("   - Backend: Redis (Global Persistent Index)")
        
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
        """Internal batch processor. Checks against Global Redis Index."""
        # 1. Compute Signatures
        # Parallel compute signatures first (MLX efficiency)
        signatures = [self.dedup_engine.compute_signature(t) for t in batch]
        
        unique_batch_docs = []
        
        # 2. Check against Global Index
        import hashlib
        
        for i, sig in enumerate(signatures):
            # Check if exists in history (Redis)
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
        
        # 3. English Normalization (Always run, safe for VN too as it targets english chars)
        text = self.eng_norm.normalize(text)

        # 4. Language Routing (White-Box Heuristic)
        # Check for Vietnamese-specific tone marks or 'đ'.
        # If detected, apply Word Segmentation.
        # If not, assume English/Code and skip segmentation to preserve structure.
        VN_SPECIFIC_CHARS = set("àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ")
        
        # Fast check: set intersection generally fast enough for streaming
        # Optimization: Check a sample or use regex? Set intersection on string is O(N).
        # Given pipeline is CPU bound, this is acceptable for correctness.
        
        is_vietnamese = any(c in VN_SPECIFIC_CHARS for c in text.lower())
        
        if is_vietnamese:
             # Word Segmentation (Compound Identification)
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
