from typing import List, Iterator, Callable
import mlx.data as dx
from ..data.ingestion import create_pipeline as dx_pipeline
from ..data.normalization import ViToneNormalizer
from .cleaning import DeepTextCleaner
# Import MinHash if integrated, but usually Dedup is a separate batch process.
# Pipeline here focuses on "Ingestion to Training Ready".

class AetherDataFactory:
    """
    The Orchestrator.
    Constructs the end-to-end data processing assembly line.
    Step 1: Raw Parquet -> Step 2: Clean -> Step 3: Normalize -> Step 4: Tokenize
    """
    
    def __init__(self, tokenizer_model_path: str = None):
        self.cleaner = DeepTextCleaner()
        self.normalizer = ViToneNormalizer()
        
        # Load Tokenizer using SentencePiece
        if tokenizer_model_path:
            import sentencepiece as spm
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(tokenizer_model_path)
        else:
            self.sp = None
            print("Warning: AetherDataFactory initialized without tokenizer. Output will be raw text.")

    def _process_text(self, text: str) -> str:
        """
        The CPU-bound processing kernel.
        """
        # 1. Clean
        text = self.cleaner.clean(text)
        
        # 2. Normalize (Tonality & Unicode)
        text = self.normalizer.normalize(text)
        
        return text

    def _tokenize(self, text: str) -> List[int]:
        if self.sp:
            return self.sp.encode_as_ids(text)
        return []

    def create_stream(self, file_paths: List[str], batch_size: int = 32):
        """
        Creates the MLX Stream with all processing steps attached.
        """
        return dx_pipeline(
            file_paths=file_paths,
            batch_size=batch_size,
            normalize_func=self._process_text, # Combines clean + normalize
            tokenizer_func=self._tokenize if self.sp else None
        )
