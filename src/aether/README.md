# Aether: Sentient Core (White-Box)

This directory contains the source code for Project Aether, a ground-up implementation of a Large Language Model architecture on Apple Silicon.

## Architecture Modules

### `data/` (The Data Factory)
- **`ingestion.py`**: Zero-Copy MLX Pipeline reading from Parquet streams.
- **`normalization.py`**: 
    - `UnicodeFirewall`: Enforces NFC standard.
    - `ViToneNormalizer`: Canonical tone mark placement for Vietnamese.

### `dedup/` (The Filter)
- **`minhash.py`**: MinHash LSH implementation for fuzzy deduplication.

### `tokenization/` (The Vocabulary)
- **`trainer.py`**: Foundry for training custom BPE models using SentencePiece.

## Compliance
- **Encoding**: UTF-8 NFC Strict.
- **Acceleration**: Metal/MLX for matrix operations.
