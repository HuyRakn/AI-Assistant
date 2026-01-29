import time
import mlx.data as dx
import numpy as np
import os
import sys
import psutil

# Add src to path
sys.path.append(os.path.abspath("src"))

from aether.data.normalization import UnicodeFirewall, ViToneNormalizer

def generate_synthetic_parquet(filename, num_rows=10000):
    """
    Generate a dummy parquet file with Vietnamese text for benchmarking.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    import random
    
    # Vocabulary for synthetic text
    words = ["Người", "Việt", "Nam", "thông", "minh", "hòa", "bình", "yêu", "nước", "lập", "trình", "AI", "học", "máy", "tương", "lai", "công", "nghệ"]
    
    texts = []
    print(f"Generating {num_rows} synthetic rows...")
    for _ in range(num_rows):
        # Generate random sentence length 10-50 words
        length = random.randint(10, 50)
        sentence = " ".join(random.choices(words, k=length))
        texts.append(sentence)
        
    table = pa.Table.from_pydict({"text": texts})
    pq.write_table(table, filename)
    return filename

def benchmark_pipeline():
    print("🚀 Starting Aether Data Factory Benchmark...")
    
    # 1. Setup
    dummy_file = "benchmark_data.parquet"
    if not os.path.exists(dummy_file):
        generate_synthetic_parquet(dummy_file, num_rows=50000) # ~1M tokens?
        
    normalizer = ViToneNormalizer()
    firewall = UnicodeFirewall()
    
    def complete_normalization(text):
        # Combine Firewall + Tone Normalization
        text = firewall.enforce_nfc(text)
        text = normalizer.normalize(text)
        return text

    # Load Real Tokenizer
    import sentencepiece as spm
    tokenizer_model = "models/tokenizer/aether_vi_titan_50k.model"
    if not os.path.exists(tokenizer_model):
        print("⚠️ Tokenizer model not found. Using dummy.")
        sp = None
    else:
        sp = spm.SentencePieceProcessor()
        sp.load(tokenizer_model)
        print(f"✅ Loaded White-Box Tokenizer: {tokenizer_model}")

    def real_tokenizer(text):
        if sp:
            return sp.encode_as_ids(text)
        return [1] * len(text.split())

    # 2. Build Pipeline
    from aether.data.ingestion import create_pipeline
    
    print("initializing Pipeline...")
    # High throughput config
    dset = create_pipeline(
        file_paths=[dummy_file],
        batch_size=512,
        normalize_func=complete_normalization,
        tokenizer_func=real_tokenizer,
        num_threads=8, 
        prefetch_size=32
    )
    
    # 3. Measure
    print("🔥 Running Stress Test...")
    start_time = time.time()
    total_tokens = 0
    total_batches = 0
    
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / 1024 / 1024
    
    # Iterate through the stream
    for batch in dset:
        # batch['text'] is numpy array of tokens if tokenizer used?
        # In our ingestion.py, we transformed 'text' to tokens array.
        # mlx.data returns dictionary of arrays.
        
        # Count tokens (sum of lengths)
        # Note: Padding might complicate valid count, but for raw throughput we count processed items.
        # Actually our ingestion returns ragged arrays or padded? 
        # mlx.data default is ragged if not strictly collimated?
        # Let's assume we count direct sizes.
        
        # For simplicity in this mock wrapper, we passed tokenizer that returns np.array.
        # mlx.data handles batching of variable length arrays.
        
        lengths = batch['text'].size 
        total_tokens += lengths
        total_batches += 1
        
        if total_batches % 10 == 0:
            print(f".", end="", flush=True)

    end_time = time.time()
    end_mem = process.memory_info().rss / 1024 / 1024
    
    duration = end_time - start_time
    tps = total_tokens / duration
    
    print(f"\n\n📊 RESULTS:")
    print(f"   Duration: {duration:.2f}s")
    print(f"   Total Tokens Processed: {total_tokens}")
    print(f"   Throughput: {tps:,.2f} tokens/sec")
    print(f"   Memory Delta: {end_mem - start_mem:.2f} MB")
    
    # Cleanup
    if os.path.exists(dummy_file):
        os.remove(dummy_file)
        
    if tps > 100000:
        print("✅ Goal Met: >100k TPS")
    else:
        print("⚠️ Goal Missed: Optimize Pipeline")

if __name__ == "__main__":
    benchmark_pipeline()
