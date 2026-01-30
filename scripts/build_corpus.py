import sys
import os
import mlx.data as dx
import time

# Add src to path
sys.path.append(os.path.abspath("src"))

from aether.data.ingestion import create_pipeline, stream_parquet_dataset
from aether.data.normalization import UnicodeFirewall, ViToneNormalizer
from aether.preprocessing.english import EnglishNormalizer

def build_corpus():
    print("🏭 Aether Data Factory: Starting Processing Line (Enterprise Sharding)...")
    
    # Paths
    input_file = "data/raw/wikipedia/wiki_vi.parquet"
    output_dir = "data/processed/shards"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return

    # Initialize Normalizers
    firewall = UnicodeFirewall()
    tone_norm = ViToneNormalizer()
    eng_norm = EnglishNormalizer()
    
    def process_text(text):
        text = firewall.enforce_nfc(text)
        text = tone_norm.normalize(text)
        # Apply English Normalization (Stemming for mixed content)
        # Note: This will stem english words embedded in VN text.
        text = eng_norm.normalize(text)
        return " ".join(text.split())

    print(f"   Input: {input_file}")
    
    start = time.time()
    total_docs = 0
    shard_size = 100000 # Docs per shard
    current_shard_idx = 0
    current_shard_docs = 0
    
    # Open first shard
    f_out = open(f"{output_dir}/corpus_vi_part_{current_shard_idx:03d}.txt", 'w', encoding='utf-8')
    
    # Stream from Parquet
    stream = stream_parquet_dataset([input_file])
    
    try:
        for record in stream:
             raw_text = record['text'].decode('utf-8')
             
             if len(raw_text) < 100: continue
                 
             clean_text = process_text(raw_text)
             
             f_out.write(clean_text + "\n")
             total_docs += 1
             current_shard_docs += 1
             
             # Rotate Shard
             if current_shard_docs >= shard_size:
                 f_out.close()
                 print(f"   Shard {current_shard_idx:03d} filled ({current_shard_docs} docs).")
                 current_shard_idx += 1
                 current_shard_docs = 0
                 f_out = open(f"{output_dir}/corpus_vi_part_{current_shard_idx:03d}.txt", 'w', encoding='utf-8')
             
             if total_docs % 10000 == 0:
                 print(f"   Processed {total_docs} docs...", end='\r')
                 
    finally:
        f_out.close()
        
    print(f"\n✅ Processing Complete.")
    print(f"   Total Docs: {total_docs}")
    print(f"   Shards Created: {current_shard_idx + (1 if current_shard_docs > 0 else 0)}")
    print(f"   Time: {time.time() - start:.2f}s")

if __name__ == "__main__":
    build_corpus()
