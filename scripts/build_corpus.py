import sys
import os
import mlx.data as dx
import time

# Add src to path
sys.path.append(os.path.abspath("src"))

from aether.data.ingestion import create_pipeline, stream_parquet_dataset
from aether.data.normalization import UnicodeFirewall, ViToneNormalizer

def build_corpus():
    print("🏭 Aether Data Factory: Starting Processing Line...")
    
    # Paths
    input_file = "data/raw/wikipedia/wiki_vi.parquet"
    output_txt = "data/processed/clean_corpus_vi.txt"
    output_parquet = "data/processed/clean_corpus_vi.parquet"
    
    os.makedirs("data/processed", exist_ok=True)
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return

    # Initialize Normalizers
    firewall = UnicodeFirewall()
    tone_norm = ViToneNormalizer()
    
    def process_text(text):
        # 1. Unicode Normalization (NFC)
        text = firewall.enforce_nfc(text)
        # 2. Tone Normalization (New Style)
        text = tone_norm.normalize(text)
        # 3. Basic Cleaning (replace newlines for TXT format if needed, but SP handles it)
        # Let's keep newlines but maybe collapse multiple spaces
        return " ".join(text.split())

    print(f"   Input: {input_file}")
    print("   Applying: Unicode Firewall + ViToneNormalizer")
    
    # We use our own generator loop instead of create_pipeline because we want to Write to disk,
    # not just stream to GPU. mlx.data is great for training loop, but for ETL to disk, 
    # a simple efficient loop using our normalization logic is sufficient and easier to control output format.
    
    start = time.time()
    count = 0
    
    with open(output_txt, 'w', encoding='utf-8') as f_txt:
        # Stream from Parquet using our helper
        stream = stream_parquet_dataset([input_file])
        
        for record in stream:
             # decode bytes from ingestion stream
             raw_text = record['text'].decode('utf-8')
             
             if len(raw_text) < 100: # Filter short articles
                 continue
                 
             clean_text = process_text(raw_text)
             
             # Write to text file (one doc per line or just concat? SP assumes one sentence per line preferably)
             # But for wiki articles, preserving structure is good. 
             # Let's write raw text with newlines escaped or just as stream?
             # SentencePiece trains on raw sentences.
             # Let's replace actual newlines with space to make "one doc per line" or "one paragraph per line"
             
             f_txt.write(clean_text + "\n")
             count += 1
             
             if count % 10000 == 0:
                 print(f"   Processed {count} docs...", end='\r')
                 
    print(f"\n✅ Processing Complete.")
    print(f"   Docs: {count}")
    print(f"   Time: {time.time() - start:.2f}s")
    print(f"   Output (TXT): {output_txt}")
    
    # We could also save Parquet here if we wanted 'id' tracking, but for now TXT is priority for Tokenizer.

if __name__ == "__main__":
    build_corpus()
