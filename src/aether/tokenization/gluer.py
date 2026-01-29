import math
import collections
from typing import Dict, List, Set, Tuple
from tqdm import tqdm

class CompoundGluer:
    """
    Titan Linguistic Engine.
    Uses Pointwise Mutual Information (PMI) to discover and glue compound words.
    Prevents "Tokenizer Blindness" by forcing important multi-word terms to be tokenized as one.
    """
    
    def __init__(self, min_count: int = 10, threshold: float = 0):
        """
        Args:
            min_count: Min frequency of a bigram to be considered.
            threshold: Min PMI score to glue. 
                       PMI = log2( P(x,y) / (P(x)*P(y)) )
        """
        self.min_count = min_count
        self.threshold = threshold
        self.bigram_counts = collections.defaultdict(int)
        self.unigram_counts = collections.defaultdict(int)
        self.total_bigrams = 0
        self.vocab: Set[str] = set()
        
    def train(self, corpus_path: str):
        """
        First Pass: Count Statistics.
        """
        print(f"🔥 Gluer: Scanning corpus stats from {corpus_path}...")
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                words = line.strip().split()
                if not words: continue
                
                # Count Unigrams
                for w in words:
                    self.unigram_counts[w] += 1
                    
                # Count Bigrams
                for i in range(len(words) - 1):
                    bigram = (words[i], words[i+1])
                    self.bigram_counts[bigram] += 1
                    self.total_bigrams += 1
                    
        print(f"   Unigrams: {len(self.unigram_counts)}")
        print(f"   Bigrams: {len(self.bigram_counts)}")
        
    def calculate_pmi(self, w1, w2) -> float:
        # P(x, y)
        prob_xy = self.bigram_counts[(w1, w2)] / self.total_bigrams
        
        # P(x), P(y). Approx total unigrams ~ total bigrams for large corpus
        prob_x = self.unigram_counts[w1] / self.total_bigrams 
        prob_y = self.unigram_counts[w2] / self.total_bigrams
        
        if prob_x == 0 or prob_y == 0: return -math.inf
        
        return math.log2(prob_xy / (prob_x * prob_y))

    def glue(self, input_path: str, output_path: str):
        """
        Second Pass: Rewrite Corpus with glued words.
        """
        print(f"🌪️  Gluer: Gluing compounds...")
        
        # Cache useful bigrams to avoid re-calc
        glue_set = set()
        # Pre-calculate qualifying bigrams
        for bigram, count in self.bigram_counts.items():
            if count < self.min_count: continue
            pmi = self.calculate_pmi(bigram[0], bigram[1])
            if pmi > self.threshold:
                glue_set.add(bigram)
                
        print(f"   Found {len(glue_set)} compound types to glue.")
        
        with open(input_path, 'r', encoding='utf-8') as fin, \
             open(output_path, 'w', encoding='utf-8') as fout:
            
            for line in tqdm(fin):
                words = line.strip().split()
                new_words = []
                i = 0
                while i < len(words):
                    if i < len(words) - 1:
                        bigram = (words[i], words[i+1])
                        if bigram in glue_set:
                            # Glue!
                            new_words.append(f"{words[i]}_{words[i+1]}")
                            i += 2 # Skip next word
                            continue
                            
                    new_words.append(words[i])
                    i += 1
                    
                fout.write(" ".join(new_words) + "\n")
                
        print(f"✅ Glued Corpus saved to {output_path}")

if __name__ == "__main__":
    pass
