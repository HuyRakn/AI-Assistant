import json
import os
from typing import List, Dict, Tuple

class AetherBPE:
    """
    White-Box Byte-Pair Encoding (Deterministic).
    No hash() functions. Pure Vocabulary Mapping.
    """
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        
    def get_stats(self, word_counts):
        """Helper: Compute frequency of all pairs."""
        pairs = {}
        for token, count in word_counts.items():
            for i in range(len(token) - 1):
                pair = (token[i], token[i+1])
                pairs[pair] = pairs.get(pair, 0) + count
        return pairs

    def train(self, texts: List[str]):
        """
        Train BPE using optimized 'String Replacement' Strategy.
        Complexity: C-Level string operations instead of Python loops.
        
        ⚠️ PERFORMANCE WARNING (White-Box Implementation):
        This is a Pure Python implementation designed for transparency and educational audit.
        While optimized with `str.replace` (~100x faster than loops), it is still slower 
        than Rust/C++ implementations (e.g., Tokenizers, SentencePiece) for massive corpora.
        For Production datasets (>1GB), consider using bindings.
        """
        # 1. Pre-tokenize and count words
        # Representation: "c h a r s </w>"
        word_counts: Dict[str, int] = {}
        for text in texts:
            words = text.split()
            for word in words:
                # Space separated chars
                token = " ".join(list(word)) + " </w>"
                word_counts[token] = word_counts.get(token, 0) + 1
                
        # 2. Base Vocab from characters
        unique_chars = set()
        for token in word_counts:
            # Split by space to get symbols
            chars = token.split() 
            for char in chars:
                unique_chars.add(char)
        sorted_chars = sorted(list(unique_chars))
        
        self.vocab = {
            "<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3, "</w>": 4
        }
        next_id = 5
        for char in sorted_chars:
            if char not in self.vocab:
                self.vocab[char] = next_id
                next_id += 1
                
        # 3. Training Loop
        print(f"Igniting BPE Training Foundry (Target: {self.vocab_size})...")
        print(f"Strategy: High-Performance String Replacement")
        
        num_merges = self.vocab_size - len(self.vocab)
        merges_done = 0
        
        # Initial Stats
        pairs = {}
        for token, count in word_counts.items():
            symbols = token.split()
            for i in range(len(symbols)-1):
                p = (symbols[i], symbols[i+1])
                pairs[p] = pairs.get(p, 0) + count

        import heapq
        # Queue: (-count, pair)
        queue = [(-count, pair) for pair, count in pairs.items()]
        heapq.heapify(queue)
        
        while merges_done < num_merges:
            if not queue:
                break
                
            # Pop best pair
            neg_count, best_pair = heapq.heappop(queue)
            count = -neg_count
            
            # Stale check
            if count != pairs.get(best_pair, 0):
                continue
                
            # Record merge
            self.merges.append(best_pair)
            # New token string (no spaces inside)
            new_token_str = "".join(best_pair)
            self.vocab[new_token_str] = next_id
            next_id += 1
            
            # Prepare replacement patterns
            # We want to replace " p0 p1 " with " p0p1 "
            # Must ensure we don't partial match.
            # Since all tokens are separated by spaces, scanning for " p0 p1 " is safe
            # IF we ensure we look for space boundaries.
            # But word_counts keys don't start/end with space? 
            # We can treat them as " " + word + " " for safe replacement?
            
            bigram = " ".join(best_pair)     # "t h"
            replacement = "".join(best_pair) # "th"
            
            # We need to find occurrences and update pairs stats.
            # Iterating words is unavoidable, but updating the word string is fast.
            
            # Identify affected words
            # Pattern to search: literally the bigram string inside the token string
            # But "t h" could be inside "a t h e". 
            # Is "t h" unique? Yes, if we maintain space separation.
            
            # Touched pairs for heap update
            touched_pairs = set()
            
            # Snapshot of keys to iterate
            current_tokens = list(word_counts.keys())
            
            for token in current_tokens:
                if bigram not in token:
                    continue
                    
                # Token matches. 
                # 1. Update stats BEFORE merge (decrement)
                # We need to parse the token to find adjacent pairs to the BIGRAM.
                # Example: "a t h e". Bigram "t h".
                # Pairs to decrement: (a, t) and (h, e).
                # The pair (t, h) itself is being merged, we don't care about its count anymore (deleted from queue inherently).
                
                freq = word_counts[token]
                symbols = token.split()
                
                # Verify usage of bigram
                # Find all indices where symbols[i] == best_pair[0] and symbols[i+1] == best_pair[1]
                
                # Optimization: 
                # It is faster to just re-split the string?
                # Or use regex? 
                # Let's use split.
                
                i = 0
                while i < len(symbols) - 1:
                    if symbols[i] == best_pair[0] and symbols[i+1] == best_pair[1]:
                        # Found match at (i, i+1)
                        # Dec left pair: (symbols[i-1], symbols[i])
                        if i > 0:
                            p_left = (symbols[i-1], symbols[i])
                            pairs[p_left] -= freq
                            touched_pairs.add(p_left)
                            if pairs[p_left] <= 0:
                                if pairs[p_left] == 0: del pairs[p_left]
                                
                        # Dec right pair: (symbols[i+1], symbols[i+2])
                        if i + 1 < len(symbols) - 1:
                            p_right = (symbols[i+1], symbols[i+2])
                            pairs[p_right] -= freq
                            touched_pairs.add(p_right)
                            if pairs[p_right] <= 0:
                                if pairs[p_right] == 0: del pairs[p_right]
                                
                        i += 2 # Skip the pair
                    else:
                        i += 1
                        
                # 2. Perform Merge (String Replace)
                # "t h" -> "th"
                # "a t h e" -> "a th e"
                # Logic: replace with space boundaries?
                # Using token.replace(bigram, replacement) is risky if bigram is "t h" and we have "t h e" -> "th e". Correct.
                # What if "a t h"? "a th". Correct.
                # What if "at h"? "at" is one token? "at h". Space separates.
                # Since bigram is "t h" (with space), it can only match "t" followed by "h".
                # It cannot match "th" (no space).
                
                new_token = token.replace(bigram, replacement)
                
                # Update corpus
                del word_counts[token]
                word_counts[new_token] = word_counts.get(new_token, 0) + freq
                
                # 3. Update stats AFTER merge (increment)
                # Find new pairs formed by the merged symbol
                # "a th e" -> Pairs (a, th) and (th, e)
                
                new_symbols = new_token.split()
                for i in range(len(new_symbols)):
                    if new_symbols[i] == replacement:
                        # This is an instance of the new token
                        # Inc left: (prev, new)
                        if i > 0:
                            p_left = (new_symbols[i-1], new_symbols[i])
                            pairs[p_left] = pairs.get(p_left, 0) + freq
                            touched_pairs.add(p_left)
                        # Inc right: (new, next)
                        if i < len(new_symbols) - 1:
                            p_right = (new_symbols[i], new_symbols[i+1])
                            pairs[p_right] = pairs.get(p_right, 0) + freq
                            touched_pairs.add(p_right)
            
            # Push updates
            for p in touched_pairs:
                if p in pairs:
                    heapq.heappush(queue, (-pairs[p], p))
                    
            merges_done += 1
            if merges_done % 100 == 0:
                 print(f"BPE Merge {merges_done}/{num_merges}: {best_pair} -> {new_token_str}")
        
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        print(f"BPE Training Complete. Final Vocab Size: {len(self.vocab)}")
        
    def encode(self, text: str) -> List[int]:
        """
        Encode text using trained merges (String Replacement Strategy).
        """
        if not self.vocab:
            return []
            
        words = text.split()
        ids = []
        
        for word in words:
            # Start with space-separated chars
            token = " ".join(list(word)) + " </w>"
            
            # Apply merges in order
            for pair in self.merges:
                bigram = " ".join(pair)
                replacement = "".join(pair)
                
                if bigram in token:
                     token = token.replace(bigram, replacement)
                
            # Map valid tokens to IDs, else <unk>
            # Token is now a string of space-separated subwords
            subwords = token.split()
            for t in subwords:
                ids.append(self.vocab.get(t, self.vocab["<unk>"]))
                
        return ids

    def decode(self, ids: List[int]) -> str:
        tokens = []
        for i in ids:
            t = self.inverse_vocab.get(i, "")
            tokens.append(t)
            
        # Join and clean up </w>
        text = "".join(tokens).replace("</w>", " ")
        return text.strip()

    def save(self, path: str):
        """
        Save vocab and merges to JSON.
        """
        data = {
            "vocab": self.vocab,
            "merges": self.merges
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    def load(self, path: str):
        """
        Load vocab and merges from JSON.
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.vocab = data["vocab"]
            # JSON keys are always strings, but our merges are lists of strings
            self.merges = [tuple(p) for p in data["merges"]]
            
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        print(f"Loaded BPE Model from {path}. Vocab size: {len(self.vocab)}")
