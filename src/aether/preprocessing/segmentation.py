import os
import math
from typing import Set, List, Dict, Tuple

class VietnameseSegmenter:
    """
    White-Box Vietnamese Word Segmentation (Enterprise Edition).
    Algorithm: Viterbi (Statistical Parsing).
    Uses a DAG (Directed Acyclic Graph) and Unigram Counts to find the optimal segmentation.
    """
    def __init__(self, dict_path: str = None):
        self.word_counts: Dict[str, int] = {}
        self.total_count = 0
        self.max_word_len = 0
        
        if dict_path and os.path.exists(dict_path):
            self.load_dictionary(dict_path)
        else:
            # Fallback/Default path relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            default_path = os.path.join(base_dir, "data", "resources", "vn_compounds.txt")
            if os.path.exists(default_path):
                self.load_dictionary(default_path)
                
    def load_dictionary(self, path: str):
        """
        Load compound words and frequencies.
        Format: word1_word2 [count]
        If count missing, defaults to 10 (common word).
        """
        print(f"Loading Vietnamese Dictionary (Statistical) from {path}...")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    
                    # Parse format: "word [count]" or "word"
                    # Our resource file usually is just "machine_learning".
                    # We assume these are verified compounds.
                    
                    word = parts[0].replace(' ', '_').lower()
                    count = 10 # Default weight for dictionary words
                    
                    if len(parts) > 1 and parts[-1].isdigit():
                        count = int(parts[-1])
                        
                    self.word_counts[word] = count
                    self.total_count += count
                    
                    # Update max len (in syllables)
                    w_len = len(word.split('_'))
                    if w_len > self.max_word_len:
                        self.max_word_len = w_len
                        
        except Exception as e:
            print(f"Error loading dictionary: {e}")

    def get_prob(self, word: str) -> float:
        """Get log probability of a word."""
        # Smoothing: if unknown, use small count 1
        return math.log(self.word_counts.get(word, 1) / (self.total_count + 1e-9))

    def segment(self, text: str) -> str:
        """
        Segment text using Viterbi Algorithm.
        Finds the path with Maximum Likelihood.
        """
        if not text:
            return ""
            
        # 1. Normalize spaces
        text = " ".join(text.split())
        words = text.split()
        n = len(words)
        
        # 2. Build DAG & Viterbi DP
        # dp[i] = (max_log_prob, start_index_of_last_word)
        # We want to find best path to reach node i (after i-th syllable)
        
        # dp[i] stores the max log-prob to segment the prefix words[:i]
        # and the index j < i such that words[j:i] caused that max prob.
        
        dp = [-float('inf')] * (n + 1)
        path = [0] * (n + 1)
        
        dp[0] = 0 # Start with prob 0 (log(1))
        
        for i in range(1, n + 1):
            # Try all possible lookbacks up to max_word_len
            # From j to i
            start_j = max(0, i - self.max_word_len)
            
            for j in range(start_j, i):
                # Candidate word: words[j:i]
                w_list = words[j:i]
                word = "_".join(w_list).lower()
                clean_word = word.strip(".,!?;:()\"'")
                
                # Check if this valid transition
                # If single word (i-j==1), always valid (OOV fallback)
                # If compound, must be in dictionary (or logic for detected entities)
                
                is_in_dict = clean_word in self.word_counts
                
                if (i - j) == 1 or is_in_dict:
                    # Calculate cost
                    # If OOV single word, prob is low but valid
                    # If In-Dict single word, prob is from dict
                    
                    if is_in_dict:
                        prob = self.get_prob(clean_word)
                    else:
                        # OOV Penalty for single words
                        # Should be lower than dict words
                        prob = math.log(1.0 / (self.total_count * 100)) 
                    
                    current_score = dp[j] + prob
                    
                    if current_score > dp[i]:
                        dp[i] = current_score
                        path[i] = j
                        
        # 3. Backtrack
        segments = []
        curr = n
        while curr > 0:
            prev = path[curr]
            # Word is words[prev:curr]
            segment = "_".join(words[prev:curr])
            segments.append(segment)
            curr = prev
            
        return " ".join(segments[::-1])
