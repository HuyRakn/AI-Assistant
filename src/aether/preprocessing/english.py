import regex as re

class EnglishNormalizer:
    """
    White-Box English Normalization.
    Implements FULL Porter Stemmer manually without NLTK/Spacy dependencies.
    Reference: Martin Porter, 1980.
    """
    
    @staticmethod
    def is_consonant(word, i):
        c = word[i]
        if c in 'aeiou': return False
        if c == 'y':
            if i == 0: return True
            else: return not EnglishNormalizer.is_consonant(word, i-1)
        return True

    @staticmethod
    def measure(word):
        # Calculate m in [C](VC){m}[V]
        # measure is number of VC sequences
        m = 0
        i = 0
        length = len(word)
        if length == 0: return 0
        
        while i < length:
            if not EnglishNormalizer.is_consonant(word, i): break
            i += 1
        while i < length:
            while i < length and not EnglishNormalizer.is_consonant(word, i): i += 1 # skip V
            if i == length: break
            while i < length and EnglishNormalizer.is_consonant(word, i): i += 1 # skip C
            m += 1
        return m

    @staticmethod
    def contains_vowel(word):
        for i in range(len(word)):
            if not EnglishNormalizer.is_consonant(word, i): return True
        return False
        
    @staticmethod
    def ends_double_consonant(word):
        if len(word) < 2: return False
        return word[-1] == word[-2] and EnglishNormalizer.is_consonant(word, len(word)-1)

    @staticmethod
    def ends_cvc(word):
        # cvc where second c is not w, x, y
        if len(word) < 3: return False
        c1 = EnglishNormalizer.is_consonant(word, len(word)-3)
        v  = not EnglishNormalizer.is_consonant(word, len(word)-2)
        c2 = EnglishNormalizer.is_consonant(word, len(word)-1)
        last = word[-1]
        return c1 and v and c2 and (last not in 'wxy')

    @staticmethod
    def step1a(word):
        if word.endswith('sses'): return word[:-2]
        if word.endswith('ies'): return word[:-2]
        if word.endswith('ss'): return word
        if word.endswith('s'): return word[:-1]
        return word

    @staticmethod
    def step1b(word):
        flag = False
        if word.endswith('eed'):
            stem = word[:-3]
            if EnglishNormalizer.measure(stem) > 0:
                word = stem + 'ee'
        elif word.endswith('ed'):
            stem = word[:-2]
            if EnglishNormalizer.contains_vowel(stem):
                word = stem
                flag = True # successful removal
        elif word.endswith('ing'):
            stem = word[:-3]
            if EnglishNormalizer.contains_vowel(stem):
                word = stem
                flag = True
        
        if flag:
            if word.endswith('at'): word += 'e'
            elif word.endswith('bl'): word += 'e'
            elif word.endswith('iz'): word += 'e'
            elif EnglishNormalizer.ends_double_consonant(word) and not word.endswith('l') and not word.endswith('s') and not word.endswith('z'):
                word = word[:-1]
            elif EnglishNormalizer.measure(word) == 1 and EnglishNormalizer.ends_cvc(word):
                word += 'e'
        return word

    @staticmethod
    def step1c(word):
        if word.endswith('y'):
            stem = word[:-1]
            if EnglishNormalizer.contains_vowel(stem):
                word = stem + 'i'
        return word

    @staticmethod
    def step2(word):
        # Map suffix -> replacement
        # Order matters: longest match first
        suffixes = [
            ('ational', 'ate'), ('tional', 'tion'), ('enci', 'ence'), ('anci', 'ance'),
            ('izer', 'ize'), ('bli', 'ble'), ('alli', 'al'), ('entli', 'ent'),
            ('eli', 'e'), ('ousli', 'ous'), ('ization', 'ize'), ('ation', 'ate'),
            ('ator', 'ate'), ('alism', 'al'), ('iveness', 'ive'), ('fulness', 'ful'),
            ('ousness', 'ous'), ('aliti', 'al'), ('iviti', 'ive'), ('biliti', 'ble'),
            ('logi', 'log')
        ]
        for s, r in suffixes:
            if word.endswith(s):
                stem = word[:-len(s)]
                if EnglishNormalizer.measure(stem) > 0:
                    return stem + r
        return word

    @staticmethod
    def step3(word):
        suffixes = [
            ('icate', 'ic'), ('ative', ''), ('alize', 'al'), ('iciti', 'ic'),
            ('ical', 'ic'), ('ful', ''), ('ness', '')
        ]
        for s, r in suffixes:
            if word.endswith(s):
                stem = word[:-len(s)]
                if EnglishNormalizer.measure(stem) > 0:
                    return stem + r
        return word

    @staticmethod
    def step4(word):
        suffixes = [
            'al', 'ance', 'ence', 'er', 'ic', 'able', 'ible', 'ant', 'ement',
            'ment', 'ent', 'ou', 'ism', 'ate', 'iti', 'ous', 'ive', 'ize'
        ]
        # logic: delete suffix if m > 1
        for s in suffixes:
            if word.endswith(s):
                stem = word[:-len(s)]
                if EnglishNormalizer.measure(stem) > 1:
                    return stem
        
        # special cases need to be handled if generic didn't match? No, Porter spec is sequential check.
        # But 'ion' is conditional on s or t.
        if word.endswith('ion'):
            stem = word[:-3]
            if EnglishNormalizer.measure(stem) > 1 and (stem.endswith('s') or stem.endswith('t')):
                return stem
        return word

    @staticmethod
    def step5a(word):
        if word.endswith('e'):
            stem = word[:-1]
            m = EnglishNormalizer.measure(stem)
            if m > 1: return stem
            if m == 1 and not EnglishNormalizer.ends_cvc(stem): return stem
        return word

    @staticmethod
    def step5b(word):
        if word.endswith('ll') and EnglishNormalizer.measure(word[:-1]) > 1:
            return word[:-1]
        return word

    @staticmethod
    def stem(word: str) -> str:
        w = word.lower()
        if len(w) <= 2: return w
        
        w = EnglishNormalizer.step1a(w)
        w = EnglishNormalizer.step1b(w)
        w = EnglishNormalizer.step1c(w)
        w = EnglishNormalizer.step2(w)
        w = EnglishNormalizer.step3(w)
        w = EnglishNormalizer.step4(w)
        w = EnglishNormalizer.step5a(w)
        w = EnglishNormalizer.step5b(w)
        return w

    @staticmethod
    def case_fold(text: str) -> str:
        return text.lower() 

    @staticmethod
    def normalize(text: str) -> str:
        """
        Smart Normalization with Truecasing Heuristics.
        - Preserves Acronyms (USA, AI)
        - Preserves Proper Nouns mid-sentence (Apple, Google)
        - Lowercases and Stems only common words.
        """
        if not text:
            return ""

        # 1. Tokenize keeping punctuation to detect sentence boundaries
        # Match words (including contractions) OR punctuation
        tokens = re.findall(r"[\w']+|[.,!?;]", text)
        
        out = []
        is_sentence_start = True
        
        for token in tokens:
            # Check if token is punctuation
            if token in ".,!?;":
                if token in ".!?":
                    is_sentence_start = True
                continue # Skip punctuation in output (or keep? Requirement said "Standardize whitespace" usually removes them)
                         # Original code replaced punct with space. Let's stick to space-separated words.
            
            # Smart Case Logic
            final_token = token
            should_stem = True
            
            if len(token) > 1 and token.isupper():
                # Acronym (USA, AI) -> Keep as is, No Stem
                should_stem = False
                
            elif token.istitle():
                if is_sentence_start:
                    # Ambiguous (Start of sentence). 
                    # Aggressive approach: Lowercase & Stem (assume common word).
                    # "Apple is red" -> apple
                    final_token = token.lower()
                    should_stem = True
                else:
                    # Mid-sentence Title -> Proper Noun (Apple) -> Keep, No Stem
                    should_stem = False
            else:
                # Lowercase regular words
                final_token = token.lower()
                should_stem = True
            
            # Apply Stemming if needed
            if should_stem:
                final_token = EnglishNormalizer.stem(final_token)
                
            out.append(final_token)
            
            # After processing a word, it's no longer start of sentence
            is_sentence_start = False
            
        return " ".join(out)
