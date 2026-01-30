import unicodedata
import regex as re
from typing import Optional

class UnicodeFirewall:
    """
    The First Line of Defense.
    Enforces NFC (Normalization Form C) on all incoming text.
    Standardizes disparate Unicode representations into a canonical byte sequence.
    """
    @staticmethod
    def enforce_nfc(text: str) -> str:
        """
        Convert text to NFC standard. 
        Combines characters and tone marks into single code points where possible.
        """
        if not text:
            return ""
        return unicodedata.normalize('NFC', text)

class ViToneNormalizer:
    """
    Vietnamese Tone Normalization Engine (White-Box).
    Resolves the "hòa" vs "hoà" ambiguity by enforcing a strict 'New Style' rule:
    Tone mark must be placed on the main vowel of the syllable.
    """
    
    def __init__(self):
        # Vowels that can carry tone marks
        self.vowels = list("aeiouyâăêôơư")
        
        # Mapping for tone marks manipulation
        # Format: vowel -> [no_tone, acute, grave, hook, tilde, dot]
        # a -> [a, á, à, ả, ã, ạ]
        self.tone_map = {
            'a': ['a', 'á', 'à', 'ả', 'ã', 'ạ'],
            'ă': ['ă', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ'],
            'â': ['â', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ'],
            'e': ['e', 'é', 'è', 'ẻ', 'ẽ', 'ẹ'],
            'ê': ['ê', 'ế', 'ề', 'ể', 'ễ', 'ệ'],
            'i': ['i', 'í', 'ì', 'ỉ', 'ĩ', 'ị'],
            'o': ['o', 'ó', 'ò', 'ỏ', 'õ', 'ọ'],
            'ô': ['ô', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ'],
            'ơ': ['ơ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ'],
            'u': ['u', 'ú', 'ù', 'ủ', 'ũ', 'ụ'],
            'ư': ['ư', 'ứ', 'ừ', 'ử', 'ữ', 'ự'],
            'y': ['y', 'ý', 'ỳ', 'ỷ', 'ỹ', 'ỵ'],
        }
        
        # Reverse mapping for fast lookup
        self.char_to_tone_index = {}
        for vowel, variants in self.tone_map.items():
            for i, variant in enumerate(variants):
                self.char_to_tone_index[variant] = (vowel, i)

    def normalize(self, text: str) -> str:
        """
        Apply tone normalization to the entire text.
        Splits text into words and processes each word.
        """
        # Improved regex to handle Vietnamese words properly, keeping punctuation intact
        # This regex matches a sequence of word characters including Vietnamese diacritics
        words = re.split(r'(\W+)', text) 
        normalized_words = [self._normalize_word(w) if w.strip() else w for w in words]
        return "".join(normalized_words)

    def _normalize_word(self, word: str) -> str:
        """
        Normalize a single word.
        Logic: 
        1. Decompose to NFD to separate base chars and tone marks.
        2. Identify the vowel cluster.
        3. Identify the tone mark.
        4. Place the tone mark on the correct vowel according to 'New Style' rules.
        5. Recompose to NFC.
        """
        if not word:
            return ""
            
        # Fast check: if no vowel, return
        has_vowel = False
        for char in word:
            if char in self.char_to_tone_index or unicodedata.normalize('NFC', char) in self.char_to_tone_index:
                has_vowel = True
                break
        if not has_vowel:
            return word

        # 1. Extract Tone and Base Vowels
        tone_index = 0
        base_chars = []
        
        for char in word:
            # Check if char carries a tone
            nfc_char = unicodedata.normalize('NFC', char)
            if nfc_char in self.char_to_tone_index:
                vowel, idx = self.char_to_tone_index[nfc_char]
                base_chars.append(vowel)
                if idx > 0:
                    tone_index = idx # Capture the tone (assuming one tone per word)
            else:
                base_chars.append(char)
        
        if tone_index == 0:
            return word # No tone to shift (ngang tone)
            
        # 2. Identify Vowel Cluster within the word
        # We need to find the sequence of vowels to determine where to put the tone
        # Example: "thuy" -> vowels "uy"
        
        # Reconstruct word without tone to find vowel positions
        chars_no_tone = list(word)
        for i, c in enumerate(chars_no_tone):
            nfc_c = unicodedata.normalize('NFC', c)
            if nfc_c in self.char_to_tone_index:
                chars_no_tone[i] = self.char_to_tone_index[nfc_c][0]
        
        word_no_tone = "".join(chars_no_tone)
        
        # Regex to find vowel cluster
        # Vowels: a, e, i, o, u, y, and their variations
        # But we normalized them to base already in word_no_tone check logic above? 
        # Actually proper logic is complex using NFD. Let's simplify with robust detection.
        
        vowel_indices = []
        for i, char in enumerate(word_no_tone):
             if char in self.tone_map: # Simple base vowels check
                 vowel_indices.append(i)
                 
        if not vowel_indices:
            return word

        # 3. Determine 'Main Vowel' to verify logic
        # Logic for New Style (Chuẩn mới):
        # - OA, OE, UY -> Tone on 2nd vowel (hòa -> hoà, túy -> tuý)
        # - OA, OE, UY + Consonant -> Tone on 2nd vowel (hoan -> hoàn, tuyen -> tuyên)
        # - UOA -> Tone on 2nd vowel 'O' (huo -> huở)
        # - Exception: qu + vowel -> u is consistent, tone on following vowel.
        
        # Wait, strictly implementing the exact rule requires parsing.
        # A simpler robust way used in NLP:
        # If 2 vowels:
        #   If ending, usually tone on 1st: 'mía', 'hóa'.
        #   BUT 'oa', 'oe', 'uy' are special diphthongs requiring tone on 2nd in New Style?
        #   Let's check standard mapping.
        #   'hòa' (old) vs 'hoà' (new). 'thủy' (old) vs 'thuỷ' (new).
        
        # Actually, simpler algorithm:
        # 1. Strip tone.
        # 2. Put tone on the vowel determined by regex rules for New Style.
        
        return self._place_mark(word_no_tone, tone_index)

    def _place_mark(self, word_no_tone: str, tone_index: int) -> str:
        """
        Places the tone mark on the correct vowel according to New Style rules.
        """
        chars = list(word_no_tone)
        
        # Find all vowels
        vowels = []
        for i, c in enumerate(chars):
            if c in self.tone_map:
                vowels.append(i)
        
        if not vowels:
            return word_no_tone
            
        target_idx = -1
        
        if len(vowels) == 1:
            target_idx = vowels[0]
        elif len(vowels) == 2:
            # Logic for 2 vowels
            # Case 1: qu-, gi- (u and i strictly functioning as consonants/glides?)
            # Handle 'qu': qua -> q-u-a. Tone on 'a'.
            if chars[vowels[0]] == 'u' and (vowels[0] > 0 and chars[vowels[0]-1].lower() == 'q'):
                target_idx = vowels[1] # quỳ, quà
            # Handle 'gi': gia -> g-i-a. Tone on 'a'.
            elif chars[vowels[0]] == 'i' and (vowels[0] > 0 and chars[vowels[0]-1].lower() == 'g'):
                target_idx = vowels[1] # già
            else:
                 # General Diphthong logic
                 # Check if ending with consonant?
                 is_ending_open = (vowels[-1] == len(chars) - 1)
                 
                 v1, v2 = chars[vowels[0]], chars[vowels[1]]
                 
                 # oa, oe, uy -> tone on 2nd (hoà, hoè, thuỷ) irrespective of ending
                 if v1 == 'o' and v2 in ['a', 'e']:
                     target_idx = vowels[1]
                 elif v1 == 'u' and v2 == 'y':
                     target_idx = vowels[1]
                 # otherwise tone on 1st if ending open (mía, múa)? No, hóa -> hoá (New style)
                 # New style tends to push tone to the end of diphthong unless it is 'ia', 'ua', 'ưa'?
                 # ia -> ía (always 1st)
                 # ua -> úa (always 1st)
                 # ưa -> ứa (always 1st)
                 elif v2 == 'a' and v1 in ['i', 'u', 'ư']:
                     target_idx = vowels[0]
                 # Exception: ơi, êu -> tone on 1st (chơi, kêu)
                 elif v1 in ['ơ', 'ê'] and v2 in ['i', 'u']:
                     target_idx = vowels[0]
                 # Exception: ao, ay, au, ai -> tone on a (chào, ngày, sáu, cái)
                 elif v1 == 'a':
                     target_idx = vowels[0]
                 else:
                     # Default to 2nd vowel for New Style in most other cases (ie -> iệ)
                     # No, 'biển' (tone on e - 2nd). 'tuần' (tone on a - 2nd).
                     # 'toán' (tone on a - 2nd).
                     target_idx = vowels[1]

        elif len(vowels) == 3:
            # Triphthongs: usually tone on middle vowel.
            # uyê -> uyể (tone on 2nd: e)
            # yêuc -> yế (tone on 2nd: ê)
            # uôi -> uổi (tone on 2nd: ô)
            # ươi -> ưới (tone on 2nd: ư)
            target_idx = vowels[1]
            
        else:
            # Fallback
            target_idx = vowels[-1]

        if target_idx != -1:
            base_char = chars[target_idx]
            chars[target_idx] = self.tone_map[base_char][tone_index]
            
        return "".join(chars)
