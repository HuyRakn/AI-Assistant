import regex as re
import unicodedata
from typing import List

class DeepTextCleaner:
    """
    The Pre-Ingestion Filter (Vệ Sinh Dữ Liệu Thô).
    Removes HTML, fixes encoding errors (Mojibake), and standardizes whitespace.
    """
    
    @staticmethod
    def remove_html_tags(text: str) -> str:
        """
        White-Box implementation of HTML removal using Regex.
        Avoids external dependencies like BS4 for speed and control.
        """
        if not text:
            return ""
        
        # Remove scripts and styles first
        clean = re.sub(r'<(script|style).*?>.*?</\1>', '', text, flags=re.DOTALL)
        
        # Remove tags
        clean = re.sub(r'<[^>]+>', ' ', clean)
        
        # Decode HTML entities (e.g. &nbsp; -> space)
        # Using python's built-in html library would be cheating? 
        # No, standard lib is allowed. "NO API Dependencies" means external cloud APIs.
        import html
        clean = html.unescape(clean)
        
        return clean.strip()

    @staticmethod
    def fix_encoding_errors(text: str) -> str:
        """
        Heuristic-based Mojibake Fixer (Sửa lỗi font).
        Detects common encoding patterns where UTF-8 was interpreted as Windows-1252/Latin-1.
        """
        if not text:
            return ""
            
        try:
            # Common case: UTF-8 stream interpreted as Latin-1 or CP1252
            # "Tiáº¿ng Viá»‡t" -> "Tiếng Việt"
            
            # Simple heuristic: If text contains typical mojibake artifacts like 'Ã' followed by space or 'Â',
            # it's a strong specific signal.
            if 'Ã' in text or 'Â' in text or 'á' in text:
                # Advanced Logic: Hybrid Recovery for Windows-1252/Latin-1 mixed signals
                # Problem: 'ờ' (E1 BB 9D) has 0x9D which is undefined in CP1252 but valid in Latin-1 (Control).
                #          'đ' (C4 91) has 0x91 which maps to U+2018 (Quote) in CP1252 but is invalid in Latin-1 range.
                # Solution: Map CP1252 display chars back to their "byte" equivalents (Control chars), then encode Latin-1.
                
                # Map common CP1252 chars to their byte values (as latin-1 chars)
                cp1252_fix_map = {
                    '\u20ac': '\x80', '\u201a': '\x82', '\u0192': '\x83', '\u201e': '\x84',
                    '\u2026': '\x85', '\u2020': '\x86', '\u2021': '\x87', '\u02c6': '\x88',
                    '\u2030': '\x89', '\u0160': '\x8a', '\u2039': '\x8b', '\u0152': '\x8c',
                    '\u017d': '\x8e', '\u2018': '\x91', '\u2019': '\x92', '\u201c': '\x93',
                    '\u201d': '\x94', '\u2022': '\x95', '\u2013': '\x96', '\u2014': '\x97',
                    '\u02dc': '\x98', '\u2122': '\x99', '\u0161': '\x9a', '\u203a': '\x9b',
                    '\u0153': '\x9c', '\u017e': '\x9e', '\u0178': '\x9f'
                }
                
                temp_text = text
                for char, replacement in cp1252_fix_map.items():
                    temp_text = temp_text.replace(char, replacement)
                
                try:
                    # Now everything should be in Latin-1 range (00-FF)
                    # This recovers the original BYTES 
                    raw_bytes = temp_text.encode('latin1')
                    
                    # Now decode as UTF-8 to fix the Mojibake
                    fixed = raw_bytes.decode('utf-8')
                    
                    if len(fixed) < len(text):
                         return fixed
                except Exception:
                    # As a last resort, try direct CP1252 ignoring errors (lossy)
                    try:
                        fixed = text.encode('cp1252', errors='ignore').decode('utf-8')
                        if len(fixed) < len(text): return fixed
                    except:
                        pass
                    
            return text
        except Exception:
            return text

    @staticmethod
    def standardize_whitespace(text: str) -> str:
        """
        Collapses multiple spaces, tabs, newlines into single space.
        """
        if not text:
            return ""
        return " ".join(text.split())

    @staticmethod
    def clean(text: str) -> str:
        """
        Full cleaning pipeline.
        """
        t = DeepTextCleaner.remove_html_tags(text)
        t = DeepTextCleaner.fix_encoding_errors(t)
        t = DeepTextCleaner.standardize_whitespace(t)
        return t
