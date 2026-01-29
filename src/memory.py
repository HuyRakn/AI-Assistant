
import chromadb
from chromadb.utils import embedding_functions
from src.logger import setup_logger
import os

logger = setup_logger("Memory")

class MemorySystem:
    def __init__(self, persist_path: str = "braindata/memory"):
        """
        Initializes the Semantic Memory System (Vector DB).
        """
        logger.info(f"Initializing Memory System at: {persist_path}")
        
        # Ensure directory exists
        os.makedirs(persist_path, exist_ok=True)
        
        try:
            self.client = chromadb.PersistentClient(path=persist_path)
            
            # Using default embedding function (all-MiniLM-L6-v2) for stability first
            # It runs locally via ONNX
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
            
            self.collection = self.client.get_or_create_collection(
                name="aether_context",
                embedding_function=self.embedding_fn
            )
            logger.info(f"Memory Collection loaded. Items: {self.collection.count()}")
            
        except Exception as e:
            logger.critical(f"Failed to initialize Memory System: {e}")
            raise

    def _redact_pii(self, text: str) -> str:
        """
        Simple regex-based PII redaction rule.
        Removes emails, phone numbers, and potential API keys/passwords.
        """
        import re
        # Email
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[REDACTED_EMAIL]', text)
        # Phone (Simple VN format)
        text = re.sub(r'(0|\+84)(3|5|7|8|9)[0-9]{8}', '[REDACTED_PHONE]', text)
        # API Keys (Simple heuristic: long alphanumeric strings with mixed case)
        text = re.sub(r'(sk-[a-zA-Z0-9]{32,})', '[REDACTED_KEY]', text)
        return text

    def save_memory(self, text: str, metadata: dict = None):
        """
        Saves a text snippet to long-term memory.
        """
        try:
            # Security: Sanitize PII before saving
            safe_text = self._redact_pii(text)
            
            # We use current timestamp as ID
            import time
            timestamp = str(time.time())
            
            # Chroma requires non-empty metadata in some versions
            safe_metadata = metadata or {}
            safe_metadata["timestamp"] = timestamp
            
            self.collection.add(
                documents=[safe_text],
                metadatas=[safe_metadata],
                ids=[timestamp]
            )
            logger.debug(f"Saved memory: {safe_text[:30]}...")
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")

    def retrieve_context(self, query: str, n_results: int = 3) -> list:
        """
        Retrieves relevant context for a query.
        Returns a list of strings.
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Chroma returns dictionary of lists
            documents = results['documents'][0]
            return documents
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            return []

    def clear_memory(self):
        """Wipes the memory (Dangerous)"""
        try:
            self.client.delete_collection("aether_context")
            self.collection = self.client.get_or_create_collection(
                name="aether_context",
                embedding_function=self.embedding_fn
            )
            logger.warning("Memory wiped.")
        except Exception as e:
            logger.error(f"Failed to wipe memory: {e}")
