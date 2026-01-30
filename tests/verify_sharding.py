import sys
import os
import shutil
import unittest
from unittest.mock import MagicMock, patch, mock_open

sys.path.append(os.getcwd()) # Add root to path

# Import module here, OUTSIDE of the patch context to avoid breaking libs like mlx
import scripts.build_corpus

class TestDataFactory(unittest.TestCase):
            
    @patch('scripts.build_corpus.stream_parquet_dataset')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.makedirs')
    def test_sharding_logic(self, mock_makedirs, m_open, mock_stream):
        # Mock stream to yield 15 records
        mock_stream.return_value = [{'text': b'Hello World'}] * 15
        
        # Run the corpus builder
        scripts.build_corpus.build_corpus()
        
        # Check if open was called with correct path convention
        # We expect "corpus_vi_part_000.txt"
        
        found = False
        for call in m_open.call_args_list:
            args, _ = call
            if len(args) > 0 and "corpus_vi_part_000.txt" in str(args[0]):
                found = True
                break
        
        self.assertTrue(found, "Did not find expected sharded filename 'corpus_vi_part_000.txt' in open() calls")
        print("✅ Sharding Filename Convention Verified.")

if __name__ == '__main__':
    unittest.main()
