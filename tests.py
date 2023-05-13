import unittest
from main import *
import tempfile

class MyTestCase(unittest.TestCase):
    def test_normalize_text(self):
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as temp:
            temp.write("This is the first sentence. Here's the second sentence!")
            temp_path = temp.name

        try:
            # Call normalize_text on the temp file
            sentences = normalize_text(temp_path)

            # Check the output
            expected_sentences = ['this is the first sentence', 'heres the second sentence']
            self.assertEqual(expected_sentences, sentences)
        finally:
            # Clean up by deleting the temporary file
            os.remove(temp_path)

    def test_init(self):
        sentences = [['this', 'is', 'a', 'first', 'sentence'], ['this', 'is', 'another', 'sentence'], ['a', 'third', 'sentence']]
        sg = SkipGram(sentences, word_count_threshold=2)

        # Check word count dictionary
        expected_word_counts = {'this': 2, 'is': 2, 'a': 2, 'sentence': 3}
        self.assertEqual(sg.word_counts, expected_word_counts)

        # Check word:index mapping
        expected_word_index = {'this': 0, 'is': 1, 'a': 2, 'sentence': 3}
        self.assertEqual(sg.word_index, expected_word_index)

        # Check index:word mapping
        expected_index_word = {0: 'this', 1: 'is', 2: 'a', 3: 'sentence'}
        self.assertEqual(sg.index_word, expected_index_word)


if __name__ == '__main__':
    unittest.main()
