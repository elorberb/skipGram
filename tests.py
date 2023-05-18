import unittest
from main import *
import tempfile


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.d = 400  # vector size
        self.context = 4  # 2 words after and before
        self.neg_samples = 2  # 2 negative samples for 1 positive sample
        self.step_size = 0.0001  # weights updating

        self.sentences = ['Mary enjoys cooking',
                          'She likes bananas',
                          'They speak English at work',
                          'The train does not leave at 12 AM',
                          'I have no money at the moment',
                          'Do they talk a lot',
                          'Does she drink coffee',
                          'You run to the party']

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
        sentences = ['etay is a first sentence', 'etay is another sentence', 'a third sentence']
        sg = SkipGram(sentences, word_count_threshold=2)

        # Check word count dictionary
        expected_word_counts = {'etay': 2, 'sentence': 3}
        self.assertEqual(sg.word_count, expected_word_counts)

        # Check word:index mapping
        expected_word_index = {'etay': 0, 'sentence': 1}
        self.assertEqual(sg.word_index, expected_word_index)

    def test_cosine_similarity(self):
        cherry = [422, 8, 2]
        digital = [5, 1683, 1670]
        information = [5, 3982, 3325]

        # Check similarity between cherry and information
        similarity = SkipGram.cosine_similarity(cherry, information)
        self.assertAlmostEqual(similarity, 0.0185, places=2)

        # Check similarity between digital and information
        similarity = SkipGram.cosine_similarity(digital, information)
        print(similarity)
        self.assertAlmostEqual(similarity, 0.996, places=3)

    def test_preprocess_sentences(self):
        sg = SkipGram(self.sentences, neg_samples=2, word_count_threshold=0)
        le = sg.preprocess_sentences()

    def test_calculate_loss(self):
        sg = SkipGram(self.sentences, neg_samples=2, word_count_threshold=0)
        sg.vocab_size = 3
        y = np.array([0, 0, 1, 0])
        val = np.array([0.2, 0.7, 0.9, 0.3])
        expected_loss = 1.8891518152
        calculated_loss = sg.calculate_loss(y, val)
        print(calculated_loss)
        self.assertAlmostEqual(calculated_loss, expected_loss, places=5)

    def test_learn_embeddings(self):
        # Set up test data and parameters
        step_size = 0.0001
        epochs = 50
        early_stopping = 3
        model_path = "test_model.pkl"
        sg = SkipGram(self.sentences, neg_samples=2, word_count_threshold=0)
        T, C = sg.learn_embeddings(step_size, epochs, early_stopping, model_path)
        print(f"T shape:\n{T.shape}")
        print(f"C shape:\n{C.shape}")


    def test_combine_vectors(self):
        # Suppose that the 'combine_vectors' function is a method in the 'Embedding' class
        # Then you would create an instance of the Embedding class:
        # embedding = Embedding()
        # But for now, I will assume the function is standalone

        # Test combo = 0
        V = combine_vectors(self.T, self.C, combo=0)
        np.testing.assert_array_equal(V, self.T)

        # Test combo = 1
        V = combine_vectors(self.T, self.C, combo=1)
        np.testing.assert_array_equal(V, self.C.T)

        # Test combo = 2
        V = combine_vectors(self.T, self.C, combo=2)
        np.testing.assert_array_equal(V, (self.T + self.C.T) / 2)

        # Test combo = 3
        V = combine_vectors(self.T, self.C, combo=3)
        np.testing.assert_array_equal(V, self.T + self.C.T)

        # Test combo = 4
        V = combine_vectors(self.T, self.C, combo=4)
        np.testing.assert_array_equal(V, np.concatenate((self.T, self.C.T), axis=1))

        # Test saving the model
        combine_vectors(self.T, self.C, combo=0, model_path=self.model_path)
        self.assertTrue(os.path.exists(self.model_path))

        # Test invalid combo value
        with self.assertRaises(ValueError):
            combine_vectors(self.T, self.C, combo=5)


if __name__ == '__main__':
    unittest.main()
