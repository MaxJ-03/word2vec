import unittest
from tree_utils import TreeNode, HuffmanTree

class TestTreeUtils(unittest.TestCase):
    def setUp(self):
        # A predictable mock dataset
        self.vocab = ['a', 'b', 'c', 'd']
        self.freqs = {'a': 5, 'b': 2, 'c': 1, 'd': 1}
        # Math Check: Total frequency of all words combined is 9

    def test_transform_vocabulary(self):
        nodes = TreeNode.transform_vocabulary_to_nodes(self.vocab, self.freqs)
        self.assertEqual(len(nodes), 4, "Should create exactly 4 leaf nodes")
        self.assertEqual(nodes[0].frequency, 5, "Node 'a' should have frequency 5")

    def test_huffman_tree_building(self):
        tree = HuffmanTree(self.vocab, self.freqs)
        self.assertIsNotNone(tree.root, "Tree root should not be None")
        self.assertEqual(tree.root.frequency, 9, "Root frequency must equal sum of all word frequencies")

    def test_convert_tree_to_dict(self):
        tree = HuffmanTree(self.vocab, self.freqs)
        huffman_dict = tree.convert_tree_to_dict(tree.root)

        # Check if all words are present and have the right dictionary keys
        for word in self.vocab:
            self.assertIn(word, huffman_dict)
            self.assertIn('code', huffman_dict[word])
            self.assertIn('path', huffman_dict[word])

        # The most frequent word ('a') MUST have a shorter code than the least frequent ('c')
        len_a = len(huffman_dict['a']['code'])
        len_c = len(huffman_dict['c']['code'])
        self.assertLess(len_a, len_c, "High frequency words must have shorter Huffman codes!")