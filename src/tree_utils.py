#file to store all tree related classes and methods
import heapq

class TreeNode:
    __slots__ = ('word', 'frequency', 'left_child', 'right_child', 'id')

    def __init__(self, word, frequency=None, id=None):
        self.word = word
        self.frequency = frequency
        self.left_child = None
        self.right_child = None
        self.id = id
    
    def add_left_child(self, child_node):
        self.left_child = child_node

    def add_right_child(self, child_node):
        self.right_child = child_node

    def set_id(self, id):
        self.id = id

    @staticmethod
    def transform_vocabulary_to_nodes(vocabulary, vocabulary_frequency):
        #create a node for each word in the vocabulary
        nodes = []
        for i, word in enumerate(vocabulary):
            node = TreeNode(
                word = word,
                frequency = vocabulary_frequency[word]
            )
            nodes.append(node)
        
        return nodes


class HuffmanTree:
    def __init__(self, vocabulary, vocabulary_frequency):
        self.vocabulary = vocabulary
        self.vocabulary_frequency = vocabulary_frequency
        self.root = None
        self.build_tree(TreeNode.transform_vocabulary_to_nodes(vocabulary, vocabulary_frequency))
    
    def build_tree(self, nodes):
        if not nodes:
            self.root = None
            return

        # Min-heap turns repeated global sorts into O(V log V) total merging work.
        heap = [(node.frequency, i, node) for i, node in enumerate(nodes)]
        heapq.heapify(heap)

        parent_id = 0
        tie_breaker = len(heap)

        while len(heap) > 1:
            _, _, left_child = heapq.heappop(heap)
            _, _, right_child = heapq.heappop(heap)

            parent_node = TreeNode(
                word=None,
                frequency=left_child.frequency + right_child.frequency,
                id=parent_id,
            )
            parent_node.add_left_child(left_child)
            parent_node.add_right_child(right_child)

            heapq.heappush(heap, (parent_node.frequency, tie_breaker, parent_node))
            parent_id += 1
            tie_breaker += 1

        self.root = heap[0][2]

    def convert_tree_to_dict(self, node, code=None, path=None, huffman_dict=None):
        if huffman_dict is None:
            huffman_dict = {}
        if node is None:
            return huffman_dict

        initial_code = [] if code is None else code
        initial_path = [] if path is None else path

        # Iterative DFS avoids recursion overhead for large vocabularies.
        stack = [(node, initial_code, initial_path)]

        while stack:
            current, current_code, current_path = stack.pop()

            if current.word is not None:
                huffman_dict[current.word] = {
                    'code': current_code,
                    'path': current_path
                }
                continue

            next_path = current_path + [current.id]

            # Push right first so left branch is processed first (same traversal order as recursion).
            if current.right_child is not None:
                stack.append((current.right_child, current_code + [-1], next_path))
            if current.left_child is not None:
                stack.append((current.left_child, current_code + [1], next_path))

        return huffman_dict