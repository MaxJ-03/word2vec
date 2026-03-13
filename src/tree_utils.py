#file to store all tree related classes and methods

class TreeNode:
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
        counter = 0
        #build the huffman tree using the nodes
        while len(nodes) > 1:
            # sort the nodes by frequency
            nodes = sorted(nodes, key=lambda x: x.frequency)
            #combine the two nodes with the lowest frequency
            left_child = nodes[0]
            right_child = nodes[1]
            parent_node = TreeNode(
                word=None, 
                frequency= left_child.frequency + right_child.frequency, 
                id = counter) 
            parent_node.add_left_child(left_child)
            parent_node.add_right_child(right_child)
            #remove the two nodes and add the parent node to the list of nodes
            nodes = nodes[2:]
            nodes.append(parent_node)
            counter += 1
        self.root = nodes[0]

    def convert_tree_to_dict(self, node, code=None, path=None, huffman_dict=None):
        if huffman_dict is None:
            huffman_dict = {}
        if code is None:
            code = []
        if path is None:
            path = []
    
        if node is not None:
            if node.word is not None:
                huffman_dict[node.word] = {
                    'code': code,
                    'path': path
                }
            else:
                self.convert_tree_to_dict(node.left_child, code + [1], path + [node.id],  huffman_dict)
                self.convert_tree_to_dict(node.right_child, code + [-1], path + [node.id] , huffman_dict)

        return huffman_dict