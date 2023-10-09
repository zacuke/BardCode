class HuffmanNode:
  def __init__(self, symbol, frequency):
    self.symbol = symbol
    self.frequency = frequency
    self.left = None
    self.right = None

def huffman_coding(symbols, frequencies):
  """Generates a Huffman code for a list of symbols and frequencies.

  Args:
    symbols: A list of symbols.
    frequencies: A list of frequencies, corresponding to the symbols.

  Returns:
    A dictionary, mapping symbols to Huffman codes.
  """

  # Create a list of Huffman nodes, one for each symbol.
  nodes = []
  for i in range(len(symbols)):
    nodes.append(HuffmanNode(symbols[i], frequencies[i]))

  # Build a Huffman tree from the list of Huffman nodes.
  while len(nodes) > 1:
    nodes.sort(key=lambda node: node.frequency)

    # Merge the two nodes with the lowest frequencies.
    left_node = nodes.pop(0)
    right_node = nodes.pop(0)
    new_node = HuffmanNode(None, left_node.frequency + right_node.frequency)
    new_node.left = left_node
    new_node.right = right_node
    nodes.append(new_node)

  # Generate Huffman codes for the symbols.
  codes = {}
  def generate_codes(node, code):
    if node.symbol is not None:
      codes[node.symbol] = code
    else:
      generate_codes(node.left, code + "0")
      generate_codes(node.right, code + "1")

  generate_codes(nodes[0], "")

  return codes

class HuffmanTree:
  def __init__(self, symbol, frequency, left=None, right=None):
    self.symbol = symbol
    self.frequency = frequency
    self.left = left
    self.right = right

  @staticmethod
  def from_codes(codes):
    """Creates a Huffman tree from a list of Huffman codes.

    Args:
      codes: A list of Huffman codes.

    Returns:
      A Huffman tree.
    """

    nodes = []
    for code in codes:
      node = HuffmanTree(None, 1)
      for bit in code:
        if bit == "0":
          if node.left is None:
            node.left = HuffmanTree(None, 1)
          node = node.left
        else:
          if node.right is None:
            node.right = HuffmanTree(None, 1)
          node = node.right

      nodes.append(node)

    while len(nodes) > 1:
      nodes.sort(key=lambda node: node.frequency)

      left_node = nodes.pop(0)
      right_node = nodes.pop(0)
      new_node = HuffmanTree(None, left_node.frequency + right_node.frequency, left_node, right_node)
      nodes.append(new_node)

    return nodes[0]

  @property
  def root(self):
    """Returns the root node of the Huffman tree."""

    return self
