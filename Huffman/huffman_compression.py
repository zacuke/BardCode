import huffman_coding
import os

def compress_file(input_file, output_file):
  """Compresses a file using Huffman coding.

  Args:
    input_file: The path to the input file.
    output_file: The path to the output file.
  """
  cwd = os.path.abspath(os.getcwd())
  input_file_path = os.path.join(os.getcwd(), input_file)
  # Read the input file.
  with open(input_file_path, "rb") as f:
    input_data = f.read()

  # Calculate the frequency of each symbol in the input data.
  frequencies = {}
  for symbol in input_data:
    if symbol not in frequencies:
      frequencies[symbol] = 0
    frequencies[symbol] += 1

  # Generate a Huffman code for each symbol.
  codes = huffman_coding.huffman_coding(list(frequencies.keys()), list(frequencies.values()))

  # Compress the input data using the Huffman codes.
  compressed_data = []
  for symbol in input_data:
    compressed_data.append(codes[symbol])

  # Write the compressed data to the output file.
  with open(output_file, "wb") as f:
    f.write(bytes("".join(compressed_data), encoding="utf-8"))

def decompress_file(input_file, output_file):
  """Decompresses a file that was compressed using Huffman coding.

  Args:
    input_file: The path to the input file.
    output_file: The path to the output file.
  """

  # Read the compressed data from the input file.
  with open(input_file, "rb") as f:
    compressed_data = f.read().decode("utf-8")

  # Generate a Huffman tree from the compressed data.
  huffman_tree = huffman_coding.HuffmanTree.from_codes(compressed_data)

  # Decompress the compressed data using the Huffman tree.
  decompressed_data = []
  node = huffman_tree.root
  if node is not None:
    for bit in compressed_data:
      if bit == "0":
        node = node.left
      else:
        node = node.right

    if node is not None and node.symbol is not None:
      decompressed_data.append(node.symbol)
      node = huffman_tree.root

  # Write the decompressed data to the output file.
  with open(output_file, "wb") as f:
    f.write(bytes("".join(decompressed_data), encoding="utf-8"))

# Example usage:

input_file = "input.txt"
output_file = "compressed.txt"

compress_file(input_file, output_file)

decompressed_file = "decompressed.txt"

decompress_file(output_file, decompressed_file)