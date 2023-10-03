import random


# This function divides a set of points into d+1 subsets, each with at least one point.
#
# Args:
#   points: A list of points.
#
# Returns:
#   A list of lists, where each inner list is a subset of the points.
def tverberg(points):
  """Divides a set of points into d+1 subsets, each with at least one point.

  Args:
    points: A list of points.

  Returns:
    A list of lists, where each inner list is a subset of the points.
  """

  # Get the dimension of the points.
  d = len(points[0])

  # Initialize a list of subsets.
  subsets = []

  # Create d+1 subsets, each with a single point.
  for i in range(d + 1):
    subset = random.sample(points, 1)
    subsets.append(subset)

  # For each point in the input, add it to the first subset that it is not
  # already in.
  for point in points:
    for subset in subsets:
      if point not in subset:
        subset.append(point)
        break

  # Return the list of subsets.
  return subsets


# This is the main function of the program.
def main():
  """The main function of the program."""

  # Create a list of points.
  points = [(0, 0), (1, 0), (0, 1), (1, 1)]

  # Compute the Tverberg partition of the points.
  subsets = tverberg(points)

  # Print the Tverberg partition to the console.
  print(subsets)


# If this is the main module, call the main function.
if __name__ == "__main__":
  main()