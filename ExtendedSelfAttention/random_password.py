import random

def generate_random_password(length=12):
  """Generates a random password of the given length.

  Args:
    length: The length of the password.

  Returns:
    A random password.
  """

  characters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
  password = ''
  for i in range(length):
    password += random.choice(characters)
  return password

# Generate a random password of length 12.
random_password = generate_random_password()

# Print the random password.
print(random_password)