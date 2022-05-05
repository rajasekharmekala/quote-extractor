class Accuracy:
  def __init__(self):
    self.correct = 0
    self.total = 0
  
  def update(self, correct):
    self.correct += correct.sum().item()
    self.total += correct.shape[0]

  def get(self):
    return self.correct / self.total


def tokenize(string: str):
  """Tokenizes an input string."""
  return string.lower().split()

def tokenize_instance(instance):
  """Simple wrapper that applies the `tokenize` function to an instance."""
  return {'tokens': tokenize(instance['quote'])}

