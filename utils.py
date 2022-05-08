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

def extract_lines(book_title):
  from dataset import prepare_dataframe
  from epub_utils import epub2dict
  import json

  filepath = f"./data/books/epub/{book_title}.epub"
  _dict = epub2dict(filepath)
  with open("sample.json", "w") as f:
    f.write(json.dumps(_dict))

  df, _  = prepare_dataframe()
  res =  df[df["title"] == book_title]["quote"].tolist()
  # _max = 0
  # for el in res:
  #   _max = max(_max, len(el))
  #   print(len(el))
  # print("MAX: ", _max)

  return res

if __name__ == '__main__':
  print(len(extract_lines("Gone Girl")))
