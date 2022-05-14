import logging
import os
import pandas as pd
from tqdm import tqdm
import json

import logging

class Accuracy:
    def __init__(self):
        self.correct = 0
        self.total = 0

    def update(self, predictions, labels):
        correct = predictions == labels
        self.correct += correct.sum().item()
        self.total += correct.shape[0]

    def get(self):
        return self.correct / self.total


def tokenize(string: str):
    """Tokenizes an input string."""
    return string.lower().split()


def tokenize_instance(instance):
    """Simple wrapper that applies the `tokenize` function to an instance."""
    return {'tokens': tokenize(instance['text'])}

def verboseprint(verbose):
     return print if verbose else lambda *a, **k: None


def get_logger(name, filename=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not os.path.exists("Logs"):
        os.makedirs("Logs")
    fh = logging.FileHandler(f"Logs/{filename if filename else name}.log")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
       "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def extract_lines(book_title):
    from dataset import prepare_stage1_dataframe
    from epub_utils import epub2dict

    filepath = f"./data/books/epub/{book_title}.epub"
    _dict = epub2dict(filepath)
    with open("sample.json", "w") as f:
        f.write(json.dumps(_dict))

    df, _ = prepare_stage1_dataframe()
    res = df[df["title"] == book_title]["text"].tolist()
    # _max = 0
    # for el in res:
    #   _max = max(_max, len(el))
    #   logger.info(len(el))
    # logger.info("MAX: ", _max)

    return res


def get_sentences_from_paragraph(text):
    return text.split(".")


def prepare_stage2_dataframe(path="./data/books/epub/"):

    from epub_utils import epub2dict
    from dataset import prepare_stage1_dataframe
  
    stage_2_df = pd.DataFrame({'text':[],'label':[]})

    progress_bar = tqdm(os.listdir(path))
    total_matches = 0
    total_sentences = 0
    total_quotes = 0
    for filename in progress_bar:
        sen_in_book = 0
        matches = 0
        file_path = os.path.join(path, filename)
        book_title = os.path.splitext(file_path)[0]

        stage_1_df, _ = prepare_stage1_dataframe()
        quotes_in_book =  set(stage_1_df[stage_1_df["title"] == book_title]["text"].tolist())

        _dict = epub2dict(file_path)
        for chapter_name in _dict:
            for paragraph in _dict[chapter_name]:
                sentences = get_sentences_from_paragraph(paragraph['text'])
                sen_in_book += len(sentences)
                for sentence in sentences:
                    label =  1 if sentence in quotes_in_book else 0
                    matches += label
                    stage_2_df.loc[len(stage_2_df.index)] = [sentence,label ]

        total_matches += matches
        total_quotes += len(quotes_in_book)
        total_sentences += sen_in_book
        progress_bar.set_description(f'Matches: { total_matches/total_quotes:.3f} {book_title} Matches: { matches/len(quotes_in_book):.3f}')
        logger.info(f"sentences in book {book_title}: {sen_in_book}")
        logger.info(f"matches in book {book_title}: {matches}")
    
    return stage_2_df


logger = get_logger("dataset")


if __name__ == '__main__':
    prepare_stage2_dataframe()
