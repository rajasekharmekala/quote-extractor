import logging
import os
import pandas as pd
from tqdm import tqdm
import json
import spacy
import textacy

import logging
import regex
from pattern_search import search_and_add_quotes, search_and_add_quotes2
class Accuracy:
    def __init__(self):
        self.correct = 0
        self.total = 1e-12

    def update(self, predictions, labels):
        correct = predictions == labels
        self.correct += correct.sum().item()
        self.total += correct.shape[0]

    def get(self):
        return self.correct / self.total

class Recall:
    def __init__(self):
        self.correct = 0
        self.total = 1e-12

    def update(self, predictions, labels):
        correct = predictions[labels==1]
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
    # logger.addHandler(ch)
    return logger


def extract_lines(book_title):
    from dataset import prepare_stage1_dataframe
    from epub_utils import epub2dict

    filepath = f"./data/books/epub/{book_title}.epub"
    _dict = epub2dict(filepath)

    df, _ = prepare_stage1_dataframe()
    res = df[df["title"] == book_title]["text"].tolist()
    # _max = 0
    # for el in res:
    #   _max = max(_max, len(el))
    #   logger.info(len(el))
    # logger.info("MAX: ", _max)

    return res


def get_sentences_from_paragraph(text):
    return [ sentence.replace('“','"').replace('”','"').strip().strip('"')  for sentence in text.split(".")]


def add_sentence_label_pairs(text, quotes, df, fuzzy_length=3, min_sentence_tokens=15, verb_phrase_dict=None):
    # input = "Monalisa was painted by Leonrdo da Vinchi  abcdefghijklmnopqrst"
    found_quotes = set()
    matches = 0
    sentences = 0

    for quote in quotes:
        try:
            x = regex.search(r'(%s){e<=%d}'%(quote,fuzzy_length), text,flags=regex.IGNORECASE)
            if x is not None:
                found_quotes.add(quote)
                matches+=1
                try:
                    match = x.group(0)
                    words = quote.split(" ")
                    match = match[match.index(words[0]): ]
                    match = match[: match.rindex(words[-1]) +len(words[-1])]
                    df.loc[len(df.index)] = [match,1 ]
                except:
                    match = x.group(0)
                text = text.replace(match,"")
        except:
            pass
    for sentence in text.split("."):
        if (len(sentence.split(" ")) < min_sentence_tokens ): continue
        sentences+=1
        df.loc[len(df.index)] = [sentence.strip(),0]
    sentences+= matches
    return quotes - found_quotes, matches, sentences, text

def get_verbs_in_quotes(quotes):
    _dt ={}
    nlp = spacy.load("en_core_web_sm")
    patterns = [{"POS": {"IN": ["ADJ", "VERB"]} }]
    for quote in quotes:
        doc = nlp(quote)
        verb_phrases = textacy.extract.matches.token_matches(doc, patterns= patterns)
        _dt[quote] = [x.text for x in verb_phrases]
    return _dt
        


def prepare_stage2_dataframe(path="./data/books/epub/"):

    from epub_utils import epub2dict
    from dataset import prepare_stage1_dataframe
  
    stage_2_df = pd.DataFrame({'text':[],'label':[]})

    progress_bar = tqdm(os.listdir(path))
    total_matches = 0
    total_sentences = 0
    total_quotes = 0
    for filename in progress_bar:
        if not filename.endswith(".epub"):
            continue
        sen_in_book = 0
        matches = 0
        file_path = os.path.join(path, filename)
        book_title = os.path.splitext(filename)[0]
        print(book_title)
        stage_1_df, _ = prepare_stage1_dataframe()
        quotes_in_book =  set(stage_1_df[stage_1_df["title"] == book_title]["text"].tolist())

        quotes_in_cur_book = len(quotes_in_book) +1e-12
        total_quotes += quotes_in_cur_book
        print("len", total_quotes)
        verb_phrase_dict = get_verbs_in_quotes(quotes_in_book)
        _dict = epub2dict(file_path)
        with open(f"metadata/{book_title}.json", "w") as f:
            f.write(json.dumps(_dict))
        remaining_texts = {}
        # for chapter_name in ["OEBPS/part1.xhtml"]:
        for chapter_name in _dict:
            print("chapter: ", chapter_name)
            # quotes_in_book, _matches, sentences = add_sentence_label_pairs(_dict[chapter_name], quotes_in_book, stage_2_df)
            quotes_in_book, _matches, sentences, remaining_text = add_sentence_label_pairs(_dict[chapter_name], quotes_in_book, stage_2_df, verb_phrase_dict=verb_phrase_dict)
            # quotes_in_book, _matches, sentences, remaining_text = search_and_add_quotes2(remaining_text, quotes_in_book, stage_2_df, verb_phrase_dict)
            remaining_texts[chapter_name] = remaining_text
            matches += _matches
            sen_in_book+= sentences
            # print(matches)

        # for chapter_name in remaining_texts:
        #     print("chapter: ", chapter_name)
        #     # quotes_in_book, _matches, sentences = add_sentence_label_pairs(_dict[chapter_name], quotes_in_book, stage_2_df)
        #     quotes_in_book, _matches, sentences, remaining_text = search_and_add_quotes2(remaining_texts[chapter_name], quotes_in_book, stage_2_df, verb_phrase_dict)
        #     remaining_texts[chapter_name] = remaining_text
        #     matches += _matches
        #     sen_in_book+= sentences
        #     print(matches)


        with open(f"quotes/{book_title}.txt", "w") as f:
            for q in quotes_in_book:
                f.write(q)
                f.write("\n")
                f.write("-------------------------")
                f.write("\n")
        with open(f"metadata/{book_title}_remaining.json", "w") as f:
            f.write(json.dumps(remaining_texts))

        total_matches += matches
        total_sentences += sen_in_book
        print("Matches: ", matches)
        progress_bar.set_description(f'Matches: { total_matches/total_quotes:.3f} {book_title} Matches: { matches/quotes_in_cur_book:.3f}')
        logger.info(f"sentences in book {book_title}: {sen_in_book}")
        logger.info(f"matches in book {book_title}: {matches}")
        if matches >0 : stage_2_df.to_pickle(f"./dataframes/stage_2_{book_title}.pkl")
    return stage_2_df


logger = get_logger("dataset")


if __name__ == '__main__':
    stage_2_df = prepare_stage2_dataframe()
    stage_2_df.to_pickle("./stage_2_df.pkl")
