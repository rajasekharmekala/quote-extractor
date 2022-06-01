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

def remove_duplicates(quotes):
    print("initial length: ", len(quotes))
    sorted_quotes = sorted(quotes, key=len)
    duplicate = set()
    unique = set()
    for i, quote in enumerate(sorted_quotes):
        if(len(quote.split(" ")) <5): continue # not removing > 350 characters?
        if quote in duplicate: continue
        for item in sorted_quotes[i+1:]:
            try:
                x = regex.search(r'(%s){e<=%d}'%(quote,3), item,flags=regex.IGNORECASE)
                if x is not None:
                    duplicate.add(item)
            except:
                pass
        unique.add(quote)
    print("final length: ", len(unique))
    return unique

def add_sentence_label_pairs(text, quotes, df, fuzzy_length=3, min_sentence_tokens=15, verb_phrase_dict=None,chapter_name=None):
    # input = "Monalisa was painted by Leonrdo da Vinchi  abcdefghijklmnopqrst"
    found_quotes = set()
    matches = 0
    sentences = 0
    sorted_quotes = sorted(list(quotes), key=len)

    for quote in sorted_quotes:
        try:
            x = regex.search(r'(%s){e<=%d}'%(quote,fuzzy_length), text,flags=regex.IGNORECASE)
            if x is not None:
                found_quotes.add(quote)
                matches+=1
                try:
                    match = x.group(0)
                    words = quote.split(" ")
                    start_string = words[0]
                    match = match[match.index(start_string): ]
                    match = match[: len(quote)]
                    if len(match)>0:
                        start = text.index(match)
                        df.loc[len(df.index)] = [match,1 , chapter_name, (start, start + len(match) )]
                except:
                    match = x.group(0)
                    # print("----------------------------------------")
                    # print("Match: ",match)
                    # print("Quote: ",quote)
                    # print("----------------------------------------")
                text = text.replace(match,"")
        except:
            pass
    for sentence in text.split("."):
        if (len(sentence.split(" ")) < min_sentence_tokens ): continue
        sentences+=1
        df.loc[len(df.index)] = [sentence.strip(),0, chapter_name, None]
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
        
def thread_function(filename, path, stage_1_df, progress_bar, total_matches, total_sentences, total_quotes, _lock):
    from epub_utils import epub2dict
    import time
    stage_2_df = pd.DataFrame({'text':[],'label':[] , 'chapter_name':[], 'pos':[]})
    if not filename.endswith(".epub"):
        return
    sen_in_book = 0
    matches = 0
    file_path = os.path.join(path, filename)
    book_title = os.path.splitext(filename)[0]
    #print(book_title)
    logger.info(book_title.lower())
    #print(stage_1_df["title"].str.lower().tolist())
    quotes_in_book =  stage_1_df[stage_1_df["title"].str.lower() == book_title.lower()]["text"].tolist() # add lower()
    quotes_in_book = remove_duplicates(quotes_in_book)

    quotes_in_cur_book = len(quotes_in_book) +1e-12
    print("waiting for a lock 1")
    while _lock.locked():
        time.sleep(25)
        continue
    _lock.acquire()
    print("Lock acquired 1")
    total_quotes += quotes_in_cur_book
    _lock.release()
    print("Lock released 1")
    print("len", quotes_in_cur_book, total_quotes)
    verb_phrase_dict = get_verbs_in_quotes(quotes_in_book)
    _dict = epub2dict(file_path)
    with open(f"metadata/{book_title}.txt", "w") as f:
        for text in _dict:
            f.write(_dict[text])
            f.write("\n")
        #f.write(json.dumps(_dict))
    remaining_texts = {}
    # for chapter_name in ["OEBPS/part1.xhtml"]:
    for chapter_name in _dict:
        print("chapter: ", book_title, chapter_name)
        quotes_in_book, _matches, sentences, remaining_text = add_sentence_label_pairs(_dict[chapter_name], quotes_in_book, stage_2_df, verb_phrase_dict=verb_phrase_dict, chapter_name=chapter_name)
        remaining_texts[chapter_name] = remaining_text
        matches += _matches
        sen_in_book+= sentences
        # print(matches)

    for chapter_name in remaining_texts:
        print("chapter second level: ", book_title, chapter_name)
        quotes_in_book, _matches, sentences, remaining_text = search_and_add_quotes(remaining_texts[chapter_name], quotes_in_book, stage_2_df, verb_phrase_dict, chapter_name)
        remaining_texts[chapter_name] = remaining_text
        matches += _matches
        sen_in_book+= sentences
        #print(matches)


    with open(f"quotes/{book_title}.txt", "w") as f:
        for q in quotes_in_book:
            f.write(q)
            f.write("\n")
            f.write("-------------------------")
            f.write("\n")
    with open(f"metadata/{book_title}_remaining.txt", "w") as f:
        for text in remaining_texts:
            f.write(remaining_texts[text])
            f.write("\n")
        #f.write(json.dumps(remaining_texts))
    while _lock.locked():
        time.sleep(25)
        continue
    print("waiting for a lock 2")
    _lock.acquire()
    print("Lock acquired 2")
    total_matches += matches
    total_sentences += sen_in_book
    _lock.release()
    print("Lock released 2")
    print("Matches: ", matches)
    progress_bar.set_description(f'Matches: { total_matches/total_quotes:.3f} {book_title} Matches: { matches/quotes_in_cur_book:.3f}')
    logger.info(f"sentences in book {book_title}: {sen_in_book}")
    logger.info(f"matches in book {book_title}: {matches}")
    if matches >0 : stage_2_df.to_pickle(f"./dataframes/stage_2_{book_title}.pkl")
    return stage_2_df

def prepare_stage2_dataframe(path="./data/books/epub/"):
    from dataset import prepare_stage1_dataframe
    import threading

    progress_bar = tqdm(os.listdir(path))
    total_matches = 0
    total_sentences = 0
    total_quotes = 0
    stage_1_df, _ = prepare_stage1_dataframe() # loading df
    _lock = threading.Lock()
    
    for filename in progress_bar:
        threads = list()
        print("Main    : create and start thread %d.", filename)
        x = threading.Thread(target=thread_function, args=(filename,path,stage_1_df,progress_bar,total_matches,total_sentences,total_quotes,_lock))
        threads.append(x)
        x.start()

    for index, thread in enumerate(threads):
        print("Main    : before joining thread %d.", index)
        thread.join()
        print("Main    : thread %d done", index)

    print("total matches", total_matches)
    print("total quotes", total_quotes)
    print("prepare_stage2_dataframe returning")
    return False
    '''
        for filename in progress_bar:
        print("Main    : create and start thread %d.", filename)
        x = threading.Thread(target=thread_function, args=(filename,path,stage_1_df,progress_bar,total_matches,total_sentences,total_quotes,_lock))
        threads.append(x)
        x.start()
        count += 1
        if count%2==0:
            for index, thread in enumerate(threads):
                print("Main    : before joining thread %d.", index)
                thread.join()
                print("Main    : thread %d done", index)
    for index, thread in enumerate(threads):
        print("Main    : before joining thread %d.", index)
        thread.join()
        print("Main    : thread %d done", index)
    '''
    stage_1_df, _ = prepare_stage1_dataframe() # loading df
    for filename in progress_bar:
        stage_2_df = pd.DataFrame({'text':[],'label':[] , 'chapter_name':[], 'pos':[]})
        if not filename.endswith(".epub"):
            continue
        sen_in_book = 0
        matches = 0
        file_path = os.path.join(path, filename)
        book_title = os.path.splitext(filename)[0]
        #print(book_title)
        print(book_title.lower())
        #print(stage_1_df["title"].str.lower().tolist())
        quotes_in_book =  stage_1_df[stage_1_df["title"].str.lower() == book_title.lower()]["text"].tolist() # add lower()
        quotes_in_book = remove_duplicates(quotes_in_book)

        quotes_in_cur_book = len(quotes_in_book) +1e-12
        total_quotes += quotes_in_cur_book
        print("len", quotes_in_cur_book, total_quotes)
        verb_phrase_dict = get_verbs_in_quotes(quotes_in_book)
        _dict = epub2dict(file_path)
        with open(f"metadata/{book_title}.txt", "w") as f:
            for text in _dict:
                f.write(_dict[text])
                f.write("\n")
            #f.write(json.dumps(_dict))
        remaining_texts = {}
        # for chapter_name in ["OEBPS/part1.xhtml"]:
        for chapter_name in _dict:
            print("chapter: ", chapter_name)
            quotes_in_book, _matches, sentences, remaining_text = add_sentence_label_pairs(_dict[chapter_name], quotes_in_book, stage_2_df, verb_phrase_dict=verb_phrase_dict, chapter_name=chapter_name)
            remaining_texts[chapter_name] = remaining_text
            matches += _matches
            sen_in_book+= sentences
            # print(matches)

        for chapter_name in remaining_texts:
            #print("chapter: ", chapter_name)
            quotes_in_book, _matches, sentences, remaining_text = search_and_add_quotes(remaining_texts[chapter_name], quotes_in_book, stage_2_df, verb_phrase_dict, chapter_name)
            remaining_texts[chapter_name] = remaining_text
            matches += _matches
            sen_in_book+= sentences
            #print(matches)


        with open(f"quotes/{book_title}.txt", "w") as f:
            for q in quotes_in_book:
                f.write(q)
                f.write("\n")
                f.write("-------------------------")
                f.write("\n")
        with open(f"metadata/{book_title}_remaining.txt", "w") as f:
            for text in remaining_texts:
                f.write(remaining_texts[text])
                f.write("\n")
            #f.write(json.dumps(remaining_texts))

        total_matches += matches
        total_sentences += sen_in_book
        print("Matches: ", matches)
        progress_bar.set_description(f'Matches: { total_matches/total_quotes:.3f} {book_title} Matches: { matches/quotes_in_cur_book:.3f}')
        logger.info(f"sentences in book {book_title}: {sen_in_book}")
        logger.info(f"matches in book {book_title}: {matches}")
        if matches >0 : stage_2_df.to_pickle(f"./dataframes/stage_2_{book_title}.pkl")
    return stage_2_df

def add_to_qa_df(text, quotes, df, fuzzy_length=3, block_size=500, stride=100,chapter_name=None, book_title = None, tokenizer = None):
    # input = "Monalisa was painted by Leonrdo da Vinchi  abcdefghijklmnopqrst"
    found_quotes = set()
    matches = 0
    sorted_quotes = sorted(list(quotes), key=len)


    if tokenizer is None:
        raise("Tokenizer not found")

    subword_tokens = tokenizer(text)["input_ids"]
    
    prevEnd = 0
    texts = {}
    count = 0
    subword_tokens = subword_tokens[1:-1]
    while prevEnd < len(subword_tokens):
        start = max(0, prevEnd - stride)
        texts[count] = {
            "text": tokenizer.decode(subword_tokens[start:start+ block_size]),
            "startIndices":[],
            "quotes":[] 
        } 
        count+=1
        prevEnd = start+block_size


    for quote in sorted_quotes:
        try:
            x = regex.search(r'(%s){e<=%d}'%(quote,fuzzy_length), text,flags=regex.IGNORECASE)
            if x is not None:
                found_quotes.add(quote)
                matches+=1
                try:
                    match = x.group(0)
                    words = quote.split(" ")
                    match = match[match.index(words[0]): ]
                    match = match[: len(quote)]
                    if len(match)>0:
                        for idx in texts:
                            try:
                                start = texts[idx]["text"].index(match)
                                texts[idx]["startIndices"].append(start)
                                texts[idx]["quotes"].append(match)
                            except:
                                pass
                except:
                    match = x.group(0)
                text = text.replace(match,"")
        except:
            pass
    for idx in texts:
        df.loc[len(df.index)] = [book_title, chapter_name, texts[idx]["text"] ,texts[idx]["startIndices"], texts[idx]["quotes"] ]
    return quotes - found_quotes, matches

        


def prepare_qa_dataframe(path="./data/books/epub/"):

    from epub_utils import epub2dict
    from dataset import prepare_stage1_dataframe

    from transformers import AutoTokenizer

    model_checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    progress_bar = tqdm(os.listdir(path))
    total_matches = 0
    total_quotes = 0
    for filename in progress_bar:
        qa_df = pd.DataFrame({'book_title': [], 'chapter_name':[], 'context':[],'answer_start':[], "text":[]})
        if not filename.endswith(".epub"):
            continue
        matches = 0
        file_path = os.path.join(path, filename)
        book_title = os.path.splitext(filename)[0]
        progress_bar.set_description("Title: "+ book_title)

        stage_1_df, _ = prepare_stage1_dataframe()
        quotes_in_book =  set(stage_1_df[stage_1_df["title"] == book_title]["text"].tolist())

        quotes_in_cur_book = len(quotes_in_book) +1e-12
        total_quotes += quotes_in_cur_book
        print("len", total_quotes)

        _dict = epub2dict(file_path)
        with open(f"metadata/{book_title}.json", "w") as f:
            f.write(json.dumps(_dict))

        for chapter_name in _dict:
            print("chapter: ", chapter_name)
            quotes_in_book, _matches = add_to_qa_df(_dict[chapter_name], quotes_in_book, qa_df, tokenizer=tokenizer, chapter_name=chapter_name, book_title= book_title)
            matches += _matches

        total_matches += matches
        print("Matches: ", matches)
        progress_bar.set_description(f'Matches: { total_matches/total_quotes:.3f} {book_title} Matches: { matches/quotes_in_cur_book:.3f}')
        if matches >0 : qa_df.to_pickle(f"./dataframes_qa/{book_title}.pkl")

logger = get_logger("dataset")


if __name__ == '__main__':
    prepare_qa_dataframe()
