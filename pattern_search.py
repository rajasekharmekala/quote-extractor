import collections
from vocab import Vocabulary
from nltk import word_tokenize
# from utils import tokenize
import regex
import pandas as pd


def tokenize(string: str):
    """Tokenizes an input string."""
    return string.lower().split()


def search_and_add_quotes(document, quotes, df):

    counter = collections.Counter()

    for quote in quotes:
        counter.update(tokenize(quote))
    
    counter.update(tokenize(document))
    words = list(counter.keys())
    # print("size: ", len(words))
    vocab = Vocabulary(words, add_unk_token=True)
    document_map = u"".join(vocab.map_tokens_to_encodings(tokenize(document)))

    found_quotes = set()
    matches = 0
    sentences = 0


    
    for quote in quotes:
        quote_embeddings = vocab.map_tokens_to_encodings(tokenize(quote))
        quote_map = u"".join(quote_embeddings)
        if len(quote_embeddings) <=10:
            fuzzy_length = 1
        elif len(quote_embeddings) <=20:
            fuzzy_length = 2
        else:
            fuzzy_length = 3
        try:
            x = regex.search(r'(%s){e<=%d}'%(quote_map,fuzzy_length), document_map,flags=regex.UNICODE)
            if x is not None:
                # if()
                matches+=1
                # print(vocab.map_tokens_to_encodings(tokenize(quote)))
                # print(vocab.map_tokens_to_encodings(tokenize(document)))
                # print(x)
                try:
                    match = x.group(0)
                    # print("+++++++++++++++++++++++++++")
                    # print("Quote: ", " ".join(vocab.map_encodings_to_tokens([x for x in quote_map if len(x)>0 ])))
                    # print("Found: ", " ".join(vocab.map_encodings_to_tokens([x for x in match if len(x)>0 ])))

                    # match = match[match.index(quote_map[0]): ]
                    # match = match[: match.rindex(quote_map[-1]) +len(quote_map[-1])]
                    df.loc[len(df.index)] = [quote,1 ]
                    found_quotes.add(quote)
                except Exception as e:
                    print("---------------------------")
                    # match = x.group(0)
                    # print("Quote: ", " ".join(vocab.map_encodings_to_tokens([x for x in quote_map if len(x)>0 ])))
                    # print("Found: ", " ".join(vocab.map_encodings_to_tokens([x for x in match if len(x)>0 ])))
                    # print(x)
                    # # print(match.split(" "))
                    # print("len: ", len(match))
                    # print(e)
                    print("---------------------------")
                    continue

                
                # document_map = document_map.replace(match,"")
        except:
            pass

    remaining_text = ""
    if len(document_map) >0:
        remaining_text = " ".join(vocab.map_encodings_to_tokens([x for x in list(document_map) if len(x)>0 ]))
    for sentence in remaining_text.split("."):
        sentences+=1
        df.loc[len(df.index)] = [sentence.strip(),0]
    sentences+= matches

    # print("found: ", found_quotes)
    return quotes - found_quotes, matches, sentences





if __name__ == '__main__':
    search_and_add_quotes("to him to see if", set(["it's happiness to see you."]),  pd.DataFrame({"text":[], "label":[]}))
