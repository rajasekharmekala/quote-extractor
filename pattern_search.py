from simhash import near_match_tokens
from vocab import Vocabulary
from nltk import word_tokenize
# from utils import tokenize
import regex
import pandas as pd


def tokenize(string: str):
    """Tokenizes an input string."""
    return string.lower().split()


# def search_and_add_quotes(document, quotes, df):

#     counter = collections.Counter()

#     for quote in quotes:
#         counter.update(tokenize(quote))
    
#     counter.update(tokenize(document))
#     words = list(counter.keys())
#     # print("size: ", len(words))
#     vocab = Vocabulary(words, add_unk_token=True)
#     document_map = u"".join(vocab.map_tokens_to_encodings(tokenize(document)))

#     found_quotes = set()
#     matches = 0
#     sentences = 0


    
#     for quote in quotes:
#         quote_embeddings = vocab.map_tokens_to_encodings(tokenize(quote))
#         quote_map = u"".join(quote_embeddings)
#         if len(quote_embeddings) <=10:
#             fuzzy_length = 1
#         elif len(quote_embeddings) <=20:
#             fuzzy_length = 2
#         else:
#             fuzzy_length = 3
#         try:
#             x = regex.search(r'(%s){e<=%d}'%(quote_map,fuzzy_length), document_map,flags=regex.UNICODE)
#             if x is not None:
#                 # if()
#                 matches+=1
#                 # print(vocab.map_tokens_to_encodings(tokenize(quote)))
#                 # print(vocab.map_tokens_to_encodings(tokenize(document)))
#                 # print(x)
#                 try:
#                     match = x.group(0)
#                     # print("+++++++++++++++++++++++++++")
#                     # print("Quote: ", " ".join(vocab.map_encodings_to_tokens([x for x in quote_map if len(x)>0 ])))
#                     # print("Found: ", " ".join(vocab.map_encodings_to_tokens([x for x in match if len(x)>0 ])))

#                     # match = match[match.index(quote_map[0]): ]
#                     # match = match[: match.rindex(quote_map[-1]) +len(quote_map[-1])]
#                     df.loc[len(df.index)] = [quote,1 ]
#                     found_quotes.add(quote)
#                 except Exception as e:
#                     print("---------------------------")
#                     # match = x.group(0)
#                     # print("Quote: ", " ".join(vocab.map_encodings_to_tokens([x for x in quote_map if len(x)>0 ])))
#                     # print("Found: ", " ".join(vocab.map_encodings_to_tokens([x for x in match if len(x)>0 ])))
#                     # print(x)
#                     # # print(match.split(" "))
#                     # print("len: ", len(match))
#                     # print(e)
#                     print("---------------------------")
#                     continue

                
#                 # document_map = document_map.replace(match,"")
#         except:
#             pass

#     remaining_text = ""
#     if len(document_map) >0:
#         remaining_text = " ".join(vocab.map_encodings_to_tokens([x for x in list(document_map) if len(x)>0 ]))
#     for sentence in remaining_text.split("."):
#         sentences+=1
#         df.loc[len(df.index)] = [sentence.strip(),0]
#     sentences+= matches

#     # print("found: ", found_quotes)
#     return quotes - found_quotes, matches, sentences


def search_and_add_quotes(document, quotes, df, verb_phrase_dict, chapter_name, min_sentence_tokens = 5):

    found_quotes = set()
    matches = 0
    sentences = 0

    prev_start_lookup = {}
    prev_end_lookup = {}

    
    for idx, quote in enumerate(quotes):
        tokens = quote.split(" ")
        quote_start_token = tokens[0]
        quote_end_token = tokens[-1]

        if len(tokens) <=10:
            fuzzy_length = 2
        elif len(tokens) <=20:
            fuzzy_length = 4
        else:
            fuzzy_length = 6
        if '(' in quote_start_token:
            quote_start_token = quote_start_token.split('(')[1]
        if ')' in quote_start_token:
            quote_start_token = quote_start_token.split(')')[0]
        if ')' in quote_end_token:
            quote_end_token = quote_end_token.split(')')[0]
        if '(' in quote_end_token:
            quote_end_token = quote_end_token.split('(')[1]
        try:
            startIndices = [m.start() for m in regex.finditer(quote_start_token, document)] if quote_start_token not in prev_start_lookup else prev_start_lookup[quote_start_token]
            endIndices = [m.end() for m in regex.finditer(quote_end_token, document)] if quote_end_token not in prev_end_lookup else prev_end_lookup[quote_end_token]
        except:
            #print("error:", quote, quote_start_token, quote_end_token)
            continue
        for startIndex in startIndices:
            for endIndex in endIndices:
                if(endIndex < startIndex) : continue
                match_str = document[startIndex: endIndex]
                match_str_tokens = match_str.split(" ")
                if abs( len(tokens) - len(match_str_tokens) ) > fuzzy_length: continue
                flag  = False
                for verb_phrase in verb_phrase_dict[quote]:
                    if verb_phrase not in match_str:
                        flag = True
                        break
                if flag: continue

                if near_match_tokens(tokens, match_str_tokens, threshold=0.9):
                    matches+=1
                    #print("quote: ", quote)
                    #print("match: ", match_str)
                    df.loc[len(df.index)] = [quote,1,chapter_name,(startIndex, startIndex + len(quote) )] # change pos
                    found_quotes.add(quote)

                    document = document.replace(match_str,"")

    for sentence in document.split("."):
        if (len(sentence.split(" ")) < min_sentence_tokens ): continue
        sentences+=1
        df.loc[len(df.index)] = [sentence.strip(),0,chapter_name,None]
    sentences+= matches

    #print("found(second-level): ", found_quotes)
    return quotes - found_quotes, matches, sentences, document


def get_match_string(document, pattern):
        text = document

        match_str = None
        while True:
            indices = [ (m.start(), m.end()) for m in regex.finditer( pattern, text) ] 
            if(len(indices) == 0):
                break
            if len(indices) >1:
                print("indices: ", len(indices))
            match_str = text[indices[0][0]: indices[0][1]]
            text = match_str[1:]
        return match_str

def search_and_add_quotes2(document, quotes, df, verb_phrase_dict, min_sentence_tokens = 5):

    found_quotes = set()
    matches = 0
    sentences = 0
    
    for idx, quote in enumerate(quotes):
        tokens = quote.strip(".").split(" ")
        quote_start_token = tokens[0]
        quote_end_token = tokens[-1]

        if len(tokens) <=10:
            fuzzy_length = 2
        elif len(tokens) <=20:
            fuzzy_length = 4
        else:
            fuzzy_length = 6
        
        # print(idx)

        if len(verb_phrase_dict[quote])>0 and verb_phrase_dict[quote][0] == quote_start_token:
            verb_phrase_dict[quote] = verb_phrase_dict[quote][1:]
        if len(verb_phrase_dict[quote])>0 and verb_phrase_dict[quote][-1] == quote_end_token:
            verb_phrase_dict[quote] = verb_phrase_dict[quote][:-1]


        pattern =  ".*?[ |-]".join(verb_phrase_dict[quote] + [quote_end_token])

        pattern = f"{quote_start_token}(.(?![ |-]{quote_start_token}))*?[ |-]"+pattern

        # pattern = quote_start_token + "" +"(?!("+  "the" +  ")+)"
        #  + pattern  + ".*[ |-]".join( verb_phrase_dict[quote] + [quote_end_token]) 

        # print(pattern)

        indices = [ (m.start(), m.end()) for m in regex.finditer( pattern, document, overlapped=True) ]

        match_str = None
        min = None
        for startIndex, endIndex in indices:
            if(min == None or (min > (endIndex-startIndex)) ):
                match_str = document[startIndex: endIndex]

        if match_str is not None:
            match_str_tokens = match_str.split(" ")
            if abs( len(tokens) - len(match_str_tokens) ) > fuzzy_length: continue

            if near_match_tokens(tokens, match_str_tokens, threshold=0.9):
                matches+=1
                # print("matches: ", matches)
                # print("quote: ", quote)
                # print("match: ", match_str)
                df.loc[len(df.index)] = [quote,1 ]
                found_quotes.add(quote)

                document = document.replace(match_str,"")

    for sentence in document.split("."):
        if (len(sentence.split(" ")) < min_sentence_tokens ): continue
        sentences+=1
        df.loc[len(df.index)] = [sentence.strip(),0]
    sentences+= matches

    # print("found: ", found_quotes)
    return quotes - found_quotes, matches, sentences, document





if __name__ == '__main__':
    search_and_add_quotes("to him to see if", set(["it's happiness to see you."]),  pd.DataFrame({"text":[], "label":[]}))
