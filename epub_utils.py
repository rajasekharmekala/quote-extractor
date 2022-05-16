import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import re
import unidecode

def epub2thtml(epub_path):
    book = epub.read_epub(epub_path)
    chapters = []
    names = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        chapters.append(item.get_content())
        names.append(item.get_name())
    return chapters, names


blacklist = [
	'[document]',
	'noscript',
	'header',
	'html',
	'meta',
	'head', 
	'input',
	'script',
	# there may be more elements you don't want, such as "style", etc.
]


allowlist = ['p']

def chap2text(chap):
    output = ''
    soup = BeautifulSoup(chap, 'html.parser')
    text = soup.find_all(text=True)
    for t in text:
        if t.parent.name not in blacklist :
            t = t.replace("\n", " ").lower()
            t = unidecode.unidecode(re.sub('\s+',' ',t)).strip()
            if len(t)>0:
                output += '{} '.format(t)
    return output

def chap2json(chap):
    list = []
    soup = BeautifulSoup(chap, 'html.parser')
    elements = soup.find_all('p')

    for el in elements:
        text = el.get_text().replace("\n", " ").strip().lower()
        text = unidecode.unidecode(re.sub('\s+',' ',text))
        if text != "":
            list.append({"name": el.name, "text": text })
    # for t in text:
    #     if t.strip() == '' or t.strip() == "\n":
    #         continue
    #     if (t.parent.name not in blacklist) and (t.parent.name in allowlist):
    #         text = t.parent.text.replace("\n", "")
    #         list.append({"name": t.parent.name, "text":text })
    return list

def thtml2dict(thtml, names):
    Output = {}
    for i, _ in enumerate(thtml):
        text =  chap2text(thtml[i])
        Output[names[i]] = text
    return Output

def epub2dict(epub_path):
    chapters, names = epub2thtml(epub_path)
    textdict = thtml2dict(chapters, names)
    return textdict


# if __name__ == '__main__':
#     import os
#     import json
#     _set = set()
#     for file in os.listdir("./data/books/epub/"):
#         print(file)
#         with open("sample.json", "w") as f:
#             f.write(json.dumps(epub2dict("./data/books/epub/" + file)))
#     print(_set)