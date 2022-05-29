
from datasets import Dataset
from datasets.dataset_dict import DatasetDict
import sqlite3
import numpy as np
import pandas as pd
import ast
import json
import re
import unidecode
import os

from utils import logger

def get_tags_dict(df):
    tags = {}
    quotes_with_tags = 0
    _dict = {}
    with open("labelmap.json", "r") as f:
        _dict = json.load(f)
    for tag in df["label"]:
        if tag == "None":
            tag_list = []
            tags["None"] = tags.get("None", 0)+1
        else:
            tag_list = ast.literal_eval(tag)
            if len(tag_list) >0:
                quotes_with_tags+=1
        for tag_el in tag_list:
            if tag_el in _dict:
                for x in _dict[tag_el]:
                    tags[x] = tags.get(x, 0)+1
            else:
                # tag_el = tag_el.split("-")[0]
                tags[tag_el] = tags.get(tag_el, 0)+1
    logger.info("Total quotes with tags: " + str(quotes_with_tags))
    return tags

def get_tags(df,min_count=500):
    tags_dict = get_tags_dict(df)
    logger.info("Total Tags: " + str(len(tags_dict)))
    tags = []
    counts = []
    for key,value in tags_dict.items():
        if value >min_count and key != "None":
            tags.append(key)
            counts.append(value)
    logger.info("Used Tags: " + str( len(tags)))
    return tags

def clean_tags(tag, tags, num_classes):
    if tag == "None":
        return num_classes-1

    else:
        tag_list = ast.literal_eval(tag)
    for tag_el in tag_list:
        try:
            return tags.index(tag_el)
        except:
            pass
    return num_classes -1
    
def replace_book_titles(df, replace_book_titles):
    with open(replace_book_titles, 'r') as file:
        for line in file:
            line = line.rstrip()
            df["title"].replace({line.split('->')[0]: line.split('->')[1]}, inplace=True)

def prepare_stage1_dataframe():
    connection = sqlite3.connect('./data/quotes.sqlite')

    df = pd.read_sql_query(r"SELECT * FROM quotes", connection)
    df.columns= df.columns.str.lower()
    df.rename(columns = {'tags':'label', 'quote': 'text'}, inplace = True)

    df["text"] = df["text"].apply(lambda sentence: unidecode.unidecode(sentence.replace("\n  ―", "").replace('“','"').replace('”','"').strip(" ").strip('"').lower()) )
    # df["title"] = df["title"].apply(lambda title: title.replace(":", ""))
    df["title"] = df["title"].apply(lambda title: re.sub(r"[:|']", "", title) )
    replace_book_titles(df, "replace_book_titles.txt")
    #print(df["title"].unique().tolist())

    tags = get_tags(df, min_count = 1000)

    # Dropping empty quotes
    df = df.drop(df[df["text"] == ""].index)
    logger.info(tags)
    num_classes = len(tags)+1
    df["label"] = df["label"].apply(lambda x: clean_tags(x, tags, num_classes))

    labels = [x for x in range(len(tags))]
    # df = df.drop(df[~df["label"].isin(labels)].index)
    logger.info(df.head(5))
    return df, num_classes-1

def get_dataset_stage1_old():
    df, num_classes = prepare_stage1_dataframe()
    logger.info("Dataset shape: " + str(df.shape))
    dataset = Dataset.from_pandas(df)
    train_testvalid = dataset.train_test_split(test_size=0.3)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
    train_test_valid_dataset = DatasetDict({
        'train': train_testvalid['train'],
        'test': test_valid['test'],
        'valid': test_valid['train'],
        'num_classes': num_classes })
    return train_test_valid_dataset

def concat_df_from_folder(folder_path="./dataframes/"):
    df = None
    for file in os.listdir(folder_path):
        print(file)
        filepath = os.path.join(folder_path, file)
        new_frame = pd.read_pickle(filepath)
        if df is None:
            df = new_frame
        else:
            df = pd.concat([df, new_frame], ignore_index = True)
    return df

def get_dataset():
    df = None
    if os.path.exists("dataframes_json/stage1_df.json"):
        df = pd.read_json("dataframes_json/stage1_df.json")
    else:
        df = concat_df_from_folder("./dataframes/")
        df.to_json("dataframes_json/stage1_df.json")
    
    df = df.drop(columns=['pos'])

    num_classes=2
    np.random.seed(10)

    negative_indices = df[df["label"]==0].index
    remove_n = int(0.5 * len(negative_indices))
    drop_indices = np.random.choice(negative_indices, remove_n, replace=False)
    df = df.drop(drop_indices)
    df = df.reset_index()
    df["id"] = df.index

    logger.info("Dataset shape: " + str(df.shape))
    print("Positive: ", len(df[df["label"]==1]), "Negative: ", len(df[df["label"]==0])  )
    dataset = Dataset.from_pandas(df)
    train_testvalid = dataset.train_test_split(test_size=0.3)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
    train_test_valid_dataset = DatasetDict({
        'train': train_testvalid['train'],
        'test': test_valid['test'],
        'valid': test_valid['train'],
        'num_classes': num_classes })
    return train_test_valid_dataset


def prepare_qa_dataset(folder_path= "./dataframes_qa/"):
    df = None
    if os.path.exists("dataframes_json/qa_df.json"):
        df = pd.read_json("dataframes_json/qa_df.json")
    else:
        for file in os.listdir(folder_path):
            filepath = os.path.join(folder_path, file)
            new_frame = pd.read_pickle(filepath)
            if df is None or len(df)<len(new_frame):
                df = new_frame
        df.to_json("dataframes_json/qa_df.json")
    
    df["question"] = ""
    df["id"] = df.index
    df["answers"] = df.apply( lambda x: {"answer_start": x[3], "text": x[4]}, axis=1)

    dataset = Dataset.from_pandas(df)
    train_testvalid = dataset.train_test_split(test_size=0.2)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
    train_test_valid_dataset = DatasetDict({
        'train': train_testvalid['train'],
        'test': test_valid['test'],
        'valid': test_valid['train']
        })
    return train_test_valid_dataset

if __name__ == '__main__':
    print(get_dataset())