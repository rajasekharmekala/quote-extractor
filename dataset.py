
from datasets import Dataset
from datasets.dataset_dict import DatasetDict
import sqlite3
import numpy as np
import pandas as pd
import ast
import json

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
    print("Total quotes with tags: ", quotes_with_tags)
    return tags

def get_tags(df,min_count=500):
    tags_dict = get_tags_dict(df)
    print("Total Tags: ", len(tags_dict))
    tags = []
    counts = []
    for key,value in tags_dict.items():
        if value >min_count and key != "None":
            tags.append(key)
            counts.append(value)
    print("Used Tags: ", len(tags))
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
    


def prepare_dataframe():
    connection = sqlite3.connect('./data/quotes.sqlite')

    df = pd.read_sql_query(r"SELECT * FROM quotes", connection)
    df.columns= df.columns.str.lower()
    df.rename(columns = {'tags':'label'}, inplace = True)


    tags = get_tags(df, min_count = 1000)

    # Dropping empty quotes
    df = df.drop(df[df["quote"] == ""].index)
    print(tags)
    num_classes = len(tags)+1
    df["label"] = df["label"].apply(lambda x: clean_tags(x, tags, num_classes))

    labels = [x for x in range(len(tags))]
    df = df.drop(df[~df["label"].isin(labels)].index)
    print(df.head(5))
    return df, num_classes-1

def get_dataset():
    df, num_classes = prepare_dataframe()
    print("Dataset shape: ", df.shape)
    dataset = Dataset.from_pandas(df)
    train_testvalid = dataset.train_test_split(test_size=0.3)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
    train_test_valid_dataset = DatasetDict({
        'train': train_testvalid['train'],
        'test': test_valid['test'],
        'valid': test_valid['train'],
        'num_classes': num_classes })
    return train_test_valid_dataset

if __name__ == '__main__':
    print(get_dataset())