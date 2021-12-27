"""Demo of transformer."""
import os
import sys
import json
from abc import ABC
from typing import List

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, AutoTokenizer, BertTokenizer
import pandas as pd


class ItemDescDataset(Dataset, ABC):
    """
    Dataset for Item Descriptions.
    """
    def __init__(self, file_path: str):
        self.item_desc_list = fetch_cat_desc(file_path)

    def __getitem__(self, item):
        return self.item_desc_list[item]

    def __len__(self):
        return len(self.item_desc_list)


def gen_sample_file(from_file: str, to_file: str, n: int) -> None:
    """
    Generate a smaller file containing sampled rows.
    :param from_file: path of original large file.
    :param to_file: path of generated smaller file.
    :param n: sample number.
    :return: void.
    """
    df = pd.read_csv(from_file)
    df.sample(n).to_csv(to_file)


def fetch_cat_desc(file_path: str) -> List[str]:
    """
    Concatenates item name, brand-name and description, returns a list of result.
    :param file_path: str
    :return: List of descriptions
    """
    with open(file_path, "r", encoding='utf-8') as f:
        # skip header
        f.readline()
        line = f.readline()
        res = []
        while len(line) > 0:
            splits = line.split(',')
            # join `name` `brand_name` `description`
            res.append('. '.join([splits[1], splits[4], splits[7]]))
            line = f.readline()  # move to next line
    return res


def text_to_vec(texts: List[str],
                model: BertModel,
                tokenizer: BertTokenizer) -> List[List[float]]:
    """
    Uses transformer to turn a list of text in to a list of numeric vectors.
    :param texts: a list of texts.
    :param model: model to use.
    :param tokenizer: tokenizer to use.
    :return: a list of embedding vectors.
    """
    # TODO(Oliver): Utilize torch data loader.
    vecs = []
    for i in tqdm(range(1, len(texts))):
         inputs = tokenizer(texts[i], return_tensors="pt")
         vecs += model(**inputs).pooler_output.tolist()
    return vecs


def text_to_vec_batch(dataloader: DataLoader,
                      model: BertModel,
                      tokenizer: BertTokenizer) -> List[List[float]]:
    """
    Uses transformer to turn a list of text in to a list of numeric vectors.
    :param dataloader: torch Dataloader to load data.
    :param model: model to use.
    :param tokenizer: tokenizer to use.
    :return: a list of embedding vectors.
    """
    # TODO(Oliver): Utilize torch data loader.
    vecs = []
    for batch_data in tqdm(dataloader):
        inputs = tokenizer(batch_data,
                           padding=True,
                           truncation=True,
                           return_tensors='pt')
        out = model(**inputs).pooler_output.tolist()
        vecs += out
    return vecs


if __name__ == '__main__':
    # initialize tokenizer and model
    tokenizer1 = AutoTokenizer.from_pretrained('bert-base-uncased')

    # set up dataset and data loader
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    train_dataset = ItemDescDataset('data/train_small.csv')
    test_dataset = ItemDescDataset('data/test.csv')
    train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=16, num_workers=4)

    # get embedding
    # descriptions = fetch_cat_desc("data/test.csv")
    embeddings = text_to_vec_batch(test_dataloader, model=bert_model, tokenizer=tokenizer1)
    with open("data/test_embeddings_new.json", "w") as f:
        json.dump(embeddings, f)

    print(len(embeddings))
    print(embeddings[0])
