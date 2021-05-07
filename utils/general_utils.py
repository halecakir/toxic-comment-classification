import functools
import math
import os

import numpy as np
import pandas as pd
import torch
from gensim.models import KeyedVectors, fasttext


def pad_sents(sents, pad_id):
    sents_padded = []
    max_length = functools.reduce(
        lambda acc, curr: len(curr) if len(curr) > acc else acc, sents, 0
    )
    sents_padded = [sent + (max_length - len(sent)) * [pad_id] for sent in sents]
    return sents_padded


class Vocab(object):
    def __init__(self, device):
        self.device = device

        self.w2i = {}
        self.w2i["<pad>"] = 0  # TODO: DIN
        self.w2i["<unk>"] = 1  # TODO: DIN
        self.pad_id = self.w2i["<pad>"]  # TODO: DIN
        self.unk_id = self.w2i["<unk>"]  # TODO: DIN

    def words2indices(self, data):
        for sent_tokens in data:
            for token in sent_tokens:
                if token not in self.w2i:
                    self.w2i[token] = len(self.w2i)

    def to_input_tensor(self, sents):
        indicies = []
        for sent in sents:
            if len(sents) == 0:
                indicies.append([self.pad_id])
            else:
                indicies.append([self.w2i.get(w, self.unk_id) for w in sent])

        sents_padded = pad_sents(indicies, self.pad_id)
        sents_var = torch.tensor(sents_padded, dtype=torch.long).to(device=self.device)
        return torch.t(sents_var)

    def to_input_lengths(self, sents):
        lengths = []
        for sent in sents:
            input_len = 1 if len(sent) == 0 else len(sent)
            lengths.append(input_len)
        return lengths


def load_embeddings_file(file_name, lower=False):
    prefix = os.path.basename(file_name).split(".")[0]

    assert prefix in ["google", "fasttext", "glove"], f"Unknown vector type {prefix}"

    if prefix == "google":
        model = KeyedVectors.load_word2vec_format(
            file_name, binary=True, unicode_errors="ignore"
        )
        words = model.index_to_key
    elif prefix == "glove":
        model = {}
        with open(file_name) as target:
            for line in target:
                fields = line.strip().split()
                vec = [float(x) for x in fields[-300:]]
                word = fields[:-300]

                word = "".join(word)

                if word not in model:
                    model[word] = vec
        words = model.keys()
    elif prefix == "fasttext":
        model = fasttext.load_facebook_vectors(file_name)
        words = model.index_to_key

    if lower:
        vectors = {word.lower(): model[word] for word in words}
    else:
        vectors = {word: model[word] for word in words}

    return vectors, len(vectors[list(words)[0]])


def df_to_X_y(df, limit):
    X, y = [], []
    # TODO : remove limit in prod
    if limit != -1:
        df = df.head(limit)

    for index, row in df.iterrows():
        X.append(row["comment_text"])
        y.append(
            [
                row["toxic"],
                row["severe_toxic"],
                row["obscene"],
                row["threat"],
                row["insult"],
                row["identity_hate"],
            ]
        )

    return X, y


def read_train_corpus(file_path, limit=-1):
    df = pd.read_csv(file_path)
    X, y = df_to_X_y(df, limit)

    return X, y


def read_test_corpus(sent_file_path, label_file_path, limit=-1):
    df_sents = pd.read_csv(sent_file_path)
    df_labels = pd.read_csv(label_file_path)
    df_labels = df_labels[df_labels["toxic"] != -1]
    df = pd.merge(df_labels, df_sents, on=["id"])

    X, y = df_to_X_y(df, limit)

    return X, y


def get_minibatches(data, batch_size, shuffle=False):
    sents, labels = data
    batch_num = math.ceil(len(sents) / batch_size)
    index_array = list(range(len(sents)))
    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size : (i + 1) * batch_size]

        sents_batch = [sents[idx] for idx in indices]
        labels_batch = [labels[idx] for idx in indices]

        composite = [[s, l] for s, l in zip(sents_batch, labels_batch)]
        composite = sorted(composite, key=lambda elem: len(elem[0]), reverse=True)

        sents_batch, labels_batch = map(list, zip(*composite))

        yield sents_batch, labels_batch
