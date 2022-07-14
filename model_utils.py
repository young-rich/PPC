from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging
import pytorch_lightning as pl
import torch
from torchtext.vocab import vocab
import os
import numpy as np
from collections import Counter, OrderedDict
from transformers import AutoTokenizer, AutoModelForPreTraining


def get_entity_id(words, entity_vocab):
    """生成实体的index

    Args:
        words (_type_): 实体的集合
        entity_vocab (_type_): 词表

    Returns:
        _type_: 实体的id
    """
    words_ids = []
    for line in words.values:
        word_ids = []
        for word in line:
            word_ids.append(entity_vocab[word])
        while(len(word_ids) < 4):
            word_ids.append(0)
        word_ids = word_ids[:4]
        words_ids.append(word_ids)
    return words_ids


def get_text_model(text):
    """加载预训练模型

    Args:
        text (_type_): 文本的集合

    Returns:
        _type_: 预训练模型
    """
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")
    model_text = AutoModelForPreTraining.from_pretrained(
        "hfl/chinese-macbert-base", output_hidden_states=True)
    inputs = tokenizer(text.values.tolist(), return_tensors="pt",
                       padding='max_length', truncation=True, max_length=64)
    inputs = inputs['input_ids']
    for param in model_text.parameters():
        param.requires_grad = False
    return model_text, inputs


def get_vocab(df):
    """生成词表

    Args:
        df (_type_): 文本集合

    Returns:
        _type_: 词表对象
    """
    all_words = []
    for i in df.words:
        all_words = np.append(all_words, i)

    unique, counts = np.unique(all_words, return_counts=True)

    vocabs = []

    unique = unique[1100:]
    for i in unique:
        if len(i) > 2:
            vocabs.append(i)
    tokens = vocabs
    v = vocab(OrderedDict([(token, 1) for token in tokens]))
    unk_token = '<unk>'
    if unk_token not in v:
        v.insert_token(unk_token, 0)
    v.set_default_index(v[unk_token])
    return v


def get_pt(model_path):
    """加载训练好的模型

    Args:
        model_path (_type_): 模型路径

    Returns:
        _type_: 加载好的模型
    """
    ckpts = []
    for i in range(1, 6):
        for filepath, dirnames, filenames in os.walk(model_path+str(i)+'_lr=3e-06_b_size=1024_agd=1_dropout=0.5/version_0/checkpoints'):
            for filename in filenames:
                ckpts.append(os.path.join(filepath, filename))
    return ckpts


def get_predict(data_loader, model):
    model.cuda()
    model.eval()
    P = []
    Y = []
    with torch.no_grad():
        for data in data_loader:
            X = data[0].cuda()
            y = data[1].tolist()[0]
            Y.append(y)
            log_p = model(X)
            p = torch.exp(log_p[0][1]).tolist()
            P.append(p)
    return P, Y


class Tabular_dataset(Dataset):
    def __init__(self, X, trg):
        super().__init__()
        self.x = X
        self.label = trg

    def __getitem__(self, index):
        return self.x[index], self.label[index]

    def __len__(self):
        return len(self.x)


class Tabular_text_entity_dataset(Dataset):
    def __init__(self, X, text, entity, trg):
        super().__init__()
        self.x = X
        self.text = text
        self.entity = entity
        self.label = trg

    def __getitem__(self, index):
        return self.x[index], self.text[index], self.entity[index], self.label[index]

    def __len__(self):
        return len(self.x)


def get_Dataloader(x, text, entity, y, train_index, valid_index, b_size, x_test, text_test, words_ids_test, y_test):
    data_train = Tabular_text_entity_dataset(
        x[train_index], text[train_index], entity[train_index], y[train_index])
    data_valid = Tabular_text_entity_dataset(
        x[valid_index], text[valid_index], entity[valid_index], y[valid_index])
    data_test = Tabular_text_entity_dataset(
        x_test, text_test, words_ids_test, y_test)

    data_loader_train = DataLoader(
        data_train, batch_size=b_size, shuffle=False)
    data_loader_valid = DataLoader(
        data_valid, batch_size=b_size, shuffle=False)
    data_loader_test = DataLoader(data_test, batch_size=1, shuffle=False)
    data_loader_valid_test = DataLoader(
        data_valid, batch_size=1, shuffle=False)
    return data_loader_train, data_loader_valid, data_loader_valid_test, data_loader_test


def get_data_loader(x, y, train_index, valid_index, b_size):
    data_train = Tabular_dataset(x[train_index], y[train_index])
    data_valid = Tabular_dataset(x[valid_index], y[valid_index])
    # data_test = Tabular_dataset(x_test, y_test)

    data_loader_train = DataLoader(
        data_train, batch_size=b_size, shuffle=False)
    data_loader_valid = DataLoader(
        data_valid, batch_size=b_size, shuffle=False)
    data_loader_valid_test = DataLoader(
        data_valid, batch_size=1, shuffle=False)
    # data_loader_test = DataLoader(data_test, batch_size=1, shuffle=False)
    return data_loader_train, data_loader_valid, data_loader_valid_test, 


def get_trainer(agd, logger, epoch):
    """生成trainer

    Args:
        agd (_type_): _description_
        logger (_type_): 日志记录对象
        epoch (_type_): 训练轮数

    Returns:
        _type_: trainer
    """

    checkpoint_callback = ModelCheckpoint(
        filename='{epoch}-{val_Precision:.5f}-{val_Recall:.5f}-{val_F1:.5f}',
        auto_insert_metric_name=True,
        # save_last = True
    )
    early_stop_callback = EarlyStopping(monitor="val_loss",
                                        min_delta=0.001,
                                        patience=10,
                                        verbose=False,
                                        mode="max")

    trainer = pl.Trainer(
        accumulate_grad_batches=agd,
        logger=logger,
        max_epochs=epoch,
        gpus=1,
        precision=16,
        callbacks=[checkpoint_callback, StochasticWeightAveraging()],
    )
    return trainer
