from pytorch_lightning import seed_everything
import numpy as np
import torch

import pandas as pd
from processing_utils import *
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_lightning.loggers import TensorBoardLogger
from models import *
from model_utils import *
"""
固定随机种子，保证实验结果的一致性
"""
seed = 625
seed_everything(seed, workers=True)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)

df = pd.read_pickle('/home/yx/肺部并发症预测/Data/model_data.pkl')  # 读取数据

entity_vocab = get_vocab(df)  # 获取术前诊断的词表

words = df.pop('words')
text = df.pop('术前诊断').fillna("无")
y = df.pop('肺部并发症').values

pre_model, text = get_text_model(text)  # 获取预训练模型

words_ids = get_entity_id(words, entity_vocab)  # 获取实体id

cat, cont = cat_cont_split(df)  # 将表格数据划分为离散型与连续型
df = remove_outliers(df, cont)  # 连续型数据异常值处理

"""
离散化连续型数据
"""
method = "cart"
for num in range(len(cont)):
    dtype = "numerical"
    binning(df, cont, num, method, y, dtype)

tab_preprocessor = TabPreprocessor(embed_cols=df.columns,
                                   for_transformer=True
                                   )
X_tab = tab_preprocessor.fit_transform(df)
"""
按照时间节点划分训练集和测试集
"""
X_tab_train, X_tab_test, y_train_valid, y_test, text_train, text_test, words_ids_train, words_ids_test = time_split(
    X_tab, text, words_ids, y, 13904)


"""
参数设置
"""
b_size = 1024
lr = 3e-6
epoch = 1000
agd = 1
dropout = 0.5

kf = KFold(n_splits=5, shuffle=True, random_state=625)
results = []
n = 0
data_loaders = []

for train_index, valid_index in kf.split(X_tab_train):

    n += 1

    y_train_valid[train_index]
    data_loader_train, data_loader_valid, data_loader_valid_test, data_loader_test = get_Dataloader(  # 数据加载
        X_tab_train, text_train, np.array(
            words_ids_train), y_train_valid, train_index, valid_index, b_size,
        X_tab_test, text_test, words_ids_test, y_test
    )

    data_loaders.append(data_loader_valid_test)
    for lr in [3e-4, 3e-5, 3e-6]:
        for dropout in [0.9]:
            pt_path = "model_checkpoint/final"
            pt_name = str(n)+"_lr="+str(lr)+"_b_size="+str(b_size) + \
                "_agd="+str(agd)+"_dropout="+str(dropout)  # 模型保存路径
            logger = TensorBoardLogger(pt_path, name=pt_name)  # tensorboard设置
            model = Tabular_text_entity(use_res=True, use_transformer=True, vocab_len=entity_vocab.__len__(
            ), pre_model=pre_model, dropout=dropout, lr=lr, column_idx=tab_preprocessor.column_idx, embed_input=tab_preprocessor.embeddings_input)  # 模型加载
            trainer = get_trainer(agd, logger, epoch)
            trainer.fit(model, data_loader_train, data_loader_valid)  # 训练
            torch.cuda.empty_cache()
    break
