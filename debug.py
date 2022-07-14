from pytorch_lightning  import seed_everything
import numpy as np
import torch

import pandas as pd
from processing_utils import *
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_lightning.loggers import TensorBoardLogger
from models import *
from model_utils import *

seed  = 625
seed_everything(seed, workers=True)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)

df = pd.read_pickle('/home/yx/肺部并发症预测/Data/model_data.pkl')
entity_vocab = get_vocab(df)

words = df.pop('words')
text =  df.pop('术前诊断').fillna("无")
y = df.pop('肺部并发症').values

pre_model, text = get_text_model(text)

words_ids = get_entity_id(words, entity_vocab)

cat, cont = cat_cont_split(df) 
df = remove_outliers(df, cont) 

method = "cart"
for num in range(len(cont)):
    dtype="numerical"
    binning(df, cont, num, method, y, dtype)

tab_preprocessor = TabPreprocessor(embed_cols=df.columns,  
                                    for_transformer=True
                                )
X_tab = tab_preprocessor.fit_transform(df)

X_tab_train, X_tab_test, y_train_valid, y_test, text_train, text_test, words_ids_train, words_ids_test = time_split(X_tab, text, words_ids, y, 13904)

b_size = 2
lr = 3e-6
epoch = 1000
agd = 1
dropout = 0.5

kf = KFold(n_splits=5, shuffle = True, random_state = 625)
results = []
n = 0
data_loaders = []
for train_index, valid_index in kf.split(X_tab_train):
    
    n += 1

    y_train_valid[train_index]
    data_loader_train, data_loader_valid, data_loader_valid_test, data_loader_test = get_Dataloader(
        X_tab_train, text_train, np.array(words_ids_train), y_train_valid, train_index, valid_index, b_size, 
        X_tab_test, text_test, words_ids_test, y_test
    )
    data_loaders.append(data_loader_valid_test)
    pt_path = "model_checkpoint/test"
    pt_name = str(n)+"_lr="+str(lr)+"_b_size="+str(b_size)+"_agd="+str(agd)+"_dropout="+str(dropout)
    logger = TensorBoardLogger(pt_path, name = pt_name)
    model = Tabular_text_entity(vocab_len = entity_vocab.__len__(), pre_model = pre_model, dropout = dropout, lr = lr,column_idx=tab_preprocessor.column_idx ,embed_input=tab_preprocessor.embeddings_input)
    trainer = get_trainer(agd, logger, epoch)
    trainer.fit(model, data_loader_train, data_loader_valid)
    break