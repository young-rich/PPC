import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1
from wdtype import *
from layers import *


class Tabular(pl.LightningModule):

    def __init__(
        self,
        lr,
        column_idx: Dict[str, int],
        embed_input: List[Tuple[str, int]],
        n_blocks: int = 1,
        input_dim: int = 8,
        n_heads: int = 1,
        keep_attn_weights=True,
        dropout: float = 0.5,
        ff_hidden_dim: int = 8 * 4,
        transformer_activation: str = "gelu",
        mlp_activation: str = "relu",
        mlp_batchnorm: bool = True,
        mlp_batchnorm_last: bool = False,
        mlp_linear_first: bool = True,
    ):
        super().__init__()

        metrics = MetricCollection([
            Precision(multiclass=False),
            Recall(multiclass=False),
            F1(multiclass=False)
        ])
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')

        self.lr = lr
        self.column_idx = column_idx
        self.embed_input = embed_input
        self.input_dim = input_dim
        self.embed_dropout = dropout

        self.categorical_cols = [ei[0] for ei in self.embed_input]
        self.n_tokens = sum([ei[1] for ei in self.embed_input])

        self._set_categ_embeddings()
        mlp_hidden_dims = self._set_mlp_hidden_dims()

        self.logits = nn.LogSoftmax()
        self.liner = nn.Linear(self.mlp_inp_l, 2)

        self.transformer_blks = nn.Sequential()
        for i in range(n_blocks):
            self.transformer_blks.add_module(
                "block" + str(i),
                TransformerEncoder(
                    input_dim,
                    n_heads,
                    keep_attn_weights,
                    ff_hidden_dim,
                    dropout,
                    transformer_activation,
                ),
            )
        self.transformer_mlp = MLP(
            mlp_hidden_dims,
            mlp_activation,
            dropout,
            mlp_batchnorm,
            mlp_batchnorm_last,
            mlp_linear_first,
        )

        self.attention_weights: List[Any] = [None] * n_blocks

    def forward(self, X):
        x = self.cat_embed(X[:, self.cat_idx].long())
        x = self.embedding_dropout(x)

        for i, blk in enumerate(self.transformer_blks):
            x = blk(x)
            self.attention_weights[i] = blk.self_attn.attn_weights
        x = x.flatten(1)
        x = self.transformer_mlp(x)
        x = self.liner(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        output = self.train_metrics(logits, y)
        self.log_dict(output)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("val_loss", loss)
        output = self.valid_metrics(logits, y)
        self.log_dict(output)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def _set_categ_embeddings(self):
        self.cat_idx = [self.column_idx[col] for col in self.categorical_cols]
        self.cat_embed = nn.Embedding(
            self.n_tokens + 1, self.input_dim, padding_idx=0
        )
        self.embedding_dropout = nn.Dropout(self.embed_dropout)

    def _set_mlp_hidden_dims(self) -> List[int]:
        mlp_inp_l = len(self.embed_input) * self.input_dim
        self.mlp_inp_l = mlp_inp_l
        mlp_hidden_dims = [mlp_inp_l, mlp_inp_l]
        return mlp_hidden_dims


class Tabular_text_entity(pl.LightningModule):

    def __init__(
        self,
        lr,
        weight_decay,
        pre_model,
        use_transformer: bool,
        use_res: bool,
        vocab_len: int,
        column_idx: Dict[str, int],
        embed_input: List[Tuple[str, int]],
        input_dim: int = 8,
        n_heads: int = 1,
        keep_attn_weights=True,
        dropout: float = 0.5,
        ff_hidden_dim: int = 8 * 2,
        transformer_activation: str = "gelu",
        mlp_activation: str = "relu",
        mlp_batchnorm: bool = True,
        mlp_batchnorm_last: bool = True,
        mlp_linear_first: bool = True,
    ):
        super().__init__()

        metrics = MetricCollection([
            Precision(multiclass=False),
            Recall(multiclass=False),
            F1(multiclass=False)
        ])
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')

        self.use_transformer = use_transformer
        self.use_res = use_res
        self.pre_model = pre_model
        self.lr = lr
        self.weight_decay = weight_decay
        self.column_idx = column_idx
        self.embed_input = embed_input
        self.input_dim = input_dim
        self.embed_dropout = dropout
        self.vocab_len = vocab_len

        self.categorical_cols = [ei[0] for ei in self.embed_input]
        self.n_tokens = sum([ei[1] for ei in self.embed_input])

        self._set_categ_embeddings()
        mlp_hidden_dims = self._set_mlp_hidden_dims()

        self.transformer_text = TransformerEncoder(
            input_dim,
            n_heads,
            keep_attn_weights,
            ff_hidden_dim,
            dropout,
            transformer_activation,
        )

        self.transformer_tabular = TransformerEncoder(
            input_dim,
            n_heads,
            keep_attn_weights,
            ff_hidden_dim,
            dropout,
            transformer_activation,
        )

        self.transformer_text_tabular = TransformerEncoder(
            input_dim,
            n_heads,
            keep_attn_weights,
            ff_hidden_dim,
            dropout,
            transformer_activation,
        )

        self.logits = nn.LogSoftmax()
        self.liner = nn.Linear(self.mlp_inp_l, 2)
        self.sentence_liner = nn.Linear(768, self.input_dim)

        self.transformer_all = TransformerEncoder(
            input_dim,
            n_heads,
            keep_attn_weights,
            ff_hidden_dim,
            dropout,
            transformer_activation,
        )

        self.transformer_mlp = MLP(
            mlp_hidden_dims,
            mlp_activation,
            dropout,
            mlp_batchnorm,
            mlp_batchnorm_last,
            mlp_linear_first,
        )

        self.entity_embed = nn.Embedding(
            self.vocab_len, self.input_dim, padding_idx=0
        )

    def forward(self, X, x_text, x_entity):
        """
        表格数据的transformer
        """
        x = self.cat_embed(X[:, self.cat_idx].long())
        x = self.embedding_dropout(x)

        if (self.use_transformer):
            x_transformer = self.transformer_tabular(x)
            if (self.use_res):
                x = nn.ReLU()(torch.add(x_transformer, x))
            else:
                x = x_transformer
        """
        文本数据的transformer
        """
        text_emb = self.pre_model(x_text)
        embs = text_emb.hidden_states[-1]
        word_embs = self.sentence_liner(embs)
        x_e_embs = self.entity_embed(x_entity)

        x_text_embs = torch.cat([word_embs, x_e_embs], 1)

        if (self.use_transformer):
            x_text_embs_transformer = self.transformer_text(x_text_embs)
            if (self.use_res):
                x_text_embs = nn.ReLU()(torch.add(x_text_embs, x_text_embs_transformer))
            else:
                x_text_embs = x_text_embs_transformer
        """
        两种数据合并transformer
        """
        x_all = torch.cat([x, x_text_embs], 1)

        if (self.use_transformer):
            x_all_transformer = self.transformer_text_tabular(x_all)
            if (self.use_res):
                x_all = nn.ReLU()(torch.add(x_all, x_all_transformer))
            else:
                x_all = x_all_transformer

        x = torch.cat([x, x_text_embs, x_all], 1)

        # x =  self.transformer_all(x)

        self.attention_weights_tabular = self.transformer_tabular.self_attn.attn_weights
        self.attention_weights_text = self.transformer_text.self_attn.attn_weights
        self.attention_weights_text_tabular = self.transformer_text_tabular.self_attn.attn_weights

        # self.attention_weights_all = self.transformer_all.self_attn.attn_weights

        x = x.flatten(1)
        x = self.transformer_mlp(x)
        x = self.liner(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, text, entity, y = batch
        logits = self(x, text, entity)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        output = self.train_metrics(logits, y)
        self.log_dict(output)
        return loss

    def validation_step(self, batch, batch_idx):
        x, text, entity, y = batch
        logits = self(x, text, entity)
        loss = F.nll_loss(logits, y)
        self.log("val_loss", loss)
        output = self.valid_metrics(logits, y)
        self.log_dict(output)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def _set_categ_embeddings(self):
        self.cat_idx = [self.column_idx[col] for col in self.categorical_cols]
        self.cat_embed = nn.Embedding(
            self.n_tokens + 1, self.input_dim, padding_idx=0
        )
        self.embedding_dropout = nn.Dropout(self.embed_dropout)

    def _set_mlp_hidden_dims(self) -> List[int]:
        mlp_inp_l = (len(self.embed_input) * self.input_dim +
                     (64 + 4) * self.input_dim) * 2
        self.mlp_inp_l = mlp_inp_l
        mlp_hidden_dims = [mlp_inp_l, mlp_inp_l]
        return mlp_hidden_dims