import sys
import torch
from types import SimpleNamespace
from typing import (
    Any,
    Dict,
    List,
    Match,
    Tuple,
    Union,
    Callable,
    Iterable,
    Iterator,
    Optional,
    Generator,
    Collection,
)
NormLayers = Union[torch.nn.Identity, torch.nn.LayerNorm, torch.nn.BatchNorm1d]