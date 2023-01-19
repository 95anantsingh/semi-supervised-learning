import os
import torch
from typing import NamedTuple, List

class SelfTrainConfig(NamedTuple):
    strategy : str
    train_data_path : str
    eval_data_path : str
    labeled_classes : List[int]
    unlabeled_classes : List[int]
    eval_size : int
    history_dir : str

    labeled_size : int = 0
    unlabeled_size : int = 0
    labeled_per_class: int = 0
    unlabeled_per_class: int = 0

    threshold : float = 0
    top_k : int = 0
    pc_top_k : int = 0
    num_epochs : int = 4
    batch_size : int = 256
    min_batch_size : int = 8
    steps_per_epoch : int = 250
    min_total_steps : int = 60
    max_patience : int = 3
    equal_split : bool = True
    sorted_data : bool = True
    save_history : bool = True
    cache_dir : str = './.cache'
    


history = torch.load('history/yahoo_answers_10/random_top_k/250_5_25_5_5000_history.pt')


print()