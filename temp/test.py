# import os
# import torch
# from typing import NamedTuple, List

# class SelfTrainConfig(NamedTuple):
#     strategy : str
#     train_data_path : str
#     eval_data_path : str
#     labeled_classes : List[int]
#     unlabeled_classes : List[int]
#     eval_size : int
#     history_dir : str

#     labeled_size : int = 0
#     unlabeled_size : int = 0
#     labeled_per_class: int = 0
#     unlabeled_per_class: int = 0

#     threshold : float = 0
#     top_k : int = 0
#     pc_top_k : int = 0
#     num_epochs : int = 4
#     batch_size : int = 256
#     min_batch_size : int = 8
#     steps_per_epoch : int = 250
#     min_total_steps : int = 60
#     max_patience : int = 3
#     equal_split : bool = True
#     sorted_data : bool = True
#     save_history : bool = True
#     cache_dir : str = './.cache'
    

# history = torch.load('history/yahoo_answers/random_top_k/250_5_25_5_5000_history.pt')


import os
from tqdm import tqdm
from time import sleep

print('\n\n\n\n\n')

train_pbar = tqdm(leave=False, desc='Training   ', unit=' batch', total= 10, colour='#7986CB' )# Green
eval_pbar  = tqdm(leave=False, desc='Validation ', unit=' batch', total= 10, colour='#3F51B5'   )# Orange
epoch_pbar = tqdm(leave=False, desc='Epoch      ', unit=' epoch', total= 10, colour='#388E3C' )
test_pbar  = tqdm(leave=False, desc='Testing    ', unit=' batch', total= 10, colour='#E0E0E0'   )
pred_pbar  = tqdm(leave=False, desc='Prediction ', unit=' batch', total= 10, colour='#42A5F5' )
iter_pbar  = tqdm(leave=False, desc='Iteration  ', unit=' batch', total= 10, colour='#EF5350' )

sleep(0.1)
train_pbar.update(10)
sleep(0.1)
eval_pbar.update(10)
sleep(0.1)
epoch_pbar.update(10)
sleep(0.1)
test_pbar.update(10)
sleep(0.1)
pred_pbar.update(10)
sleep(0.1)
iter_pbar.update(10)
sleep(0.1)



print()