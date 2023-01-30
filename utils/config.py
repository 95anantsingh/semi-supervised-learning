
"""
This file contains logic for configurations.
"""

import json
from typing import List, TypeVar

StrategyValue = TypeVar('StrategyValue',int,float)

class SelfTrainConfig(object):
    """Configuration for training a model."""

    def __init__(self,    
        run_name : str, 
        strategy : str,
        strategy_value : StrategyValue,
        main_dataset : str,
        other_datasets : List[str],
        
        main_sampling_ratio: float = 0.3,
        labeled_size : int = 0,
        labeled_sampling : str = 'equal',
        unlabeled_size : int = 0,
        unlabeled_in_domain_ratio : float = 0.5,
        eval_size : int = 3000,

        num_epochs : int = 4,
        num_iters : int = 20,
        max_batch_size : int = 256,
        min_batch_size : int = 8,
        steps_per_epoch : int = 250,
        min_total_steps : int = 60,
        max_patience : int = 3,
        save_interval : int = 10,
        sorted_data : bool = True,
        history_dir : str = './history',
        data_dir : str = './data',
        cache_dir : str = './.cache',
        resume_from: str = '' 
        ) -> None:

        """
        Create a new Self-Training config.

        :param run_name:
        :param strategy:
        :param strategy_value:
        :param main_dataset:
        :param other_datasets:

        :param main_sampling_ratio:
        :param labeled_size:
        :param labeled_sampling:
        :param unlabeled_size:
        :param unlabeled_in_domain_ratio:
        :param eval_size:

        :param num_epochs:
        :param num_iters:
        :param max_batch_size:
        :param min_batch_size:
        :param steps_per_epoch:
        :param min_total_steps:
        :param max_patience:
        :param save_interval:
        :param sorted_data:
        :param history_dir:
        :param data_dir:
        :param cache_dir:
        :param resume_from:
        """
        self.run_name = run_name
        self.strategy = strategy
        self.strategy_value = strategy_value
        self.main_dataset = main_dataset
        self.other_datasets = other_datasets

        self.main_sampling_ratio = main_sampling_ratio
        self.labeled_size = labeled_size
        self.labeled_sampling = labeled_sampling
        self.unlabeled_size = unlabeled_size
        self.unlabeled_in_domain_ratio = unlabeled_in_domain_ratio
        self.eval_size = eval_size

        self.num_epochs = num_epochs
        self.num_iters = num_iters
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.steps_per_epoch = steps_per_epoch
        self.min_total_steps = min_total_steps
        self.max_patience = max_patience
        self.save_interval = save_interval
        self.sorted_data = sorted_data
        self.history_dir = history_dir
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.resume_from = resume_from

    def __repr__(self):
        return repr(self.as_dict())

    def save_to_json(self, path: str):
        """Save this config to a json file."""
        with open(path, 'w', encoding='utf8') as file:
            json.dump(self.as_dict(), file)
    
    def as_dict(self):
        return self.__dict__

    @classmethod
    def load_from_json(cls, path: str):
        """Load a config from a json file."""
        config = cls.__new__(cls)
        with open(path, 'r', encoding='utf8') as file:
            config.__dict__ = json.load(file)
        return config
    
    @classmethod
    def load_from_dict(cls, dictionary: dict):
        """Load a config from a dictionary."""
        config = cls.__new__(cls)
        config.__dict__ = dictionary
        return config