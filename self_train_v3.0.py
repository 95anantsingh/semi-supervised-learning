import os
import gc
import json
import torch
import random
import warnings
from tqdm import tqdm
from time import sleep
from copy import deepcopy
from typing import Callable
from argparse import ArgumentParser
from tqdm.contrib.logging import logging_redirect_tqdm

from utils.log import get_logger
from utils.python_utils import dotdict
from utils.config import StrategyValue, SelfTrainConfig

from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax
from torch.utils.data import Dataset, DataLoader
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score

from datasets import logging as dlog
from datasets import concatenate_datasets
from datasets.load import load_from_disk

from transformers import logging as tlog
from transformers import AdamW, DataCollatorWithPadding
from transformers import RobertaTokenizer, RobertaForSequenceClassification

tlog.set_verbosity_error()
dlog.set_verbosity_error()
global_seed = 100
random.seed(global_seed)
warnings.filterwarnings("ignore")
    

VERSION = '3.0'


class STDataset(Dataset):
    def __init__(self, dataset, indices=None, labels=None) -> None:
        self.indices = indices
        self.data = dataset
        self.labels = labels

    def __getitem__(self, index):
        input_ids = self.data[index]['input_ids']
        attention_mask = self.data[index]['attention_mask']
        label = self.labels[index] if self.labels != None else self.data[index]['labels']
        return {'input_ids':input_ids, 'attention_mask':attention_mask, 'label':label}

    def __len__(self):
        if self.indices: return len(self.indices)
        else: return len(self.data)


class Trainer(object):

    def __init__(self, config: SelfTrainConfig) -> None:
        
        self.version = VERSION
        self.config = config
        
        self.print_banner()
        self.config_check()
        
        self.labels = []
        self.org_labels = []
        self.eval_indices = []
        self.labeled_indices = []
        self.unlabeled_indices = []
        self.selected_indices = []

        # Setup Logger
        self.logger = get_logger('root')

        # Setup paths
        self.data_cache_path = os.path.join(self.config.cache_dir,'v'+str(self.version),'data',self.config.main_dataset)
        self.run_cache_path = os.path.join(self.data_cache_path,self.config.run_name)
        self.history_path = os.path.join(self.config.history_dir,'v'+self.version,self.config.main_dataset,self.config.run_name)
        self.state_path = os.path.join(self.history_path,'states')
        self.raw_weights_path = os.path.join(self.config.cache_dir,'model')

        os.makedirs(self.run_cache_path,exist_ok=True)
        os.makedirs(self.state_path,exist_ok=True)
        os.makedirs(self.raw_weights_path,exist_ok=True)

        if self.config.strategy=='threshold': prefix = '%.4f' %(self.config.strategy_value)    
        else: prefix = str(int(self.config.strategy_value))
        # threshold/top_k  labeled_smapling labeled_size unlabeled_size unlabeled_in_domain_ratio history.pt 
        self.config_prefix = prefix+'_'+self.config.labeled_sampling+'_'+\
            str(self.config.labeled_size)+'_'+str(self.config.unlabeled_size)+'_'+\
            str(int(self.config.unlabeled_in_domain_ratio*100))

        # Load datasets
        self.load_datsets()
        self.datasets = {idx:dataset for idx,dataset in enumerate([self.config.main_dataset]+self.config.other_datasets)}   
        self.num_classes_dict = {}
        for idx, dataset in self.datasets.items():
            dataset_info = json.load(open(os.path.join(self.config.data_dir,dataset,'tokenized/train/dataset_info.json'),'r'))
            self.num_classes_dict[idx] = len(dataset_info['features']['labels']['names'])
        self.num_classes = self.num_classes_dict[0]    

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.criterion = CrossEntropyLoss()
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Setup metrics
        self.acc_metric = MulticlassAccuracy(device=self.device)
        self.cw_f1_metric = MulticlassF1Score(num_classes=self.num_classes, average=None, device=self.device)
        self.cw_acc_metric = MulticlassAccuracy(num_classes=self.num_classes, average=None, device=self.device)

        # Setup Model
        self.logger.info('Setting up Model')
        self.model = RobertaForSequenceClassification.from_pretrained("roberta-base",
            num_labels=self.num_classes, 
            cache_dir=os.path.join(self.config.cache_dir,'model'))
        self.model.to(self.device)

        self.raw_weights_file_path = os.path.join(self.raw_weights_path,'raw_weights_'+str(self.num_classes)+'.pt')
        if not os.path.exists(self.raw_weights_file_path):
            torch.save(self.model.state_dict(),self.raw_weights_file_path)

        # Setup Test Dataloader
        test_data = STDataset(self.test_data)
        self.test_dataloader = DataLoader(test_data, batch_size=self.config.max_batch_size,collate_fn=self.data_collator)

        # Setup history dict
        self.history = {
            'strategy': self.config.strategy, 'config':self.config.as_dict(),
            'iteration':[],'batch_size':[], 'epoch':[],
            
            'train_loss':[], 'train_acc':[],
            'val_loss':[], 'val_cw_f1':[], 'val_acc':[], 'val_cw_acc':[],
            'test_loss':[], 'test_cw_f1':[], 'test_acc':[], 'test_cw_acc':[],

            'labeled_set_size':[], 'unlabeled_set_size':[], 'selected_set_size':[], 'selected_in_domain':[],
            'selected_out_domain':[],

            'train_distribution':[], 'labeled_distribution':[], 'unlabeled_distribution':[],
            'selected_distribution':[],'predicted_distribution':[],'pseudo_distribution': [],
            
            'predicted_probs':[],
            'predicted_prob_mean':[], 'predicted_prob_std':[], 'predicted_prob_max':[], 'predicted_prob_min':[],

            'selected_probs':[],
            'selected_prob_mean':[], 'selected_prob_std':[], 'selected_prob_max':[], 'selected_prob_min':[],
            }

        # Make sets
        self._make_sets()

        # Setup Eval Dataloader
        eval_data = STDataset(self.test_data.select(self.eval_indices),self.eval_indices)
        self.eval_dataloader = DataLoader(eval_data, batch_size=self.config.max_batch_size,collate_fn=self.data_collator)


    def _make_sets(self):
        # Set Paths
        eval_idx_path = os.path.join(self.data_cache_path,'eval_indices.pt')
        labeled_idx_path = os.path.join(self.run_cache_path,'msr_'+str(int(self.config.main_sampling_ratio*100))+'_'+
            self.config.labeled_sampling+'_'+str(self.config.labeled_size)+'_labeled_indices.pt')
        unlabeled_idx_path = os.path.join(self.run_cache_path,'msr_'+str(int(self.config.main_sampling_ratio*100))+'_ubr_'+
            str(int(self.config.unlabeled_in_domain_ratio*100))+'_'+str(self.config.unlabeled_size)+'_unlabeled_indices.pt')
        

        self.org_labeled_indices = []
        self.org_unlabeled_indices = []
        
        labeled_lock = os.path.join(self.run_cache_path,'labeled_processing.lock')
        unlabeled_lock = os.path.join(self.run_cache_path,'unlabeled_processing.lock')
        eval_lock = os.path.join(self.data_cache_path,'eval_processing.lock')
  
        # Check if unlabeled setup is running
        while os.path.isfile(labeled_lock): sleep(120)

        # Load Labeled Set
        self.logger.info('Loading Sets')
        if os.path.isfile(labeled_idx_path):
            self.labeled_indices = torch.load(labeled_idx_path)
        else:
            file = open(labeled_lock,'w')
            file.close()
            self.logger.debug('Cache not found for Labeled Set')
            self._make_labeled_set()
            torch.save(self.labeled_indices,labeled_idx_path)
            os.remove(labeled_lock)
        self.org_labeled_indices = self.labeled_indices

        # Check if unlabeled setup is running
        while os.path.isfile(unlabeled_lock): sleep(120)

        # Load Unlabeled Set
        if os.path.isfile(unlabeled_idx_path):
            unlabeled_set = torch.load(unlabeled_idx_path)
            in_domain_idx = unlabeled_set['in_domain_idx']
            out_of_domain_idx = unlabeled_set['out_of_domain_idx']
        else:
            file = open(unlabeled_lock,'w')
            file.close()
            self.logger.debug('Cache not found for Unlabeled Set')
            in_domain_idx, out_of_domain_idx = self._make_unlabeled_set()
            unlabeled_set = dict(in_domain_idx=in_domain_idx,out_of_domain_idx=out_of_domain_idx)
            torch.save(unlabeled_set,unlabeled_idx_path)
            os.remove(unlabeled_lock)
        self.org_unlabeled_indices = unlabeled_set
        self.unlabeled_data = concatenate_datasets([self.train_data.select(in_domain_idx),self.other_data.select(out_of_domain_idx)])
        self.unlabeled_data = self.unlabeled_data.shuffle()
        self.unlabeled_indices = [*range(len(self.unlabeled_data))]  
        
        self.logger.debug('In Domain Size: '+str(len(in_domain_idx)))
        self.logger.debug('Out of Domain Size: '+str(len(out_of_domain_idx)))

        # Make Complete Traning Set
        self.train_data = concatenate_datasets([self.unlabeled_data, self.train_data.select(self.labeled_indices)])

        # Update Labeled indices
        self.labeled_indices = [*range(self.config.unlabeled_size,self.config.unlabeled_size+self.config.labeled_size)] 

        # Collect garbage
        del(self.unlabeled_data)
        del(self.other_data)
        gc.collect()

        # Check if eval setup is running
        while os.path.isfile(eval_lock): sleep(120)

        # Load Eval Set
        if os.path.isfile(eval_idx_path):
            self.eval_indices = torch.load(eval_idx_path)
        else:
            self.logger.debug('Cache not found for Evaluation Set')
            file = open(eval_lock,'w')
            file.close()
            self._make_eval_set()
            torch.save(self.eval_indices,eval_idx_path)
            os.remove(eval_lock)
        self.logger.info('All Sets loaded')


    def _make_labeled_set(self):
        if not self.config.sorted_data:
            raise NotImplementedError(self.__class__.__name__ + ': No implementation found for unsorted data')

        classwise_indices, max_class_size = self.get_class_indices(self.train_data)
        
        # Limit the samples
        split_point = int(self.config.main_sampling_ratio * max_class_size)
        classwise_indices = {key:val[0:split_point] for key, val in classwise_indices.items()}

        # Equally sampled labeled set
        if self.config.labeled_sampling=='equal':
            per_class = self.config.labeled_size // self.num_classes
            assert per_class < split_point, "Not enough samples to create Labeled Set."
            for class_ in range(self.num_classes):
                self.labeled_indices += classwise_indices[class_][0:per_class]
            self.labeled_indices = random.sample(self.labeled_indices,len(self.labeled_indices))
        # Randomly sampled labeled set
        elif self.config.labeled_sampling=='random':
            all_labeled_indices = []
            for class_, indices in classwise_indices.items():
                all_labeled_indices+=indices
            assert self.labeled_size <= len(all_labeled_indices), "Not enough samples to create Labeled Set."
            self.labeled_indices = random.sample(all_labeled_indices,self.labeled_size)
        else:
            raise NotImplementedError(self.__class__.__name__ + \
                ': No implementation found for Labeled Set sampling strategy: '+self.config.labeled_sampling)


    def _make_unlabeled_set(self):
        
        classwise_indices, max_class_size = self.get_class_indices(self.train_data)
        
        # Limit the samples
        split_point = int(self.config.main_sampling_ratio * max_class_size)
        classwise_indices = {key:val[split_point:] for key, val in classwise_indices.items()}

        all_unlabeled_indices = []
        for indices in classwise_indices.values():
            all_unlabeled_indices+=indices

        in_domain_size = int(self.config.unlabeled_in_domain_ratio * self.config.unlabeled_size)
        out_of_domain_size = self.config.unlabeled_size - in_domain_size

        self.logger.debug('In Domain Size: '+str(in_domain_size))
        self.logger.debug('Out of Domain Size: '+str(out_of_domain_size))

        assert in_domain_size <= len(all_unlabeled_indices), "Not enough main dataset samples to create Unabeled Set."

        # in_domain_idx = random.sample(list(set([*range(len(self.train_data))]) - set(self.labeled_indices)),in_domain_size)
        in_domain_idx = random.sample(all_unlabeled_indices,in_domain_size)
        out_of_domain_idx = random.sample([*range(len(self.other_data))],out_of_domain_size)
        
        # self.unlabeled_indices = random.sample(self.unlabeled_indices,len(self.unlabeled_indices))
        return in_domain_idx, out_of_domain_idx



    def _make_eval_set(self):

        # Set paths
        eval_model_path = os.path.join(self.data_cache_path,'eval_model.pt')
        eval_preds_path = os.path.join(self.data_cache_path,'eval_preds.pt')

        # Setup model
        model = RobertaForSequenceClassification.from_pretrained("roberta-base",
            num_labels=self.num_classes, 
            cache_dir=os.path.join(self.config.cache_dir,'model'))
        model.to(self.device)
        
        # Train model on full Train Set
        if os.path.isfile(eval_model_path):
            model.load_state_dict(torch.load(eval_model_path,map_location=self.device))
        else:
            num_epochs = self.config.num_epochs
            train_data = STDataset(self.train_data)
            train_dataloader = DataLoader(train_data, batch_size=self.config.max_batch_size,
                collate_fn=self.data_collator,shuffle=True)
            
            optimizer = AdamW(model.parameters(), lr=5e-05)

            train_pbar = tqdm(desc='Training   ', unit=' batch', colour='blue', total= len(train_dataloader))
            eval_pbar  = tqdm(desc='Validation ', unit=' batch', colour='white', total= len(self.test_dataloader))
            epoch_pbar = tqdm(desc='Epoch      ', unit=' epoch', colour='green', total= num_epochs)

            max_acc = 0
            for epoch in range(num_epochs):
                tr_loss, tr_acc = self._train(model, train_dataloader, optimizer, train_pbar)
                ev_loss, _, ev_acc, _ = self._eval(model, self.test_dataloader, eval_pbar)
                epoch_pbar.write('Epoch: %s ;   Train Loss: %.3f ; Train Acc: %.1f ;   Val Loss: %.3f ; Val Acc: %.1f' \
                                %(epoch+1, tr_loss, tr_acc*100, ev_loss, ev_acc*100))
                epoch_pbar.update(1)
                
                train_pbar.reset()
                eval_pbar.reset()  
                
                if max_acc < ev_acc:
                    torch.save(model.state_dict(), eval_model_path)
                    with open(os.path.join(self.config.cache_dir,'model',
                        self.config.main_dataset+'_eval_model.log'),'a') as file:
                        line = 'Epoch: '+str(epoch+1)+' ; Val Acc: '+str(ev_acc)+'\n'
                        file.write(line)
                    max_acc=ev_acc

            train_pbar.close()
            eval_pbar.close()
            epoch_pbar.close()
        
        # Get predictions
        if os.path.isfile(eval_preds_path):
            preds= torch.load(eval_preds_path,map_location=self.device)
        else:
            test_pbar  = tqdm(desc='Predicting ', unit=' batch', colour='cyan', total=len(self.test_dataloader))
            preds, _ = self._predict(model, self.test_dataloader, test_pbar)   
            test_pbar.close()
            torch.save(preds,eval_preds_path)

        # Seperate predictions
        targets = self.test_data['labels'].to(self.device)
        comparison = (preds != targets.view_as(preds))
        wrong_idx = comparison.nonzero().flatten()
        correct_idx = comparison.logical_not().nonzero().flatten()

        # Get classwise predictions
        wrong_samples_cw_idx = {class_:[] for class_ in range(self.num_classes)}
        correct_samples_cw_idx = {class_:[] for class_ in range(self.num_classes)}
        labels = self.test_data['labels']
        for idx in wrong_idx:
            wrong_samples_cw_idx[labels[idx].item()].append(idx)
        for idx in correct_idx:
            correct_samples_cw_idx[labels[idx].item()].append(idx)

        # Get metrics
        accuracy = len(correct_idx)/(len(wrong_idx)+len(correct_idx))
        cw_accuracy = {cr_key:len(cr_val)/(len(cr_val)+len(wr_val)) for (cr_key, cr_val),
            (_, wr_val) in zip(correct_samples_cw_idx.items(), wrong_samples_cw_idx.items())}

        print('\nAccuracy :', accuracy,'\nCW Acc   :', cw_accuracy,'\n')
        
        # Make Evaluation Set
        samples_per_class = self.config.eval_size // self.num_classes
    
        for class_ in range(self.num_classes):
            num_correct = int(samples_per_class * cw_accuracy[class_])
            num_wrong = samples_per_class - num_correct
            assert num_correct <= len(correct_samples_cw_idx[class_]), "Not enough samples to create Eval Set."
            assert num_wrong <= len(wrong_samples_cw_idx[class_]), "Not enough samples to create Eval Set."
            self.eval_indices += correct_samples_cw_idx[class_][0:num_correct]
            self.eval_indices += wrong_samples_cw_idx[class_][0:num_wrong]

        # self.eval_indices = random.sample(self.eval_indices,len(self.eval_indices))


    def _make_sets_old(self):
        print('\nMaking sets...')
        self.labeled_size = self.config.labeled_size
        self.unlabeled_size = self.config.unlabeled_size

        if self.config.labeled_per_class:
            self.labeled_size = self.config.labeled_per_class * len(self.config.labeled_classes)
            print('\nSelecting',self.config.labeled_per_class,'samples per class (',self.labeled_size,') for Labeled Set...')
        else:
            print('\nSelecting',self.labeled_size,'samples for Labeled Set...')

        if self.config.unlabeled_per_class:
            self.unlabeled_size = self.config.unlabeled_per_class * len(self.config.unlabeled_classes)
            print('Selecting',self.config.unlabeled_per_class,'samples per class (',self.unlabeled_size,') for Unlabeled Set...')
        else:
            print('Selecting',self.unlabeled_size,'samples for Unlabeled Set...')
        
        print('\nSelected',len(self.eval_indices),'samples for Evaluation Set.')

        # Shuffle Indices
        self.labeled_indices = random.sample(self.labeled_indices,len(self.labeled_indices))
        self.unlabeled_indices = random.sample(self.unlabeled_indices,len(self.unlabeled_indices))
        self.eval_indices = random.sample(self.eval_indices,len(self.eval_indices))

        # print(self.get_class_distribution(self.config.labeled_classes,self.train_data.select(self.labeled_indices)))
        # print(self.get_class_distribution(self.config.unlabeled_classes,self.train_data.select(self.unlabeled_indices)))
        # print(self.get_class_distribution(self.config.unlabeled_classes,self.eval_data.select(self.eval_indices)))


    def config_check(self):
        implemented_stategies = ['threshold', 'top_k', 'pc_top_k']
        if self.config.strategy not in implemented_stategies:
            raise NotImplementedError(self.__class__.__name__ + ': This strategy is not implemented yet.')

        assert self.config.strategy_value != 0, 'Value for "'+self.config.strategy+'" should be non-zero.'
        assert self.config.labeled_size, 'Labeled set size should be given.'
        assert self.config.unlabeled_size, ' UnLabeled set size should be given.'


    def load_datsets(self):

        # Set paths
        main_data_path = os.path.join(self.data_cache_path,'dataset_cache','main_data')
        other_data_path = os.path.join(self.data_cache_path,'dataset_cache','other_data')

        self.logger.info('Loading Main Dataset')
        try:
            self.train_data = load_from_disk(os.path.join(main_data_path,'train'))
        except:
            self.logger.debug('Cache not found for train set of main_dataset')
            self.train_data = load_from_disk(os.path.join(self.config.data_dir,self.config.main_dataset,'tokenized/train'))
            self.train_data = self._format_dataset(self.train_data)
            self.train_data = self.train_data.add_column('dataset', [0]*len(self.train_data))
            self.train_data.save_to_disk(os.path.join(main_data_path,'train'))

        try:
            self.test_data = load_from_disk(os.path.join(main_data_path,'test'))
        except:
            self.logger.debug('Cache not found for test set of main_dataset')
            self.test_data = load_from_disk(os.path.join(self.config.data_dir,self.config.main_dataset,'tokenized/test'))
            self.test_data = self._format_dataset(self.test_data)  
            self.test_data.save_to_disk(os.path.join(main_data_path,'test'))   
        
        self.logger.info('Loading Other Datasets')
        other_data_path = os.path.join(self.data_cache_path,'dataset_cache','other_data')
        os.makedirs(other_data_path,exist_ok=True)
        try:
            self.other_data = load_from_disk(other_data_path)
        except:
            self.logger.debug('Cache not found for combined other_datasets')
            datasets = []   
            for dataset_idx, dataset_name in enumerate(self.config.other_datasets,1):
                self.logger.debug('Loading '+dataset_name)
                dataset = load_from_disk(os.path.join(self.config.data_dir,dataset_name,'tokenized/train'))
                dataset = self._format_dataset(dataset)
                dataset = dataset.add_column('dataset', [dataset_idx]*len(dataset))
                datasets.append(dataset)
            self.other_data = concatenate_datasets(datasets)
            self.other_data.save_to_disk(other_data_path)

        self.logger.info('Datasets loaded')
        
              
    def _format_dataset(self,dataset):
        columns_to_keep = {'input_ids', 'attention_mask'}
        labels = dataset['labels']
        content = dataset['content']            
        dataset = dataset.remove_columns(list(set(dataset.features.keys())-columns_to_keep))
        dataset = dataset.add_column('labels', labels.tolist())
        dataset = dataset.add_column('content', content)
        return dataset


    def print_banner(self):
        try: console_width, _ = os.get_terminal_size(0)
        except : console_width = 50
        print('\n\n'+'-'*console_width)
        print('Self Training v3.0\n'+'-'*console_width)
        print('Run Name          :', self.config.run_name)
        print('Strategy          :', self.config.strategy)
        print('Strategy Value    :', self.config.strategy_value)
        print('Main Dataset      :', self.config.main_dataset)
        print('Other Datasets    :', self.config.other_datasets)
        print('Main Sampling R   :', self.config.main_sampling_ratio)
        print('Labeled Size      :', self.config.labeled_size)
        print('Labeled Sampling  :', self.config.labeled_sampling)
        print('Unlabeled Size    :', self.config.unlabeled_size)
        print('Unlabeled InDom R :', self.config.unlabeled_in_domain_ratio)
        print('Eval Size         :', self.config.eval_size)
        print('\n')


    def get_class_indices(self, data):
        class_indices = {}
        prev_label = 0
        start_index = 0
        max_size = data.shape[0]
        for index, label in enumerate(data['labels']):
            if label != prev_label:
                class_indices[label.item()-1]=[*range(start_index,index)]
                if index - start_index < max_size:
                    max_size = index - start_index
                start_index=index
                prev_label=label

        class_indices[label.item()]=[*range(start_index,index+1)]

        if index - start_index < max_size:
            max_size = index - start_index

        return class_indices, max_size
    

    def get_class_distribution(self,classes,data=None,labels=None):
        if labels != None:
            distribution = {clss:0 for clss in range(classes)}
            for lb in labels:
                distribution[lb.item()] += 1 
        else:
            distribution = {self.datasets[dataset_idx]:{clss:0 for clss in range(classes[dataset_idx])} \
                for dataset_idx in range(len(self.config.other_datasets)+1)}
            for dataset,class_ in zip(data['dataset'],data['labels']):
                distribution[self.datasets[dataset]][class_] += 1 
        return distribution


    def get_selection_count(self, distribution, dataset_idx):
        selection_count = 0
        if type(dataset_idx) is not list:
            dataset_idx = [dataset_idx]
        for dataset in dataset_idx:
            for count in distribution[self.datasets[dataset]].values():
                selection_count += count
        return selection_count


    def get_batch_size(self):
        batch_size = min(len(self.labeled_indices)//self.config.steps_per_epoch, self.config.max_batch_size)
        batch_size = max(batch_size, self.config.min_batch_size)
        return batch_size


    def _train(self, model, dataloader, optimizer, progress_bar):
        train_loss = 0.0
        model.train()
        for batch in dataloader:
            batch = tuple(input.to(self.device) for input in batch.data.values())

            optimizer.zero_grad()
            input_ids, attention_mask = batch[0], batch[1]
            output = model(input_ids, attention_mask)

            labels= batch[2]
            loss = self.criterion(output.logits, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            self.acc_metric.update(output.logits, labels)

            sleep(0.1)
            progress_bar.update(1)
        
        train_loss = train_loss / len(dataloader)
        accuracy = self.acc_metric.compute().item()

        self.acc_metric.reset()
        
        return train_loss, accuracy


    def _eval(self, model, dataloader, progress_bar):
        eval_loss = 0.0
        model.eval()
        for batch in dataloader:
            batch = tuple(input.to(self.device) for input in batch.data.values())
           
            input_ids, attension_mask = batch[0], batch[1]
            with torch.no_grad():
                output = model(input_ids, attension_mask)
                labels= batch[2]
                loss = self.criterion(output.logits, labels)

            eval_loss += loss.item()
            self.acc_metric.update(output.logits, labels)
            self.cw_f1_metric.update(output.logits, labels)
            self.cw_acc_metric.update(output.logits, labels)
            
            progress_bar.update(1)
        
        eval_loss = eval_loss / len(dataloader)
        
        accuracy = self.acc_metric.compute().item()
        cw_f1_score = (self.cw_f1_metric.compute().cpu().numpy()).tolist()
        cw_accuracy = (self.cw_acc_metric.compute().cpu().numpy()).tolist()

        self.acc_metric.reset()
        self.cw_f1_metric.reset()
        self.cw_acc_metric.reset()

        return eval_loss, cw_f1_score, accuracy, cw_accuracy


    def _test(self, model, dataloader, progress_bar):
        return self._eval(model, dataloader, progress_bar)


    def _predict(self, model, dataloader, progress_bar):
        probs = []
        preds = []
        model.eval()
        for batch in dataloader:
            batch = tuple(input.to(self.device) for input in batch.data.values())
           
            input_ids, attension_mask = batch[0], batch[1]
            with torch.no_grad():
                output = model(input_ids, attension_mask)

            probabilities = torch.max(softmax(torch.tensor(output.logits),dim=1), dim=1)
            probs.append(probabilities.values)
            preds.append(probabilities.indices)

            progress_bar.update(1)

        return torch.cat(preds,dim=0), torch.cat(probs,dim=0)


    def _train_iteration(self,train_pbar,eval_pbar,epoch_pbar,test_pbar,train_dataloader):
    
        self.history['epoch'].append([])
        self.history['train_loss'].append([])
        self.history['train_acc'].append([])
        self.history['val_loss'].append([])
        self.history['val_cw_f1'].append([])
        self.history['val_acc'].append([])
        self.history['val_cw_acc'].append([])
        
        self.model.load_state_dict(torch.load(self.raw_weights_file_path, map_location=self.device))
        optimizer = AdamW(self.model.parameters(), lr=5e-05)
        
        train_pbar.total = len(train_dataloader)
        train_pbar.refresh()
        eval_pbar.total = len(self.eval_dataloader)
        eval_pbar.refresh()

        if self.config.num_epochs * len(train_dataloader) < self.config.min_total_steps:
            num_epochs = self.config.min_total_steps//len(train_dataloader)
        else: num_epochs = self.config.num_epochs

        epoch_pbar.total  = num_epochs
        epoch_pbar.refresh()
        
        best_model = dotdict(state=None, acc=-1)
        for epoch in range(num_epochs):
            tr_loss, tr_acc = self._train(self.model, train_dataloader, optimizer, train_pbar)
            ev_loss, ev_cw_f1, ev_acc, ev_cw_acc = self._eval(self.model, self.eval_dataloader, eval_pbar)
            
            if ev_acc > best_model.acc:
                best_model.acc = ev_acc
                best_model.state = deepcopy(self.model.state_dict())
                
            self.history['epoch'][-1].append(epoch+1)
            self.history['train_loss'][-1].append(tr_loss)
            self.history['train_acc'][-1].append(tr_acc)
            self.history['val_loss'][-1].append(ev_loss)
            self.history['val_cw_f1'][-1].append(ev_cw_f1)
            self.history['val_acc'][-1].append(ev_acc)
            self.history['val_cw_acc'][-1].append(ev_cw_acc)

            epoch_pbar.update(1)
            epoch_pbar.write('Epoch: %s ;   Train Loss: %.3f ; Train Acc: %.1f ;   Val Loss: %.3f ; Val Acc: %.1f' \
                            %(epoch+1, tr_loss, tr_acc*100, ev_loss, ev_acc*100))

            train_pbar.reset()
            eval_pbar.reset()

        self.model.load_state_dict(best_model.state)
        ts_loss, ts_cw_f1, ts_acc, ts_cw_acc = self._test(self.model, self.test_dataloader, test_pbar)

        self.history['test_loss'].append(ts_loss)
        self.history['test_cw_f1'].append(ts_cw_f1)
        self.history['test_acc'].append(ts_acc)
        self.history['test_cw_acc'].append(ts_cw_acc)

        epoch_pbar.write('Test Loss: %.3f ; Test Acc: %.1f' \
            %(ts_loss, ts_acc*100))
        
        return best_model

    def _select_iteration(self,pred_pbar):
        self.selected_indices = []

        if len(self.unlabeled_indices):
            pred_data = STDataset(self.train_data.select(self.unlabeled_indices),self.unlabeled_indices)
            pred_dataloader = DataLoader(pred_data, batch_size=self.config.max_batch_size,collate_fn=self.data_collator,shuffle=False)
            pred_pbar.total = len(pred_dataloader)
            pred_pbar.refresh()

            predicted_labels, predicted_probs = self._predict(self.model, pred_dataloader, pred_pbar)

            self.history['predicted_probs'].append(predicted_probs.detach().cpu().numpy())
            self.history['predicted_prob_mean'].append(torch.mean(predicted_probs).item())
            self.history['predicted_prob_std'].append(torch.std(predicted_probs).item())
            self.history['predicted_prob_max'].append(torch.max(predicted_probs).item())
            self.history['predicted_prob_min'].append(torch.min(predicted_probs).item())
            self.history['predicted_distribution'].append(self.get_class_distribution(self.num_classes,labels=predicted_labels))

            selected_probs = []
            if self.config.strategy=='threshold':
                for idx,pseudo_prob,pseudo_label in zip(self.unlabeled_indices,predicted_probs,predicted_labels):
                    if pseudo_prob > self.config.strategy_value:
                        self.labels[idx] = pseudo_label
                        self.selected_indices.append(idx)
                        selected_probs.append(pseudo_prob.detach().item())
                selected_probs = torch.tensor(selected_probs)

            elif self.config.strategy=='top_k':
                top_k = min(len(self.unlabeled_indices),int(self.config.strategy_value))
                selected_probs = torch.topk(predicted_probs,top_k)
                for pseudo_prob, idx in zip(selected_probs.values, selected_probs.indices):
                    self.labels[self.unlabeled_indices[idx]] = predicted_labels[idx]
                    self.selected_indices.append(self.unlabeled_indices[idx])
                selected_probs = selected_probs.values.detach()
            
            elif self.config.strategy=='pc_top_k':
                for class_ in range(self.num_classes):
                    class_indices = torch.where(predicted_labels==class_)[0]
                    top_k = min(len(class_indices),int(self.config.strategy_value))
                    class_probs = torch.index_select(predicted_probs, 0, class_indices)
                    selected = torch.topk(class_probs,top_k)
                    for pseudo_prob, idx in zip(selected.values, selected.indices):
                        self.labels[self.unlabeled_indices[class_indices[idx]]] = predicted_labels[class_indices[idx]]
                        self.selected_indices.append(self.unlabeled_indices[class_indices[idx]])
                    selected_probs.append(selected.values)
                selected_probs = torch.cat(selected_probs,dim=0).detach()

            self.history['selected_probs'].append(selected_probs.cpu().numpy())
            self.history['selected_prob_mean'].append(torch.mean(selected_probs).item())
            self.history['selected_prob_std'].append(torch.std(selected_probs).item())
            self.history['selected_prob_min'].append(torch.min(selected_probs).item())
            self.history['selected_prob_max'].append(torch.max(selected_probs).item())

            pseudo_labels = torch.index_select(self.labels, 0, torch.tensor(self.selected_indices, device=self.device))

            self.history['pseudo_distribution'].append(self.get_class_distribution(self.num_classes, labels=pseudo_labels))
            self.history['selected_distribution'].append(self.get_class_distribution(self.num_classes_dict, self.train_data.select(self.selected_indices)))
            self.history['selected_set_size'].append(len(self.selected_indices))
            self.history['selected_in_domain'].append(self.get_selection_count(self.history['selected_distribution'][-1],0))
            self.history['selected_out_domain'].append(self.get_selection_count(self.history['selected_distribution'][-1],[*range(1,len(self.datasets))]))


    def self_train(self):       

        self.labels = torch.tensor(self.train_data['labels']).to(self.device)
        
        iteration = 0
        max_iters = 10000 if self.config.num_iters==-1 else self.config.num_iters
        last_length = -1
        patience_counter = 0
        try: console_width, _ = os.get_terminal_size(0)
        except : console_width = 50
        last_total_time = 0
        last_state_path = ""
        with logging_redirect_tqdm():

            train_pbar = tqdm(leave=False, desc='Training   ', unit=' batch', colour='#7986CB')
            eval_pbar  = tqdm(leave=False, desc='Validation ', unit=' batch', colour='#3F51B5')
            epoch_pbar = tqdm(leave=False, desc='Epoch      ', unit=' epoch', colour='#43A047',total= self.config.num_epochs)
            test_pbar  = tqdm(leave=False, desc='Testing    ', unit=' batch', colour='#E0E0E0',total= len(self.test_dataloader))
            pred_pbar = tqdm(leave=False,  desc='Prediction ', unit=' batch', colour='#42A5F5',
                total=self.config.unlabeled_size//self.config.max_batch_size)
            iter_pbar = tqdm(leave=False,  desc='Iteration  ', unit=' itr',   colour='#EF5350',total= 1)

            while len(self.unlabeled_indices)>0 and patience_counter < self.config.max_patience and iteration<max_iters:      
                iteration += 1
                iter_pbar.write('\n\n'+'-'*console_width+'\n\nIteration: '+str(iteration)+'\n')
                iter_pbar.total += 1
                iter_pbar.refresh()

                self.labeled_indices += self.selected_indices 
                self.unlabeled_indices = list(set(self.unlabeled_indices) - set(self.selected_indices))

                batch_size = self.get_batch_size()
                
                labels = torch.index_select(self.labels, 0, torch.tensor(self.labeled_indices, device=self.device))
                train_data = STDataset(self.train_data.select(self.labeled_indices), self.labeled_indices, labels)
                train_dataloader = DataLoader(train_data, batch_size=batch_size,collate_fn=self.data_collator,shuffle=True)
                
                self.history['iteration'].append(iteration)    
                self.history['labeled_set_size'].append(len(self.labeled_indices))
                self.history['unlabeled_set_size'].append(len(self.unlabeled_indices))
                self.history['batch_size'].append(batch_size)
                self.history['train_distribution'].append(self.get_class_distribution(self.num_classes,labels=labels))
                self.history['labeled_distribution'].append(self.get_class_distribution(self.num_classes_dict, self.train_data.select(self.labeled_indices)))
                self.history['unlabeled_distribution'].append(self.get_class_distribution(self.num_classes_dict, self.train_data.select(self.unlabeled_indices)))
                
                iter_pbar.write('\nBatch Size: %s;   Labeled Size: %s ; Unlabeled Size: %s'\
                                %(batch_size, len(self.labeled_indices), len(self.unlabeled_indices)))
                iter_pbar.write('\nClass Distribution')
                iter_pbar.write('Train Set     : '+str(self.history['train_distribution'][-1]))
                iter_pbar.write('Labeled Set   : '+str(self.history['labeled_distribution'][-1]))
                iter_pbar.write('Unlabeled Set : '+str(self.history['unlabeled_distribution'][-1])+'\n')
            
                # train_pbar.close()
                # eval_pbar.close()
                # epoch_pbar.close()
                # test_pbar.close()
                # pred_pbar.close()
                # iter_pbar.close()

                # exit()
                # Training
                best_model = self._train_iteration(train_pbar,eval_pbar,epoch_pbar,test_pbar,train_dataloader)

                # Selection 
                self._select_iteration(pred_pbar)
                
                # Update patience counter
                if len(self.unlabeled_indices) == last_length:
                    patience_counter+=1
                last_length = len(self.unlabeled_indices)

                # Save state
                if iteration%self.config.save_interval==0 or len(self.unlabeled_indices)==0 or \
                    patience_counter == self.config.max_patience or iteration==max_iters:
                    if len(self.unlabeled_indices)==0 or \
                        patience_counter == self.config.max_patience or iteration==max_iters:
                        history = "Saved in History Dir seperately"
                    else: history = self.history
                    state = dict(
                        state_dict = best_model.state,
                        acc = best_model.acc,
                        labeled_indices = self.org_labeled_indices,
                        unlabeled_indices = self.org_unlabeled_indices,
                        history = history 
                    )
                    if os.path.isfile(last_state_path): os.remove(last_state_path)
                    save_path = os.path.join(self.state_path,self.config_prefix+'_e'+str(iteration)+'_state.pt')
                    torch.save(state, save_path)
                    last_state_path = save_path

                
                iter_pbar.write('\nSelected Size: %s ; In Domain: %s ; Out of Domain: %s ; Min Prob: %.3f ; Max Prob: %.3f ;'\
                    %(len(self.selected_indices),self.history['selected_in_domain'][-1],self.history['selected_out_domain'][-1],
                    self.history['predicted_prob_min'][-1], self.history['predicted_prob_max'][-1]))
                
                iter_pbar.write('\nClass Distribution')
                iter_pbar.write('Predictions   : '+str(self.history['predicted_distribution'][-1]))
                iter_pbar.write('Selected Set  : '+str(self.history['selected_distribution'][-1]))
                iter_pbar.write('Pseudo Labels : '+str(self.history['pseudo_distribution'][-1]))
                total_time = iter_pbar.format_dict['elapsed']
                iter_pbar.write('\nTime Taken : '+iter_pbar.format_interval(total_time-last_total_time))
                iter_pbar.write('Total Time : '+iter_pbar.format_interval(total_time))
                last_total_time = iter_pbar.format_dict['elapsed']
    
                iter_pbar.update(1)
                epoch_pbar.reset()
                test_pbar.reset()
                pred_pbar.reset()
                
            iter_pbar.total -= 1
            iter_pbar.refresh()
            train_pbar.close()
            eval_pbar.close()
            epoch_pbar.close()
            test_pbar.close()
            pred_pbar.close()
            iter_pbar.close()
        
        # Print reason for stopping
        if patience_counter == self.config.max_patience:
            print("\nEarly Stopping, Max Patience Over.\n")
        if len(self.unlabeled_indices) == 0:
            print("\nEmpty Unlabeled Set, Stopping.\n")

        # Save history
        self.history_file_path = os.path.join(self.history_path, self.config_prefix+'_e'+str(iteration)+'_history.pt')
        try:
            torch.save(self.history,self.history_file_path)
        except Exception as e:
            print('Exception Occured:\n%s'%(e))
            # name = input("\nEnter file name to save: ",type=str)
            name = "history_file.pt"
            torch.save(self.history,name)
                
        return self.history


def parse_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-n','--run_name', type=str)
    arg_parser.add_argument('-s','--strategy', type=str)
    arg_parser.add_argument('-sv','--strategy_value', type=float)
    arg_parser.add_argument('-md','--main_dataset', type=str)
    arg_parser.add_argument('-od','--other_datasets', type=str, nargs='+')
    
    arg_parser.add_argument('-msr','--main_sampling_ratio', type=float)
    arg_parser.add_argument('-ls','--labeled_size', type=int)
    arg_parser.add_argument('-lbs','--labeled_sampling', type=str)
    arg_parser.add_argument('-us','--unlabeled_size', type=int)
    arg_parser.add_argument('-ubr','--unlabeled_in_domain_ratio', type=float)
    arg_parser.add_argument('-es','--eval_size', type=int)
    arg_parser.add_argument('-hd','--history_dir', type=str)

    return arg_parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # Test
    if args.run_name == None:
        args.run_name = 'test'
        args.strategy = 'pc_top_k'
        args.strategy_value = 100
        args.main_dataset = 'yahoo_answers'
        args.other_datasets = ['ag_news', 'dbpedia_14', 'yelp_review_full']
    
        args.labeled_size = 50
        args.labeled_sampling = 'equal'
        args.unlabeled_size = 500
        args.unlabeled_in_domain_ratio = 0.5
    
    
    trainer_config = SelfTrainConfig(
        run_name= args.run_name,
        strategy = args.strategy,
        strategy_value = args.strategy_value,
        main_dataset = args.main_dataset,
        other_datasets = args.other_datasets,
        labeled_size = args.labeled_size,
        labeled_sampling = args.labeled_sampling,
        unlabeled_size = args.unlabeled_size,
        unlabeled_in_domain_ratio = args.unlabeled_in_domain_ratio)


    trainer = Trainer(trainer_config)
    trainer.self_train()