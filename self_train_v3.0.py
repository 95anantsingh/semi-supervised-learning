import os
import torch
import random
import warnings
from tqdm import tqdm
from argparse import ArgumentParser
from typing import NamedTuple, List
from numpy.random import default_rng

from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax
from torch.utils.data import Dataset, DataLoader
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score

from datasets import logging as dlog
from datasets.load import load_from_disk

from transformers import logging as tlog
from transformers import AdamW, DataCollatorWithPadding
from transformers import RobertaTokenizer, RobertaForSequenceClassification


tlog.set_verbosity_error()
dlog.set_verbosity_error()
global_seed = 100
random.seed(global_seed)
warnings.filterwarnings("ignore")



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
    

class STDataset(Dataset):
    def __init__(self, dataset, indices, labels=None) -> None:
        self.indices = indices
        self.data = dataset
        self.labels = labels

    def __getitem__(self, index):
        input_ids = self.data[index]['input_ids']
        attention_mask = self.data[index]['attention_mask']
        label = self.labels[index] if self.labels != None else self.data[index]['labels']
        return {'input_ids':input_ids, 'attention_mask':attention_mask, 'label':label}

    def __len__(self):
        return len(self.indices)


class Trainer(object):

    def __init__(self, config: SelfTrainConfig) -> None:
        self.config = config

        print('\n\n'+'-'*os.get_terminal_size().columns)
        print('Self Training\n'+'-'*os.get_terminal_size().columns)
        print('Dataset           :', self.config.train_data_path.split('/')[-3])
        print('Strategy          :', self.config.strategy)
        if self.config.strategy=='top_k':
            print('Top K             :', self.config.top_k)
        elif self.config.strategy=='pc_top_k':
            print('Per Class Top K   :', self.config.pc_top_k)
        elif self.config.strategy=='threshold':
            print('Threshold         :', self.config.threshold)
        print('Labeled Classes   :', self.config.labeled_classes)
        print('Unlabeled Classes :', self.config.unlabeled_classes)
        print('History Dir       :', self.config.history_dir)
       
        implemented_stategies = ['threshold', 'top_k', 'pc_top_k']
        if self.config.strategy not in implemented_stategies:
            raise NotImplementedError(self.__class__.__name__ + ': This strategy is not implemented yet.')

        # TODO: assert strategy value is non zero

        assert self.config.labeled_size or self.config.labeled_per_class,\
            'Either per class or total Labeled set size should be given.'
        assert self.config.unlabeled_size or self.config.unlabeled_per_class,\
            'Either per class or total UnLabeled set size should be given.'

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        self.train_data = load_from_disk(self.config.train_data_path)
        self.eval_data  = load_from_disk(self.config.eval_data_path)

        self.criterion = CrossEntropyLoss()

        self.weights_path = os.path.join(self.config.cache_dir,'model','model_weights_'+str(len(self.config.labeled_classes))+'.pt')
        self.model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=len(self.config.labeled_classes), 
                    cache_dir=os.path.join(self.config.cache_dir,'model'))
        self.model.to(self.device)
        
        # Save model with number of output class and only read after that
        if not os.path.exists(self.weights_path):
            torch.save(self.model.state_dict(),self.weights_path)

        self.history = {'strategy': self.config.strategy, 'config':self.config._asdict(),
                        'iteration':[],'batch_size':[], 'epoch':[],
                        
                        'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_cw_f1':[], 'val_acc':[], 'val_cw_acc':[],

                        'labeled_set_size':[], 'unlabeled_set_size':[], 'selected_set_size':[], 'selected_in_domain':[],
                        'selected_out_domain':[],

                        'train_distribution':[], 'labeled_distribution':[], 'unlabeled_distribution':[],
                        'selected_distribution':[],'predicted_distribution':[],'pseudo_distribution': [],
                        
                        'predicted_probs':[],
                        'predicted_prob_mean':[], 'predicted_prob_std':[], 'predicted_prob_max':[], 'predicted_prob_min':[],

                        'selected_probs':[],
                        'selected_prob_mean':[], 'selected_prob_std':[], 'selected_prob_max':[], 'selected_prob_min':[],
                       }

        self.labels = []
        self.org_labels = []
        self.eval_indices = []
        self.labeled_indices = []
        self.unlabeled_indices = []
        self.selected_indices = []
        
        self.acc_metric = MulticlassAccuracy(device=self.device)
        self.cw_f1_metric = MulticlassF1Score(num_classes=len(self.config.labeled_classes), average=None, device=self.device)
        self.cw_acc_metric = MulticlassAccuracy(num_classes=len(self.config.labeled_classes), average=None, device=self.device)

        self._make_sets()


    def _make_sets(self):
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

        # Get classwise indices
        train_classwise_indices, max_train_class_size = self.get_class_indices(self.train_data)
        eval_classwise_indices, max_eval_class_size = self.get_class_indices(self.eval_data)

        if self.config.sorted_data:
            if self.config.equal_split:
                # Prepare labeled set
                # Calculate split point
                labeled_split_point = self.labeled_size//len(self.config.labeled_classes)
                # Raise error
                assert labeled_split_point < max_train_class_size, "Not enough samples to create Labeled set."
                # Append Indices
                for class_ in self.config.labeled_classes:
                    self.labeled_indices += train_classwise_indices[class_][0:labeled_split_point]

                # Prepare unlabeled set
                # Calculate split point
                unlabeled_split_point = labeled_split_point + (self.unlabeled_size//len(self.config.unlabeled_classes))
                # Raise error
                assert unlabeled_split_point < max_train_class_size, "Not enough samples to create Unlabeled set."
                # Append Indices
                for class_ in self.config.unlabeled_classes:
                    self.unlabeled_indices += train_classwise_indices[class_][labeled_split_point:unlabeled_split_point]

                print('\nSelected training sets Equally.')

            else:
                # Prepare labeled set
                labeled_class_indices = []
                for class_, indices in train_classwise_indices.items():
                    if class_ in self.config.labeled_classes:
                        labeled_class_indices+=indices
                # Raise error
                assert self.labeled_size <= len(labeled_class_indices), "Not enough samples to create Labeled set."
                # Select labeled indices
                self.labeled_indices = random.sample(labeled_class_indices,self.labeled_size)

                # Prepare unlabeled set
                all_indices = []
                for class_, indices in train_classwise_indices.items():
                    if class_ in self.config.unlabeled_classes:
                        all_indices+=indices
                all_indices=list(set(all_indices) - set(self.labeled_indices))
                # Raise error
                assert self.unlabeled_size <= len(all_indices), "Not enough samples to create Unlabeled set."
                # Select unlabeled indices
                self.unlabeled_indices = random.sample(all_indices,self.unlabeled_size)

                print('\nSelected training sets Randomly.')

        else:
            raise NotImplementedError(self.__class__.__name__ + ': No implementation found for unsorted data')

        # Prepare evaluation set
        # Calculate split point
        eval_split_point = self.config.eval_size//len(self.config.labeled_classes)
        # Raise error
        assert eval_split_point < max_eval_class_size, "Not enough samples to create Evaluation set."
        # Append eval indices
        for class_ in self.config.labeled_classes:
            self.eval_indices += eval_classwise_indices[class_][0:eval_split_point]
        
        print('\nSelected',len(self.eval_indices),'samples for Evaluation Set.')

        # Shuffle Indices
        self.labeled_indices = random.sample(self.labeled_indices,len(self.labeled_indices))
        self.unlabeled_indices = random.sample(self.unlabeled_indices,len(self.unlabeled_indices))
        self.eval_indices = random.sample(self.eval_indices,len(self.eval_indices))

        # print(self.get_class_distribution(self.config.labeled_classes,self.train_data.select(self.labeled_indices)))
        # print(self.get_class_distribution(self.config.unlabeled_classes,self.train_data.select(self.unlabeled_indices)))
        # print(self.get_class_distribution(self.config.unlabeled_classes,self.eval_data.select(self.eval_indices)))


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
        distribution = {clss:0 for clss in sorted(classes)}
        if labels != None:
            for lb in labels:
                distribution[lb.item()] += 1 
        else:
            for val in data['labels'].numpy():
                distribution[val] += 1 
        return distribution


    def get_class_count(self, distribution, classes):
        class_count = 0
        for class_, count in distribution.items():
            if class_ in classes : class_count += count
        return class_count


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
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            self.acc_metric.update(output.logits, labels)

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


    def self_train(self):       

        self.labels = self.train_data['labels'].to(self.device)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        iteration = 0
        last_length = -1
        patience_counter = 0

        train_pbar = tqdm(leave=False, desc='Training   ', unit=' batch', colour='blue')
        eval_pbar  = tqdm(leave=False, desc='Validation ', unit=' batch', colour='yellow')
        epoch_pbar = tqdm(leave=False, desc='Epoch      ', unit=' epoch', colour='green', total= self.config.num_epochs)
        pred_pbar = tqdm(leave=False,  desc='Prediction ', unit=' batch', colour='magenta',\
            total=self.unlabeled_size//self.config.batch_size)
        iter_pbar = tqdm(leave=False,  desc='Iteration  ', unit=' itr', colour='red', total= 1)
        console_width, _ = os.get_terminal_size(0)
        last_total_time = 0
        while len(self.unlabeled_indices)>0 and patience_counter < self.config.max_patience:      
                  
            iteration += 1
            iter_pbar.write('\n\n'+'-'*console_width+'\n\nIteration: '+str(iteration)+'\n')

            self.labeled_indices  += self.selected_indices 
            self.unlabeled_indices = list(set(self.unlabeled_indices) - set(self.selected_indices))

            batch_size = min(len(self.labeled_indices)//self.config.steps_per_epoch, self.config.batch_size)
            batch_size = max(batch_size, self.config.min_batch_size)
            
            labels = torch.index_select(self.labels, 0, torch.tensor(self.labeled_indices, device=self.device))

            train_data = STDataset(self.train_data.select(self.labeled_indices), self.labeled_indices, labels)
            train_dataloader = DataLoader(train_data, batch_size=batch_size,collate_fn=data_collator,shuffle=True)
            
            eval_data = STDataset(self.eval_data.select(self.eval_indices),self.eval_indices)
            eval_dataloader = DataLoader(eval_data, batch_size=self.config.batch_size,collate_fn=data_collator,shuffle=True)
            

            self.history['epoch'].append([])
            self.history['train_loss'].append([])
            self.history['train_acc'].append([])
            self.history['val_loss'].append([])
            self.history['val_cw_f1'].append([])
            self.history['val_acc'].append([])
            self.history['val_cw_acc'].append([])
            self.history['iteration'].append(iteration)    
            self.history['labeled_set_size'].append(len(self.labeled_indices))
            self.history['unlabeled_set_size'].append(len(self.unlabeled_indices))
            self.history['batch_size'].append(batch_size)
            self.history['train_distribution'].append(self.get_class_distribution(self.config.labeled_classes,labels=labels))
            self.history['labeled_distribution'].append(self.get_class_distribution(self.config.unlabeled_classes, self.train_data.select(self.labeled_indices)))
            self.history['unlabeled_distribution'].append(self.get_class_distribution(self.config.unlabeled_classes, self.train_data.select(self.unlabeled_indices)))
            

            iter_pbar.write('\nBatch Size: %s;   Labeled Size: %s ; Unlabeled Size: %s'\
                            %(batch_size, len(self.labeled_indices), len(self.unlabeled_indices)))
            iter_pbar.write('\nClass Distribution')
            iter_pbar.write('Train Set     : '+str(self.history['train_distribution'][-1]))
            iter_pbar.write('Labeled Set   : '+str(self.history['labeled_distribution'][-1]))
            iter_pbar.write('Unlabeled Set : '+str(self.history['unlabeled_distribution'][-1])+'\n')

            self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
            optimizer = AdamW(self.model.parameters(), lr=5e-05)
            
            train_pbar.total = len(train_dataloader)
            train_pbar.refresh()
            eval_pbar.total = len(eval_dataloader)
            eval_pbar.refresh()
            iter_pbar.total += 1
            iter_pbar.refresh()

            if self.config.num_epochs * len(train_dataloader) < self.config.min_total_steps:
                num_epochs = self.config.min_total_steps//len(train_dataloader)
            else: num_epochs = self.config.num_epochs

            epoch_pbar.total  = num_epochs
            epoch_pbar.refresh()
            for epoch in range(num_epochs):
                tr_loss, tr_acc = self._train(self.model, train_dataloader, optimizer, train_pbar)
                ev_loss, ev_cw_f1, ev_acc, ev_cw_acc = self._eval(self.model, eval_dataloader, eval_pbar)


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

            self.selected_indices = []

            if len(self.unlabeled_indices):
                pred_data = STDataset(self.train_data.select(self.unlabeled_indices),self.unlabeled_indices)
                pred_dataloader = DataLoader(pred_data, batch_size=self.config.batch_size,collate_fn=data_collator,shuffle=False)
                pred_pbar.total = len(pred_dataloader)
                pred_pbar.refresh()

                predicted_labels, predicted_probs = self._predict(self.model, pred_dataloader, pred_pbar)


                self.history['predicted_probs'].append(predicted_probs.detach().cpu().numpy())
                self.history['predicted_prob_mean'].append(torch.mean(predicted_probs).item())
                self.history['predicted_prob_std'].append(torch.std(predicted_probs).item())
                self.history['predicted_prob_max'].append(torch.max(predicted_probs).item())
                self.history['predicted_prob_min'].append(torch.min(predicted_probs).item())
                self.history['predicted_distribution'].append(self.get_class_distribution(self.config.labeled_classes,labels=predicted_labels))


                selected_probs = []
                if self.config.strategy=='threshold':
                    for idx,pseudo_prob,pseudo_label in zip(self.unlabeled_indices,predicted_probs,predicted_labels):
                        if pseudo_prob > self.config.threshold:
                            self.labels[idx] = pseudo_label
                            self.selected_indices.append(idx)
                            selected_probs.append(pseudo_prob.detach().item())
                    selected_probs = torch.tensor(selected_probs)

                elif self.config.strategy=='top_k':
                    top_k = min(len(self.unlabeled_indices),self.config.top_k)
                    selected_probs = torch.topk(predicted_probs,top_k)
                    for pseudo_prob, idx in zip(selected_probs.values, selected_probs.indices):
                        self.labels[self.unlabeled_indices[idx]] = predicted_labels[idx]
                        self.selected_indices.append(self.unlabeled_indices[idx])
                    selected_probs = selected_probs.values.detach()
                
                elif self.config.strategy=='pc_top_k':
                    for class_ in self.config.labeled_classes:
                        class_indices = torch.where(predicted_labels==class_)[0]
                        top_k = min(len(class_indices),self.config.pc_top_k) 
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

                self.history['pseudo_distribution'].append(self.get_class_distribution(self.config.labeled_classes, labels=pseudo_labels))
                self.history['selected_distribution'].append(self.get_class_distribution(self.config.unlabeled_classes,self.train_data.select(self.selected_indices)))
                self.history['selected_set_size'].append(len(self.selected_indices))
                self.history['selected_in_domain'].append(self.get_class_count(self.history['selected_distribution'][-1],self.config.labeled_classes))
                self.history['selected_out_domain'].append(self.get_class_count(self.history['selected_distribution'][-1],
                    list(set(self.config.unlabeled_classes) - set(self.config.labeled_classes))))
    

            if len(self.unlabeled_indices) == last_length:
                patience_counter+=1
            last_length = len(self.unlabeled_indices)
            
            iter_pbar.write('\nSelected Size: %s ; Min Prob: %.3f ; Max Prob: %.3f ;'\
                            %(len(self.selected_indices),self.history['predicted_prob_min'][-1], self.history['predicted_prob_max'][-1]))
            
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
            pred_pbar.reset()
            
        iter_pbar.total -= 1
        iter_pbar.refresh()
        train_pbar.close()
        eval_pbar.close()
        epoch_pbar.close()
        pred_pbar.close()
        iter_pbar.close()
            
        if patience_counter == self.config.max_patience:
            print("\nEarly Stopping, Max Patience Over.\n")
        if len(self.unlabeled_indices) == 0:
            print("\nEmpty Unlabeled Set, Stopping.\n")

        if self.config.save_history:
            os.makedirs(self.config.history_dir,exist_ok=True)
            
            if self.config.strategy=='threshold':
                prefix = '%.4f' %(self.config.threshold)    
            elif self.config.strategy=='top_k':
                prefix = str(self.config.top_k)
            elif self.config.strategy=='pc_top_k':
                prefix = str(self.config.pc_top_k)
            else:
                raise NotImplementedError
                
            # threshold/top_k  labeled_classes  labeled_size  unlabeled_classes  unlabeled_size  history.pt 
            history_file = self.config.history_dir +'/'+ prefix +'_'+\
                str(len(self.config.labeled_classes))+'_'+str(self.labeled_size)+'_'+\
                str(len(self.config.unlabeled_classes)) +'_'+str(self.unlabeled_size)+'_history.pt'
            try:
                torch.save(self.history,history_file)
            except Exception as e:
                print('Exception Occured:\n%s'%(e))
                name = input("\nEnter file name to save: ",type=str)
                torch.save(self.history,name)
                
        return self.history


def parse_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-s','--strategy', type=str)
    arg_parser.add_argument('-th','--threshold', type=int)
    arg_parser.add_argument('-tk','--top_k', type=int)
    arg_parser.add_argument('-ptk','--pc_top_k', type=int)
    arg_parser.add_argument('-ls','--labeled_size', type=int)
    arg_parser.add_argument('-us','--unlabeled_size', type=int)
    arg_parser.add_argument('-lpc','--labeled_per_class', type=int)
    arg_parser.add_argument('-upc','--unlabeled_per_class', type=int)
    arg_parser.add_argument('-es','--eval_size', type=int)
    arg_parser.add_argument('-lc','--labeled_classes', type=int)
    arg_parser.add_argument('-uc','--unlabeled_classes', type=int)
    arg_parser.add_argument('-d','--dataset', type=str)
    arg_parser.add_argument('-eqs','--equal_split',action='store_true', default=False)
    arg_parser.add_argument('-hd','--history_dir', type=str)

    return arg_parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    
    # args.strategy = 'top_k'
    # args.top_k = 500
    # args.labeled_classes = 5
    # args.unlabeled_classes = 10
    # args.labeled_size = 70
    # args.unlabeled_size = 2100
    # args.eval_size = 500
    # args.dataset = 'yahoo_answers_10'
    # args.equal_split = False

    # args.strategy = 'pc_top_k'
    # args.pc_top_k = 100
    # args.labeled_classes = 5
    # args.unlabeled_classes = 10
    # args.labeled_per_class = 10
    # args.unlabeled_per_class = 200
    # args.eval_size = 500
    # args.dataset = 'yahoo_answers_10'
    # args.equal_split = False
    
    # args.history_dir = './test'

    data_dir = './data/' + args.dataset
    # history_dir = os.path.join('./history',args.dataset,args.strategy)

    
    
    trainer_config = SelfTrainConfig(
                                    strategy = args.strategy,
                                    threshold= args.threshold,
                                    top_k = args.top_k,
                                    pc_top_k= args.pc_top_k,
                                    train_data_path = data_dir+'/tokenized/train',
                                    eval_data_path = data_dir+'/tokenized/test',
                                    eval_size= args.eval_size,
                                    history_dir = args.history_dir,
                                    labeled_classes = [*range(args.labeled_classes)],
                                    unlabeled_classes = [*range(args.unlabeled_classes)],
                                    labeled_size = args.labeled_size,
                                    unlabeled_size = args.unlabeled_size,
                                    labeled_per_class=args.labeled_per_class,
                                    unlabeled_per_class=args.unlabeled_per_class,
                                    equal_split=args.equal_split
                                    )

    trainer = Trainer(trainer_config)
    trainer.self_train()