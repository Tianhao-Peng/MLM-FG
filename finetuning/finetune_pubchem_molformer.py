import argparse
import glob
import torch
import shutil
import rdkit
from torch import nn
import args
import os
import numpy as np
import random
import getpass
import sys
import traceback
import yaml
import csv
import copy
import json
import regex as re
import time
import pandas as pd
import pickle
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
from model.model_builder import SmilesEncoderBuilder as smiles_model_builder
from model.model_builder import TwoDimensionEncoderBuilder as twoD_model_builder
from model.model_builder import CrossAttentionEncoderBuilder as cross_attention_model_builder
from model.model_builder import RotateEncoderBuilder as rotate_builder
from model.model_builder import CrossAttentionLinearRotary
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
from fast_transformers.feature_maps import Favor,GeneralizedRandomFeatures
from datasets import load_dataset, concatenate_datasets, load_from_disk
# from pubchem_encoder import Encoder
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_warn, rank_zero_only, seed
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import LambdaLR
from collections import defaultdict, Counter

from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.masking import LengthMask as LM
from model.position_encoding import GraphPositionalEncoding, SmilesPositionalEncoding
import torch.nn.functional as F
from functools import partial
from apex import optimizers
from torch.optim import AdamW
from torch.utils.data.sampler import SubsetRandomSampler
from transformers import get_linear_schedule_with_warmup
from transformers import RobertaConfig, RobertaModel

from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, accuracy_score
from torch.utils.data import DataLoader
import subprocess
from model.cross_attention import CrossAttentionTail
from utils import get_filelist, save_scripts, append_to_file
from dataset_test import PropertyPredictionDataset, MolTestDataset, MolTranBertTokenizer
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from torch_scatter import scatter
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import rdmolfiles
from rdkit.Chem import rdmolops

from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')  

class LightningModule(pl.LightningModule):

    def __init__(self, config, vocab, graph_vocab, dataset_length, normalizer, trainer):
        super(LightningModule, self).__init__()

        self.save_hyperparameters(config)
        self.vocabulary = vocab
        self.normalizer = normalizer
        self.dataset_length = dataset_length
        #location of cache File
        # Special symbols

        self.debug = config.debug
        # self.text_encoder = Encoder(config.max_len, config.modal_type)
        # Word embeddings layer
        n_vocab, d_emb = len(vocab.keys()), config.n_embd
        if config.model_arch == 'SMILESonly':
            smiles_builder = rotate_builder.from_kwargs(
                n_layers=config.n_layer,
                n_heads=config.n_head,
                query_dimensions=config.n_embd//config.n_head,
                value_dimensions=config.n_embd//config.n_head,
                feed_forward_dimensions=config.n_embd,
                attention_type='linear',
                feature_map=partial(GeneralizedRandomFeatures, n_dims=config.num_feats),
                activation='gelu',
                )
            self.blocks = smiles_builder.get()
        if config.model_arch == 'RoBERTa':
            robertaConfig = RobertaConfig(
                vocab_size=len(self.vocabulary),  # RoBERTa base 使用的词汇量大小
                max_position_embeddings=1000,
                num_attention_heads=config.n_head,
                num_hidden_layers=config.n_layer,
                type_vocab_size=1,
            )
            self.blocks = RobertaModel(robertaConfig)
        else:
            raise ValueError('Error config.model_arch:[{model_arch}]'.format(config.model_arch))
        self.smiles_pos_encoding = SmilesPositionalEncoding(d_model=config.n_embd//config.n_head, n_heads=config.n_head)
        self.graph_pos_encoding = GraphPositionalEncoding(n_heads=config.n_head, 
                                                          d_model=config.n_embd//config.n_head, pos_type=config.pos_type, device=torch.cuda.current_device()) #laplacian |mpnn
            
        # input embedding stem
        self.pos_emb = None
        self.smiles_tok_emb = nn.Embedding(n_vocab+10, config.n_embd)
        n_graph_vocab = len(graph_vocab.keys())  # 123
        self.graph_tok_emb = nn.Embedding(n_graph_vocab+10, config.n_embd)
        if torch.any(torch.isnan(self.graph_tok_emb.weight)):
            print("NaNs in embedding weights before forward pass")

            
        self.drop = nn.Dropout(config.d_dropout)
        self.train_config = config
        
        if config.task == 'classification':
            self.pred_head = self.Net(
                config.n_embd, 1, dropout=config.dropout,
                task=config.task,
            )
            self.criterion = nn.BCEWithLogitsLoss() #nn.CrossEntropyLoss()# nn.CrossEntropyLoss() | nn.BCEWithLogitsLoss(reduction='none')
            self.measure_name = 'AUC'
        elif config.task == 'regression':
            self.pred_head = self.Net(
                config.n_embd, 1, dropout=config.dropout,
                # config.n_embd*2, 1, dropout=config.dropout,
                task=config.task,
            )
            if self.train_config.dataset_name in ['qm7', 'qm8', 'qm9']:
                # self.criterion = nn.MSELoss()
                self.criterion = nn.L1Loss()
                # self.criterion = nn.SmoothL1Loss()
                self.measure_name = 'MAE'
            else:
                self.criterion = nn.MSELoss()
                # self.criterion = nn.SmoothL1Loss()
                self.measure_name = 'RMSE'
        else:
            raise ValueError('Error config.task:{}'.format(config['task']))

        #if we are starting from scratch set seeds
        seed.seed_everything(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        self.trainer = trainer
        if trainer.global_rank == 0:
            log_path = trainer.logger.log_dir
        else:
            log_path_raw = trainer.logger.log_dir.split('_')
            prefix = "_".join(log_path_raw[:-1])
            num = int(log_path_raw[-1]) - 1
            log_path = prefix + "_" + str(num)
        self.prediction_path = os.path.join(log_path, "prediction_{}_underBestValLoss.pkl".format(self.train_config.target))
        self.log_path = log_path
        self.tb_validation_path = os.path.join(log_path, "results_validation_{}.csv".format(trainer.global_rank))
        with open(self.tb_validation_path, 'w') as f:
            f.write(
                f"{self.train_config.dataset_name}, current_epoch,"
                + f"{'validation_loss'},"
                + f"{'validation_' + self.measure_name},"
                + f"{'test_loss'},"
                + f"{'test_' + self.measure_name},"
                + f"{'min_epoch'},"
                + f"{'min_validation_loss'},"
                + f"{'min_test_loss'},"
                + f"{'best_test_' + self.measure_name}"
                + "\n"
            )
        self.tb_training_path = os.path.join(log_path, "results_training_{}.csv".format(trainer.global_rank))
        with open(self.tb_training_path, 'w') as f:
            f.write(
                f"{self.train_config.dataset_name}, current_epoch,"
                + f"training_epoch_loss," + f"training_{self.measure_name}"
                + "\n"
            )
        
        self.min_loss = {
            "min_validation_loss": torch.finfo(torch.float32).max,
            "min_validation_loss_measure": torch.finfo(torch.float32).max,
            "min_epoch": 0,
            "max_test_cls_his": torch.finfo(torch.float32).min,
            "min_test_reg_his": torch.finfo(torch.float32).max
        }

    class Net(nn.Module):
        dims = [150, 50, 50, 2]

        
        def __init__(self, smiles_embed_dim, num_classes, dims=dims, dropout=0.2, task='regression'):
            super().__init__()
            self.desc_skip_connection = True 
            self.fcs = []  # nn.ModuleList()
            print('dropout is {}'.format(dropout))

            self.fc1 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
            self.dropout1 = nn.Dropout(dropout)
            self.relu1 = nn.GELU()
            self.fc2 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
            self.dropout2 = nn.Dropout(dropout)
            self.relu2 = nn.GELU()
            self.final = nn.Linear(smiles_embed_dim, num_classes) #classif
            self.task = task

        def forward(self, smiles_emb):
            x_out = self.fc1(smiles_emb)
            x_out = self.dropout1(x_out)
            x_out = self.relu1(x_out)

            if self.desc_skip_connection is True:
                x_out = x_out + smiles_emb

            z = self.fc2(x_out)
            z = self.dropout2(z)
            z = self.relu2(z)
            if self.desc_skip_connection is True:
                z = self.final(z + x_out)
            else:
                z = self.final(z)

            #z = self.layers(smiles_emb)
            print('Net z', z)
            if self.final.out_features == 1 and self.task=='classification':
                return z
            elif self.final.out_features == 2 and self.task=='classification':
                return F.softmax(z, dim=-1)
            else:
                return z
        
    class Net2(nn.Module):
        def __init__(self, smiles_embed_dim, num_classes, dropout=0.2):
            super().__init__()

            self.dropout = nn.Dropout(dropout)
            self.final = nn.Linear(smiles_embed_dim, num_classes) #classif
            
        def forward(self, smiles_emb):
            z = self.dropout(smiles_emb)
            z = self.final(z)

            return z
        
    def on_save_checkpoint(self, checkpoint):
        #save RNG states each time the model and states are saved
        out_dict = dict()
        out_dict['torch_state']=torch.get_rng_state()
        out_dict['cuda_state']=torch.cuda.get_rng_state()
        if np:
            out_dict['numpy_state']=np.random.get_state()
        if random:
            out_dict['python_state']=random.getstate()
        checkpoint['rng'] = out_dict

    def on_load_checkpoint(self, checkpoint):
        #load RNG states each time the model and states are loaded from checkpoint
        rng = checkpoint['rng']
        for key, value in rng.items():
            if key =='torch_state':
                torch.set_rng_state(value)
            elif key =='cuda_state':
                torch.cuda.set_rng_state(value)
            elif key =='numpy_state':
                np.random.set_state(value)
            elif key =='python_state':
                random.setstate(value)
            else:
                print('unrecognized state')

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm1d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)


        if self.pos_emb != None:
            no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.0},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        if self._hparams.measure_name == 'r2':
            betas = (0.9, 0.999)
        else:
            betas = (0.9, 0.99)
        print('betas are {}'.format(betas))
        learning_rate = self.train_config.lr_start * self.train_config.lr_multiplier
        optimizer = optimizers.FusedLAMB(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def one_step(self, batch, batch_idx, flag=None):
        # forward the model
        smiles_idx, graph_idx, graph_adj, graph_pos, hyper_adj, edge_index, edge_attr, labels, smiles_mask, graph_mask = batch
        smiles_pos = None
        
        smiles_emb = self.smiles_tok_emb(smiles_idx) # each index maps to a (learnable) vector
        graph_emb =  self.graph_tok_emb(graph_idx)

        # forward
        if config.model_arch in ['SMILESonly']:
            smiles_emb = self.drop(smiles_emb)
            smiles_output = self.blocks(smiles_emb, length_mask=LM(smiles_mask.sum(-1)))  # Shape: (batch_size, seq_len, head_num, hidden_dim)
            token_embeddings = smiles_output
            mask = smiles_mask
        if config.model_arch in ['RoBERTa']:
            smiles_emb = self.drop(smiles_emb)
            smiles_output = self.blocks(inputs_embeds=smiles_emb, attention_mask=smiles_mask)['last_hidden_state']
            token_embeddings = smiles_output
            mask = smiles_mask
            
        output = self.output_func(token_embeddings, mask, flag)
        pred = self.pred_head(output)
        
        return pred, labels
    
    def output_func(self, token_embeddings, mask, flag=None):
        input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        output = sum_embeddings / sum_mask
        return output
    
    def calculate_loss(self, pred, labels):
        if self.train_config.task == 'classification':
            if self.pred_head.final.out_features == 1:
                print('calculate_loss pred.flatten()', pred.flatten().shape, pred.flatten())
                print('calculate_loss labels.float()', labels.shape, labels)
                loss = self.criterion(pred.flatten(), labels.float())
            else:
                loss = self.criterion(pred, labels.flatten())
            
        elif self.train_config.task == 'regression':
            if self.normalizer:
                loss = self.criterion(pred.flatten(), self.normalizer.norm(labels))
            else:
                loss = self.criterion(pred.flatten(), labels)
        else:
            raise ValueError('Error self.train_config.task:{}'.format(self.train_config.task))
        
        return loss
    
    def get_final_pred(self, pred):
        if self.normalizer:
            pred = self.normalizer.denorm(pred)

        if self.train_config.task == 'classification':
            if self.pred_head.final.out_features == 1:
                pred = F.sigmoid(pred) 
            else:
                pred = F.softmax(pred, dim=-1)
        
        return pred
    
    def evaluate_epoch_end(self, outputs):
        
        def cal_accuracy_score(labels, x):
            exp_x = np.exp(x - np.max(x)) 
            y_pred_probs = exp_x / exp_x.sum(axis=1, keepdims=True)
            y_pred = np.argmax(y_pred_probs, axis=1)
            acc = accuracy_score(labels, y_pred)
            return acc
        
        def cal_accuracy_score_sigmoid(labels, x):
            y_pred = []
            for pred in x:
                if pred >= 0.5:
                    y_pred.append(1)
                else:
                    y_pred.append(0)
            acc = accuracy_score(labels, y_pred)
            return acc
        
        predictions, labels, losses = [], [], []
        for output in outputs:
            predictions.extend(output['predictions'].cpu().numpy())
            labels.extend(output['labels'].cpu().numpy())
            losses.append(output['loss'])
        print('evaluate_epoch_end losses', losses)
        avg_loss = torch.tensor(losses).mean().item()
        
        if self.train_config.task == 'regression':
            if self.train_config.dataset_name in ['qm7', 'qm8', 'qm9']:
                mae = mean_absolute_error(labels, predictions)
                measure_result = mae
            else:
                rmse = mean_squared_error(labels, predictions, squared=False)
                measure_result = rmse
        elif self.train_config.task == 'classification': 
            predictions = np.array(predictions)
            if self.pred_head.final.out_features == 1:
                roc_auc = roc_auc_score(labels, predictions)
                acc = cal_accuracy_score_sigmoid(labels, predictions)
            else:
                roc_auc = roc_auc_score(labels, predictions[:,1])
                acc = cal_accuracy_score(labels, predictions)
            measure_result = [roc_auc, acc]
        
        return avg_loss, measure_result
        
    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()  # free unused memory
        
        pred, labels = self.one_step(batch, batch_idx, flag='training_one_step')
        loss = self.calculate_loss(pred, labels)
        pred = self.get_final_pred(pred)

        return {'loss':loss, 'predictions':pred.detach(), 'labels':labels.detach()}
    
    def training_epoch_end(self, outputs):
        training_epoch_loss, training_measure_result_all = self.evaluate_epoch_end(outputs)
        if self.train_config.task == 'classification':
            training_measure_result, acc = training_measure_result_all
        else:
            training_measure_result, acc = training_measure_result_all, 0
            
        self.log('training_loss', training_epoch_loss)
        self.log('training_{}'.format(self.measure_name), training_measure_result)
        self.log('training_{}'.format('acc'), acc)
        
        print('\ntraining ' + f"{self.train_config.dataset_name}, {self.train_config.target}, {self.current_epoch},"
              + f"training_epoch_loss:{training_epoch_loss}, training_measure_result:{training_measure_result}, acc:{acc}")
        append_to_file(self.tb_training_path,
            f"{self.train_config.dataset_name}, {self.current_epoch},"
            + f"{training_epoch_loss}, {training_measure_result}, {acc}",
        )
        
    def validation_step(self, batch, batch_idx, dataset_idx):
        torch.cuda.empty_cache()  # free unused memory
        
        pred, labels = self.one_step(batch, batch_idx, flag='validation_one_step')
        
        print('validation_step pred', pred)
        print('validation_step labels', labels)
        
        loss = self.calculate_loss(pred, labels)
        pred = self.get_final_pred(pred)
        
        return {'loss':loss, 'predictions':pred.detach(), 'labels':labels.detach(), 'dataset_idx':dataset_idx}

    def validation_epoch_end(self, outputs_all):
        tensorboard_logs = {}
        split_name_list = ['validation', 'test']
            
        for dataset_idx, outputs in enumerate(outputs_all):
            split_name = split_name_list[dataset_idx]
            avg_loss, measure_result_all = self.evaluate_epoch_end(outputs)
            if self.train_config.task == 'classification':
                measure_result, acc = measure_result_all
            else:
                measure_result, acc = measure_result_all, 0

            tensorboard_logs.update({split_name + "_loss": avg_loss,
                                     split_name + "_" + self.measure_name: measure_result,
                                     split_name + "_" + 'loss_measure': avg_loss-measure_result,
                                     split_name + "_" + 'acc': acc},
                                   )
        if self.measure_name == 'AUC':
            self.min_loss['max_test_cls_his'] = max(self.min_loss['max_test_cls_his'], tensorboard_logs['test_' + self.measure_name])
        else:
            self.min_loss['min_test_reg_his'] = min(self.min_loss['min_test_reg_his'], tensorboard_logs['test_' + self.measure_name])
        
        with open(os.path.join(self.log_path, 'prediction_{}'.format(self.current_epoch)), 'wb') as f:
            pickle.dump(outputs_all, f)
                
        if (tensorboard_logs["validation_loss"] < self.min_loss["min_validation_loss"]):
            with open(self.prediction_path, 'wb') as f:
                pickle.dump(outputs_all, f)
            self.min_loss["min_validation_loss_measure"] = tensorboard_logs["validation_loss_measure"]
            # if self.current_epoch>(self.train_config.unfreeze):
            #     self.min_loss["min_validation_loss"] = tensorboard_logs["validation_loss"]
            self.min_loss["min_validation_loss"] = tensorboard_logs["validation_loss"]
            self.min_loss["min_test_loss"] = tensorboard_logs["test_loss"]
            self.min_loss["min_epoch"] = self.current_epoch
            self.min_loss['best_test_' + self.measure_name] = tensorboard_logs['test_' + self.measure_name]
            self.min_loss['best_test_acc'] = tensorboard_logs['test_acc']

        self.logger.log_metrics(tensorboard_logs, self.global_step)

        for k in tensorboard_logs.keys():
            self.log(k, tensorboard_logs[k])
        for k in self.min_loss.keys():
            self.log(k, self.min_loss[k])

        print("\nValidation: Current Epoch", self.current_epoch, tensorboard_logs, '\n', self.min_loss)
        append_to_file(self.tb_validation_path,
            f"{self.train_config.dataset_name}, {self.current_epoch},"
            + f"{tensorboard_logs['validation_loss']},"
            + f"{tensorboard_logs['validation_' + self.measure_name]},"
            + f"{tensorboard_logs['validation_acc']},"
            + f"{tensorboard_logs['test_loss']},"
            + f"test_measure:{tensorboard_logs['test_' + self.measure_name]},"
            + f"test_acc:{tensorboard_logs['test_acc']},"
            + f"{self.min_loss['min_epoch']},"
            + f"{self.min_loss['min_validation_loss']},"
            + f"{self.min_loss['min_test_loss']},"
            + f"{self.min_loss['best_test_' + self.measure_name]},"
            + f"best_test_acc:{self.min_loss['best_test_acc']}",
            )
            # return {'validation_loss':avg_loss, 'mae':mae, 'rmse':rmse, 'roc_auc':roc_auc}
        
        if self.current_epoch == self.train_config.unfreeze:
            for param in self.blocks.parameters():
                param.requires_grad = True
            for name, param in self.blocks.named_parameters():
                print('frozen param:{} | {}'.format(name, param.requires_grad))
            
    def test_step(self, batch, batch_idx):
        torch.cuda.empty_cache()  # free unused memory
        
        pred, labels = self.one_step(batch, batch_idx)
        loss = self.calculate_loss(pred, labels)
        pred = self.get_final_pred(pred)

        return {'loss':loss, 'predictions':pred.detach(), 'labels':labels.detach()}

    def test_epoch_end(self, outputs):
        avg_loss, measure_result_all = self.evaluate_epoch_end(outputs)
        
        if self.train_config.task == 'classification':
            measure_result, acc = measure_result_all
        else:
            measure_result, acc = measure_result_all, 0
        self.log('test_loss', avg_loss)
        self.log(f'{self.measure_name}', measure_result)
        self.log('acc', acc)
        
        print('test' + f"{self.train_config.dataset_name}, {self.current_epoch},"
              + f"avg_loss:{avg_loss}, measure_result:{measure_result}, acc:{acc}")

def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold

def generate_scaffolds(dataset, balanced, val_size, test_size, log_every_n=1000, seed=None):
    scaffolds = {}
    data_len = len(dataset)
    print(data_len)

    print("About to generate scaffolds")
    for ind, smiles in enumerate(dataset):
        if ind % log_every_n == 0:
            print("Generating scaffold %d/%d" % (ind, data_len))
        scaffold = _generate_scaffold(smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)
            
    if balanced == 'True':  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
        index_sets = list(scaffolds.values())
        big_index_sets = []
        small_index_sets = []
        val_size = val_size*len(dataset)
        test_size = test_size*len(dataset)
        for index_set in index_sets:
            if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(seed)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        scaffold_sets = big_index_sets + small_index_sets
    elif balanced == 'False':  # Sort from largest to smallest scaffold sets
        # Sort from largest to smallest scaffold sets
        scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
        scaffold_sets = [
            scaffold_set for (scaffold, scaffold_set) in sorted(
                scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
        ]
    elif balanced == 'random':
        index_sets = list(scaffolds.values())
        big_index_sets = []
        small_index_sets = []
        val_size = val_size*len(dataset)
        test_size = test_size*len(dataset)
        for index_set in index_sets:
            if len(index_set) > val_size  or len(index_set) > test_size :
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        scaffold_sets = big_index_sets + small_index_sets
        
    return scaffold_sets


def scaffold_split(dataset, valid_size, test_size, balanced, seed=None, log_every_n=1000):
    random.seed(seed)
    
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(dataset, balanced=balanced, val_size=valid_size, test_size=test_size, seed=seed)

    train_cutoff = train_size * len(dataset)
    valid_cutoff = (train_size + valid_size) * len(dataset)
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []
    
    for scaffold_set in scaffold_sets:
        # print('len(valid_inds) + len(scaffold_set)', len(valid_inds) + len(scaffold_set))
        if len(train_inds) + len(scaffold_set) <= train_cutoff:
            train_inds += scaffold_set
        elif len(train_inds) + len(valid_inds) + len(scaffold_set) <= valid_cutoff:
            valid_inds += scaffold_set
        else:
            test_inds += scaffold_set

    return train_inds, valid_inds, test_inds

def load_npz_to_data_list(npz_file):
    """
    Reload the data list save by ``save_data_list_to_npz``.

    Args:
        npz_file(str): the npz file location.

    Returns:
        a list of data where each data is a dict of numpy ndarray.
    """
    def _split_data(values, seq_lens, singular):
        res = []
        s = 0
        for l in seq_lens:
            if singular == 0:
                res.append(values[s: s + l])
            else:
                res.append(values[s])
            s += l
        return res

    merged_data = np.load(npz_file, allow_pickle=True)
    names = [name for name in merged_data.keys() 
            if not name.endswith('.seq_len') and not name.endswith('.singular')]
    data_dict = {}
    for name in names:
        data_dict[name] = _split_data(
                merged_data[name], 
                merged_data[name + '.seq_len'],
                merged_data[name + '.singular'])

    data_list = []
    n = len(data_dict[names[0]])
    for i in range(n):
        data = {name:data_dict[name][i] for name in names}
        data_list.append(data)
    return data_list

def read_smiles(data_path, target, task):
    # chem_data = load_npz_to_data_list(os.path.join(data_path, 'chem_data.npz')) 

    smiles_data, labels = [], []
    with open(data_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            smiles = row['smiles']
            label = row[target]
            mol = Chem.MolFromSmiles(smiles)
            if mol != None and label != '':
                normalized_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
                smiles_data.append(normalized_smiles)
                # smiles_data.append(smiles)
                if task == 'classification':
                    labels.append(int(label))
                elif task == 'regression':
                    labels.append(float(label))
                else:
                    ValueError('task must be either regression or classification')
            else:
                print('Invalid smiles:{}, label:{}'.format(smiles, label))
    # print(len(smiles_data))
    return smiles_data, labels

class MoleculeModule(pl.LightningDataModule):
    def __init__(self, 
        batch_size, num_workers, valid_size, test_size, 
        data_path, target, task, splitting, results_path, graph_padding, train_load
    ):
        super(MoleculeModule, self).__init__()
        self.results_path = results_path
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.target = target
        self.task = task
        self.splitting = splitting
        assert splitting in ['random', 'scaffold', 'scaffold_balance', 'molformer', 'scaffold_random']
        
        vocab_path = train_load
        with open(os.path.join('../data/pretrain/', vocab_path, 
                               'extend-{}-smiles-vocab-full.json'.format(vocab_path)), 'r') as f:
            self.vocab_encoder = json.load(f)
        self.char2id = {"<bos>":0, "<eos>":1, "<pad>":2, "<mask>":3}
        self.id2char = {0:"<bos>", 1:"<eos>", 2:"<pad>", 3:"<mask>"}
        self.pad  = self.char2id['<pad>']
        self.mask = self.char2id['<mask>']
        self.eos  = self.char2id['<eos>']
        self.bos  = self.char2id['<bos>']
        
        self.pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        self.regex = re.compile(self.pattern)
        self.graph_padding = graph_padding
        
        pos = 0
        for key, value in self.vocab_encoder.items():
            self.char2id[key] = pos+4
            self.id2char[pos+4] = key
            pos += 1
        self.char2id["<unk>"] = pos + 4
        self.id2char[pos+4] = "<unk>"
        pos += 1
        self.char2id["<cls>"] = pos + 4
        self.id2char[pos+4] = "<cls>"
        print('self.char2id', self.char2id)
        
        self.graph_char2id = {"<bos>":0, "<eos>":1, "<pad>":2, "<mask>":3}
        self.graph_pad  = self.graph_char2id['<pad>']
        self.graph_mask = self.graph_char2id['<mask>']
        self.graph_eos  = self.graph_char2id['<eos>']
        self.graph_bos  = self.graph_char2id['<bos>']
        with open('./periodic_table_of_elements.json', 'r') as f:
            element_to_index = json.load(f) # 'H': 1, ...
            for key, value in element_to_index.items():
                self.graph_char2id[key] = value + 3
        self.graph_char2id["<unk>"] = len(self.graph_char2id)
        self.graph_char2id["<cls>"] = len(self.graph_char2id)
        print('self.graph_char2id', self.graph_char2id)

    def read_file(self, filename):
        with open(os.path.join(self.data_path, filename), 'r') as f:
            df = pd.read_csv(f)
            smiles, labels = df['smiles'].tolist(), df[self.target].tolist()
        normalized_smiles, normalized_labels = [], []
        for s, l in zip(smiles, labels):
            mol = Chem.MolFromSmiles(s)
            if mol == None:
                continue
            nor_s = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
            mol = Chem.MolFromSmiles(nor_s)
            if mol == None:
                continue
            normalized_smiles.append(nor_s)
            normalized_labels.append(l)
        return normalized_smiles, normalized_labels
    
    def aug_data(self, smiles_list, label_list):
        print('aug training data\n'*10)
        def generate_smiles(mol, num=3):
            smiles_list = []
            num_atoms = mol.GetNumAtoms()
            # If num is not provided or is too large, set it to num_atoms
            num = min(num_atoms, num) if num else num_atoms

            for i in range(num):
                try:
                    new_smiles = rdmolfiles.MolToSmiles(mol, rootedAtAtom=i, canonical=False, doRandom=True)
                    new_mol = Chem.MolFromSmiles(new_smiles)
                    if (new_smiles not in smiles_list) and (new_mol != None):
                        smiles_list.append(new_smiles)
                except:
                    continue
            return smiles_list

        aug_smiles_list = []
        aug_label_list = []
        for i in range(len(smiles_list)):
            smiles = smiles_list[i]
            label = label_list[i]
            mol = Chem.MolFromSmiles(smiles)
            if mol == None:
                continue
            aug_smiles_list.append(smiles)
            aug_label_list.append(label)
            smiles_cand = generate_smiles(mol)
            for one_smiles_cand in smiles_cand:
                aug_smiles_list.append(one_smiles_cand)
                aug_label_list.append(label)
        return aug_smiles_list, aug_label_list
            
    def prepare_data(self):
        if self.splitting == 'molformer':
            self.prepare_data_manual()
        else:
            self.prepare_data_auto_split()
            # append_to_file(self.results_path, 
            #                f"{self.data_path}, {self.target}, {self.task}, {self.splitting}, prepare_data_auto_split")
    
    def prepare_data_manual(self):
        print('prepare_data_from_molformer\n'*10)
        train_smiles, train_labels = read_smiles(os.path.join(self.data_path, 'train.csv'), self.target, self.task)
        if config.aug_training_dataset == 'True':
            train_smiles, train_labels = self.aug_data(train_smiles, train_labels)
            print('AUG: train_smiles:{}, positive ratio:{}'.format(len(train_labels), Counter(train_labels)))
        else:
            print('Not aug training dataset')
        valid_smiles, valid_labels = read_smiles(os.path.join(self.data_path, 'valid.csv'), self.target, self.task)
        test_smiles,  test_labels  = read_smiles(os.path.join(self.data_path, 'test.csv'), self.target, self.task)
        valid_smiles = valid_smiles + test_smiles
        valid_labels = valid_labels + test_labels
        
        train_dataset = MolTestDataset(dataset=train_smiles, labels=train_labels, data_path=self.data_path, 
                                       target=self.target, char2id=self.char2id, graph_char2id=self.graph_char2id, regex=self.regex, 
                                       graph_padding=self.graph_padding)
        valid_dataset = MolTestDataset(dataset=valid_smiles, labels=valid_labels, data_path=self.data_path, 
                                       target=self.target, char2id=self.char2id, graph_char2id=self.graph_char2id, regex=self.regex, 
                                       graph_padding=self.graph_padding)
        test_dataset = MolTestDataset(dataset=test_smiles, labels=test_labels, data_path=self.data_path, 
                                       target=self.target, char2id=self.char2id, graph_char2id=self.graph_char2id, regex=self.regex, 
                                       graph_padding=self.graph_padding)
        
        print('train_smiles:{}, positive ratio:{}'.format(len(train_labels), Counter(train_labels)))
        print('valid_smiles:{}, positive ratio:{}'.format(len(valid_labels), Counter(valid_labels)))
        print('test_smiles:{}, positive ratio:{}'.format(len(test_labels), Counter(test_labels)))
        
        if config.balance_training_dataset == 'True':
            train_smiles,train_labels = self.balance_train_dataset(train_smiles, train_labels)
            print('Balance: train_smiles:{}, positive ratio:{}'.format(len(train_labels), Counter(train_labels)))
        else:
            print('Not balance training dataset')
        
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, pin_memory=True,
            num_workers=self.num_workers, drop_last=False, collate_fn=train_dataset.my_collate_fn, shuffle=True
        )
        self.valid_loader = DataLoader(
            valid_dataset, batch_size=self.batch_size, pin_memory=True,
            num_workers=self.num_workers, drop_last=False, collate_fn=valid_dataset.my_collate_fn, shuffle=False
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, pin_memory=True,
            num_workers=self.num_workers, drop_last=False, collate_fn=test_dataset.my_collate_fn, shuffle=False
        )
        
    def prepare_data_auto_split(self):
        self.smiles_data, self.labels = read_smiles(data_path=self.data_path, target=self.target, task=self.task)
        print('prepare_data_auto_split\n'*10)
        if self.splitting == 'random':
            # obtain training indices that will be used for validation
            num_train = len(self.smiles_data)
            indices = list(range(num_train))
            np.random.shuffle(indices)

            split = int(np.floor(self.valid_size * num_train))
            split2 = int(np.floor(self.test_size * num_train))
            valid_idx, test_idx, train_idx = indices[:split], indices[split:split+split2], indices[split+split2:]
        
        elif self.splitting == 'scaffold':
            train_idx, valid_idx, test_idx = scaffold_split(self.smiles_data, self.valid_size, self.test_size, seed=config.seed, balanced='False')
        elif self.splitting == 'scaffold_balance':
            train_idx, valid_idx, test_idx = scaffold_split(self.smiles_data, self.valid_size, self.test_size, seed=config.seed, balanced='True')
        elif self.splitting == 'scaffold_random':
            train_idx, valid_idx, test_idx = scaffold_split(self.smiles_data, self.valid_size, self.test_size, seed=config.seed, balanced='random')

        # define samplers for obtaining training and validation batches
        # train_sampler = SubsetRandomSampler(train_idx)
        # valid_sampler = SubsetRandomSampler(valid_idx)
        # test_sampler = SubsetRandomSampler(test_idx)
        
        train_smiles, train_labels = [self.smiles_data[idx] for idx in train_idx], [self.labels[idx] for idx in train_idx]
        
        if config.aug_training_dataset == 'True':
            train_smiles, train_labels = self.aug_data(train_smiles, train_labels)
            print('AUG: train_smiles:{}, positive ratio:{}'.format(len(train_labels), Counter(train_labels)))
        else:
            print('Not aug training dataset')
            
        valid_smiles, valid_labels = [self.smiles_data[idx] for idx in valid_idx], [self.labels[idx] for idx in valid_idx]
        test_smiles, test_labels = [self.smiles_data[idx] for idx in test_idx], [self.labels[idx] for idx in test_idx]
        
        print('train_smiles:{}, positive ratio:{}'.format(len(train_labels), Counter(train_labels)))
        print('valid_smiles:{}, positive ratio:{}'.format(len(valid_labels), Counter(valid_labels)))
        print('test_smiles:{}, positive ratio:{}'.format(len(test_labels), Counter(test_labels)))
        
        if config.balance_training_dataset == 'True':
            train_smiles,train_labels = self.balance_train_dataset(train_smiles, train_labels)
            print('Balance: train_smiles:{}, positive ratio:{}'.format(len(train_labels), Counter(train_labels)))
        else:
            print('Not balance training dataset')
        
        train_dataset = MolTestDataset(dataset=train_smiles, labels=train_labels, data_path=self.data_path, 
                                       target=self.target, char2id=self.char2id, graph_char2id=self.graph_char2id, regex=self.regex, 
                                       graph_padding=self.graph_padding)
        valid_dataset = MolTestDataset(dataset=valid_smiles, labels=valid_labels, data_path=self.data_path, 
                                       target=self.target, char2id=self.char2id, graph_char2id=self.graph_char2id, regex=self.regex, 
                                       graph_padding=self.graph_padding)
        test_dataset = MolTestDataset(dataset=test_smiles, labels=test_labels, data_path=self.data_path, 
                                       target=self.target, char2id=self.char2id, graph_char2id=self.graph_char2id, regex=self.regex, 
                                       graph_padding=self.graph_padding)
#         train_dataset = PropertyPredictionDataset(dataset=train_smiles, labels=train_labels, data_path=self.data_path, target=self.target, tokenizer=self.tokenizer)
#         valid_dataset = PropertyPredictionDataset(dataset=valid_smiles, labels=valid_labels, data_path=self.data_path, target=self.target, tokenizer=self.tokenizer)
#         test_dataset = PropertyPredictionDataset(dataset=test_smiles, labels=test_labels, data_path=self.data_path, target=self.target, tokenizer=self.tokenizer)
        
        # print('train_dataset:{}, valid_dataset:{}, test_dataset:{}'.format(
        #                                 len(train_dataset), len(valid_dataset), len(test_dataset)))

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, pin_memory=True,
            num_workers=self.num_workers, drop_last=False, collate_fn=train_dataset.my_collate_fn, shuffle=True
        )
        self.valid_loader = DataLoader(
            valid_dataset, batch_size=self.batch_size, pin_memory=True,
            num_workers=self.num_workers, drop_last=False, collate_fn=valid_dataset.my_collate_fn, shuffle=False
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, pin_memory=True,
            num_workers=self.num_workers, drop_last=False, collate_fn=test_dataset.my_collate_fn, shuffle=False
        )
        
    def balance_train_dataset(self, X, y):
        y = np.array(y)
        X = np.array(X)
        minority_class_indices = np.where(y == 1)[0]

        majority_class_count = sum(y == 0)
        minority_class_count = sum(y == 1)
        num_samples_to_add = majority_class_count - minority_class_count

        random_samples_to_add = np.random.choice(minority_class_indices, num_samples_to_add)

        X_resampled = np.hstack([X, X[random_samples_to_add]])
        y_resampled = np.hstack([y, y[random_samples_to_add]])

        return X_resampled, y_resampled

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return [self.valid_loader, self.test_loader]
        # return [self.valid_loader, self.test_loader]

    # def test_dataloader(self):
    #     return self.test_loader
    
    def get_test_dataloader(self):
        return self.test_loader
    
    def get_vocab(self):
        #using home made tokenizer, should look into existing tokenizer
        return self.char2id
    
    def get_graph_vocab(self):
        return self.graph_char2id
    
class CheckpointEveryNSteps(pl.Callback):
    """
        Save a checkpoint every N steps, instead of Lightning's default that checkpoints
        based on validation loss.
    """

    def __init__(self, save_step_frequency=-1,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
        ):
        """
        Args:
        save_step_frequency: how often to save in steps
        prefix: add a prefix to the name, only used if
        use_modelcheckpoint_filename=False
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename
    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        return
            
class ModelCheckpointAtEpochEnd(pl.Callback):
    
    def __init__(self, every_n_epochs, model, test_dataloader):
        self.every_n_epochs = every_n_epochs
        self.model = model
        self.test_dataloader = test_dataloader
        self.last_epoch = 0
    
    def on_epoch_end(self, trainer, pl_module):
        return 
    


def get_nccl_socket_ifname():
    ipa = subprocess.run(['ip', 'a'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = ipa.stdout.decode('utf-8').split('\n')
    all_names = []
    name = None
    for line in lines:
        if line and not line[0] == ' ':
            name = line.split(':')[1].strip()
            continue
        if 'link/infiniband' in line:
            all_names.append(name)
    os.environ['NCCL_SOCKET_IFNAME'] = ','.join(all_names)

def fix_infiniband():
    # os.environ['NCCL_SOCKET_IFNAME'] = "^lo,docker,virbr,vmnet,vboxnet,wl,ww,ppp,bond"

    # ifname = os.environ.get('NCCL_SOCKET_IFNAME', None)
    # if ifname is None:
    #     os.environ['NCCL_SOCKET_IFNAME'] = '^lo,docker0'

    get_nccl_socket_ifname()
    os.environ['NCCL_IB_CUDA_SUPPORT'] = '1'
    ibv = subprocess.run('ibv_devinfo', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = ibv.stdout.decode('utf-8').split('\n')
    exclude = ''
    for line in lines:
        if 'hca_id:' in line:
            name = line.split(':')[1].strip()
        if '\tport:' in line:
            port = line.split(':')[1].strip()
        if 'link_layer:' in line and 'Ethernet' in line:
            exclude = exclude + f'{name}:{port},'
    if exclude:
        exclude = '^' + exclude[:-1]
        # print(exclude)
        os.environ['NCCL_IB_HCA'] = exclude

class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']
        
def load_model_weights(model, checkpoint_path):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    model_dict = model.state_dict()

    # Filter out unnecessary keys and get mismatched keys
    pretrained_dict = {}
    mismatched_keys = []

    for k, v in checkpoint['state_dict'].items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                pretrained_dict[k] = v
            else:
                mismatched_keys.append((k, model_dict[k].shape, v.shape))
        elif k=='tok_emb.weight' and model_dict['smiles_tok_emb.weight'].shape == v.shape:
            pretrained_dict['smiles_tok_emb.weight'] = v
        else:
            mismatched_keys.append((k, None, v.shape))
        
    # Update the model's state dict with the pretrained weights
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)

    # Print mismatched keys
    if mismatched_keys:
        print("Mismatched keys and shapes:")
        for key, model_shape, checkpoint_shape in mismatched_keys:
            print(f"Mismatched Key: {key} | Model shape: {model_shape} | Checkpoint shape: {checkpoint_shape}")
            assert key.split('.')[0] in ['smiles_lang_model', 'graph_lang_model', 'lang_model']
    else:
        print("All keys matched successfully!")

    return model


class CustomEarlyStopping(Callback):
    def __init__(self, config, patience=3):
        super().__init__()
        self.patience = patience
        self.val_loss_counter = 0
        self.val_auc_counter = 0
        self.best_val_loss = float('inf')
        self.best_val_metric = float('-inf')
        
        if config.task == 'classification':
            self.measure_name = 'AUC'
        elif config.task == 'regression':
            if config.dataset_name in ['qm7', 'qm8', 'qm9']:
                self.measure_name = 'MAE'
            else:
                self.measure_name = 'RMSE'

    def on_validation_end(self, trainer, pl_module):
        current_val_loss = trainer.callback_metrics.get('validation_loss')
        current_val_metric = trainer.callback_metrics.get('validation_'+self.measure_name)

        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.val_loss_counter = 0
        else:
            self.val_loss_counter += 1

        if current_val_metric > self.best_val_metric:
            self.best_val_metric = current_val_metric
            self.val_auc_counter = 0
        else:
            self.val_auc_counter += 1

        print('val_loss_counter:{}, val_auc_counter:{}, patience:{}'.format(self.val_loss_counter, self.val_auc_counter, self.patience))
        if self.val_loss_counter >= self.patience and self.val_auc_counter >= self.patience:
            trainer.should_stop = True


def main(config):
    if config.num_nodes > 1:
        LSB_MCPU_HOSTS = os.environ["LSB_MCPU_HOSTS"].split(' ') # Parses Node list set by LSF, in format hostname proceeded by number of cores requested
        HOST_LIST = LSB_MCPU_HOSTS[::2] # Strips the cores per node items in the list
        os.environ["MASTER_ADDR"] = HOST_LIST[0] # Sets the MasterNode to thefirst node on the list of hosts
        os.environ["MASTER_PORT"] = "54968"
        os.environ["NODE_RANK"] = str(HOST_LIST.index(os.environ["HOSTNAME"])) #Uses the list index for node rank, master node rank must be 0
        os.environ["NCCL_DEBUG"] = "INFO" #sets NCCL debug to info, during distributed training, bugs in code show up as nccl errors
        print(os.environ["HOSTNAME"] + " MASTER_ADDR: " + os.environ["MASTER_ADDR"])
        print(os.environ["HOSTNAME"] + " MASTER_PORT: " + os.environ["MASTER_PORT"])
        print(os.environ["HOSTNAME"] + " NODE_RANK " + os.environ["NODE_RANK"])
        print(os.environ["HOSTNAME"] + " NCCL_SOCKET_IFNAME: " + os.environ["NCCL_SOCKET_IFNAME"])
        print(os.environ["HOSTNAME"] + " NCCL_DEBUG: " + os.environ["NCCL_DEBUG"])
        print(os.environ["HOSTNAME"] + " NCCL_IB_CUDA_SUPPORT: " + os.environ["NCCL_IB_CUDA_SUPPORT"])
        print("Using " + str(config.num_nodes) + " Nodes---------------------------------------------------------------------")
        print("Using " + str(torch.cuda.device_count()) + " GPUs---------------------------------------------------------------------")
    else:
        print("Using " + str(config.num_nodes) + " Node----------------------------------------------------------------------")
        print("Using " + str(torch.cuda.device_count()) + " GPUs---------------------------------------------------------------------")

    train_config = {'batch_size':config.n_batch, 'num_workers':config.num_workers, 'valid_size':config.valid_size, 'test_size':config.test_size, 'data_path':config.data_path, 'target':config.target, 'task':config.task, 'splitting':config.splitting, 'results_path':config.results_path, 'graph_padding':config.graph_padding, 'train_load':config.train_load}
    dataset_loader = MoleculeModule(**train_config)
    dataset_loader.prepare_data()
    print('dataset_loader finish')
    
    if config.dataset_name in ['qm7', 'qm9']:
        labels = copy.deepcopy(dataset_loader.labels)
        normalizer = Normalizer(torch.tensor(labels))
        print('normalizer', normalizer.mean, normalizer.std)
    else:
        print('Not using normalizer')

    ## this should allow us to save a model for every x iterations and it should overwrite
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='validation_loss', # validation_AUC | validation_loss
                                                       mode='min',  # max | min
                                                       save_top_k=0, 
                                                       verbose=True,
                                                       save_last=False,
                                                       filename='{}-{}-{}'.format(config.dataset_name, config.task, config.target))
                                                       # auto_insert_metric_name=True,
                                                       # dirpath='./checkpoints', 
                                                       # filename='{}-{}-{}'.format(config.dataset_name, config.task, config.target)+'-{epoch}-{validation_loss:.2f}')
    early_stop_callback = CustomEarlyStopping(config, patience=50)

    trainer = pl.Trainer(default_root_dir=config.root_dir,
                max_epochs=config.max_epochs,
                strategy=config.accelerator,
                num_nodes=config.num_nodes,
                gpus=config.gpus,
                devices=[config.device_id],
                auto_select_gpus=True,
                callbacks=[checkpoint_callback, early_stop_callback], # early_stop_callback
                checkpoint_callback=True,
                # resume_from_checkpoint=config.restart_path if config.restart_path != "" else None,
                accumulate_grad_batches=config.grad_acc,
                num_sanity_val_steps=0,
                # val_check_interval=config.eval_every,
                weights_summary='full',
                auto_scale_batch_size='binsearch',
    )
    
    # # backup scripts
    if trainer.global_rank == 0:
        save_scripts(trainer=trainer)
    print('------')
    print('len(dataset_loader.train_dataloader())', len(dataset_loader.train_dataloader()))
    model = LightningModule(config, dataset_loader.get_vocab(), dataset_loader.get_graph_vocab(), len(dataset_loader.train_dataloader()), normalizer, trainer)
    print('model arch:', model)
    total = sum(p.numel() for p in model.parameters())
    print('Total params:{}'.format(total))
    # load pre-trained params
    if config.resume_from_checkpoint != 'from_scratch':
        model = load_model_weights(model, config.resume_from_checkpoint)
        if config.freeze == 'True':
            for param in model.blocks.parameters():
                param.requires_grad = False
            for name, param in model.named_parameters():
                print('frozen param:{} | {}'.format(name, param.requires_grad))
        else:
            print('Not freeze params')
    else:
        print('Training from scratch')
        # raise ValueError(config.resume_from_checkpoint)
        
    trainer.test(model, dataset_loader.get_test_dataloader())
    # trainer.fit(model=model, datamodule=dataset_loader)
    try:
        trainer.fit(model=model, datamodule=dataset_loader)
        interrupt_flag = False
    except:
        traceback.print_exc()
        interrupt_flag = True
    print('interrupt_flag', interrupt_flag)
        
    append_to_file(config.results_path, 
        f"{config.model_arch}," + f"{config.attention_type}," + f"{config.splitting}," 
        + f"{config.dataset_name}," + f"{config.target}," + f"{config.seed},"
        + f"{model.min_loss['min_epoch']},"
        + f"{model.min_loss['min_validation_loss']},"
        + f"{model.min_loss['min_test_loss']},"
        + f"best_test_measure:{model.min_loss['best_test_' + model.measure_name]},"
        + f"best_test_acc:{model.min_loss['best_test_acc']},"
        + f"min_test_reg_his:{model.min_loss['min_test_reg_his']},"
        + f"max_test_cls_his:{model.min_loss['max_test_cls_his']},"
        + f"freeze:{config.freeze},"
        + f"unfreeze:{config.unfreeze},"
        + f"aug_training_dataset:{config.aug_training_dataset},"
        + f"{trainer.checkpoint_callback.dirpath},"
        + f"{config.resume_from_checkpoint}",
    )

    if config.debug is True:
        pass
    else:
        rank_zero_warn('Debug mode not found eraseing cache')
    return interrupt_flag, model.min_loss['best_test_' + model.measure_name], model.min_loss['best_test_acc'], model.min_loss['max_test_cls_his'], model.min_loss['min_test_reg_his']
        

if __name__ == '__main__':
    config = args.parse_args()
    if config.resume_from_checkpoint != 'from_scratch':
        import re
        pattern = r'#([^/]+)'
        ckt_name = re.search(pattern, config.resume_from_checkpoint).group(1)
        epoch_name = config.resume_from_checkpoint.split('/')[-1]
        config.results_path = 'merged_results/merged_results_{}_{}_{}_{}_graph_padding{}_{}.csv'.format(config.model_arch, config.attention_type, ckt_name, epoch_name, config.graph_padding, config.pos_type)
    else:
        config.results_path = 'merged_results/merged_results_{}_{}_{}_graph_padding{}_{}.csv'.format(config.model_arch, config.attention_type, 'fromScratch', config.graph_padding, config.pos_type)
        
    print('config', config)

    if config.dataset_name == 'BBBP':
        config.task = 'classification'
        if config.splitting == 'molformer':
            config.data_path = '../data/finetune/molformer/bbbp/'
        else:
            config.data_path = '../data/finetune/bbbp/BBBP.csv'
        config.n_batch = min(128, config.n_batch)
        target_list = ["p_np"]

    elif config.dataset_name == 'Tox21':
        config.task = 'classification'
        if config.splitting == 'molformer':
            config.data_path = '../data/finetune/molformer/tox21'
        else:
            config.data_path = '../data/finetune/tox21/tox21.csv'
        target_list = [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", 
            "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
        ]

    elif config.dataset_name == 'ClinTox':
        config.task = 'classification'
        if config.splitting == 'molformer':
            config.data_path = '../data/finetune/molformer/clintox'
        else:
            config.data_path = '../data/finetune/clintox/clintox.csv'
        target_list = ['CT_TOX', 'FDA_APPROVED']

    elif config.dataset_name == 'HIV':
        config.task = 'classification'
        if config.splitting == 'molformer':
            config.data_path = '../data/finetune/molformer/hiv'
        else:
            config.data_path = '../data/finetune/hiv/HIV.csv'
        target_list = ["HIV_active"]

    elif config.dataset_name == 'BACE':
        config.task = 'classification'
        if config.splitting == 'molformer':
            config.data_path = '../data/finetune/molformer/bace/'
        else:
            config.data_path = '../data/finetune/bace/bace.csv'
        target_list = ["Class"]

    elif config.dataset_name == 'SIDER':
        config.task = 'classification'
        
        if config.splitting == 'molformer':
            config.data_path = '../data/finetune/molformer/sider'
        else:
            config.data_path = '../data/finetune/sider/sider.csv'
        target_list = [
            "Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues", 
            "Eye disorders", "Investigations", "Musculoskeletal and connective tissue disorders", 
            "Gastrointestinal disorders", "Social circumstances", "Immune system disorders", 
            "Reproductive system and breast disorders", 
            "Neoplasms benign, malignant and unspecified (incl cysts and polyps)", 
            "General disorders and administration site conditions", "Endocrine disorders", 
            "Surgical and medical procedures", "Vascular disorders", 
            "Blood and lymphatic system disorders", "Skin and subcutaneous tissue disorders", 
            "Congenital, familial and genetic disorders", "Infections and infestations", 
            "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders", 
            "Renal and urinary disorders", "Pregnancy, puerperium and perinatal conditions", 
            "Ear and labyrinth disorders", "Cardiac disorders", 
            "Nervous system disorders", "Injury, poisoning and procedural complications"
        ]

    elif config.dataset_name == 'MUV':
        config.task = 'classification'
        config.data_path = '../data/finetune/muv/muv.csv'
        target_list = [
            'MUV-692', 'MUV-689', 'MUV-846', 'MUV-859', 'MUV-644', 'MUV-548', 'MUV-852',
            'MUV-600', 'MUV-810', 'MUV-712', 'MUV-737', 'MUV-858', 'MUV-713', 'MUV-733',
            'MUV-652', 'MUV-466', 'MUV-832'
        ]

    elif config.dataset_name == 'FreeSolv':
        config.task = 'regression'
        if config.splitting == 'molformer':
            config.data_path = '../data/finetune/molformer/freesolv'
        else:
            config.data_path = '../data/finetune/freesolv/freesolv.csv'
        target_list = ["expt"]

    elif config.dataset_name == 'ESOL':
        config.task = 'regression'
        if config.splitting == 'molformer':
            config.data_path = '../data/finetune/molformer/esol'
        else:
            config.data_path = '../data/finetune/esol/esol.csv'
        target_list = ["measured log solubility in mols per litre"]

    elif config.dataset_name == 'Lipo':
        config.task = 'regression'
        if config.splitting == 'molformer':
            config.data_path = '../data/finetune/molformer/lipo'
        else:
            config.data_path = '../data/finetune/lipophilicity/Lipophilicity.csv'
        target_list = ["exp"]

    elif config.dataset_name == 'qm7':
        config.task = 'regression'
        config.data_path = '../data/finetune/qm7/qm7.csv'
        target_list = ["u0_atom"]

    elif config.dataset_name == 'qm8':
        config.task = 'regression'
        config.data_path = '../data/finetune/qm8/qm8.csv'
        target_list = [
            "E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", 
            "E1-PBE0", "E2-PBE0", 
            "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM", "f1-CAM","f2-CAM"
        ]

    elif config.dataset_name == 'qm9':
        config.task = 'regression'
        config.splitting = 'random'
        if config.splitting == 'molformer':
            config.data_path = '../data/finetune/molformer/qm9'
        else:
            config.splitting = 'random'
            config.data_path = '../data/finetune/qm9/qm9.csv'
        target_list = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'cv']

    else:
        raise ValueError('Error dataset_name:{}'.format(config.dataset_name))

    results_list = [[],[],[]]
    acc_list = []
    flag = False
    for target in target_list:
        config.target = target
        t_start = time.perf_counter()
        interrupt_flag, result, acc, max_test_cls, min_test_reg = main(config)
        flag = flag or interrupt_flag
        t_end = time.perf_counter()
        results_list[0].append(result)
        results_list[1].append(max_test_cls)
        results_list[2].append(min_test_reg)
        acc_list.append(acc)
        print('time consumption:{}'.format(t_end-t_start), results_list, acc_list)
    
    if not flag:
        append_to_file(config.results_path, 
            f"{config.model_arch}," + f"{config.attention_type}," + f"{config.splitting},"  + f"aug_training_dataset:{config.aug_training_dataset}," 
            + f"{config.dataset_name},"+ f"{str(target_list)},"  + f"{config.seed}," 
            + "-,-,-,"
            + f"measure:{np.mean(results_list[0])}|{np.std(results_list[0])},"
            + f"acc:{np.mean(acc_list)}|{np.std(acc_list)},"
            + f"max_test_cls:{np.mean(results_list[1])}|{np.std(results_list[1])},"
            + f"min_test_reg:{np.mean(results_list[2])}|{np.std(results_list[2])},"
            + f"freeze:{config.freeze}, unfreeze:{config.unfreeze}"
        )