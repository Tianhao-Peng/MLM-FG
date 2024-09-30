import torch
import shutil
from torch import nn
import args
import os
import numpy as np
import random
import getpass
import traceback
from datasets import load_dataset
from pubchem_encoder import Encoder
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_warn, rank_zero_only, seed
from torch.optim.lr_scheduler import LambdaLR

from model.model_builder import RotateEncoderBuilder as rotate_builder
from transformers import RobertaConfig, RobertaModel

from fast_transformers.feature_maps import GeneralizedRandomFeatures
from model.position_encoding import SmilesPositionalEncoding
import torch.nn.functional as F
from functools import partial
from torch.optim import AdamW

from torch.utils.data import DataLoader
import subprocess
from model.corss_attention import CrossAttentionTail
from utils import save_scripts


class LightningModule(pl.LightningModule):

    def __init__(self, config, vocab, graph_vocab):
        super(LightningModule, self).__init__()

        self.save_hyperparameters(config)
        self.vocabulary = vocab
        #location of cache File
        # Special symbols

        self.debug = config.debug
        # Word embeddings layer
        n_vocab, d_emb = len(vocab.keys()), config.n_embd
        
        ## transformer
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
        elif config.model_arch == 'RoBERTa':
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
        
        # input embedding stem
        self.pos_emb = None
        self.smiles_tok_emb = nn.Embedding(n_vocab+10, config.n_embd)
        n_graph_vocab = len(graph_vocab.keys())  # 123
        self.graph_tok_emb = nn.Embedding(n_graph_vocab+10, config.n_embd)
            
        self.drop = nn.Dropout(config.d_dropout)
        self.smiles_lang_model = self.lm_layer(config.n_embd, n_vocab)
        self.graph_lang_model = self.lm_layer(config.n_embd, n_graph_vocab)
        self.train_config = config
        #if we are starting from scratch set seeds
        if config.restart_path == "":
            seed.seed_everything(config.seed)
        
    class lm_layer(nn.Module):
        def __init__(self, n_embd, n_vocab):
            super().__init__()
            self.embed = nn.Linear(n_embd, n_embd)
            self.ln_f = nn.LayerNorm(n_embd)
            self.head = nn.Linear(n_embd, n_vocab, bias=False)
        def forward(self, tensor):
            tensor = self.embed(tensor)
            tensor = F.gelu(tensor)
            tensor = self.ln_f(tensor)
            tensor = self.head(tensor)
            return tensor
            
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
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
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
        betas = (0.9, 0.99)
        learning_rate = self.train_config.lr_start * self.train_config.lr_multiplier
        optimizer = AdamW(optim_groups, lr=learning_rate, betas=betas)
        
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "training_loss"}

    
    def loss_cl(self, x1, x2):
        T = 0.5
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix_a = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix_a = torch.exp(sim_matrix_a / T)
        pos_sim_a = sim_matrix_a[range(batch_size), range(batch_size)]
        loss_a = pos_sim_a / (sim_matrix_a.sum(dim=1) - pos_sim_a)
        loss_a = - torch.log(loss_a).mean()

        sim_matrix_b = torch.einsum('ik,jk->ij', x2, x1) / torch.einsum('i,j->ij', x2_abs, x1_abs)
        sim_matrix_b = torch.exp(sim_matrix_b / T)
        pos_sim_b = sim_matrix_b[range(batch_size), range(batch_size)]
        loss_b = pos_sim_b / (sim_matrix_b.sum(dim=1) - pos_sim_b)
        loss_b = - torch.log(loss_b).mean()

        loss = (loss_a + loss_b) / 2
        return loss
    
    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()  # free unused memory
        
        idxl =     batch[0]
        targetsl = batch[1]
        postition_embeddingl = batch[2]
        hyper_adjl = batch[3]

        loss, smiles_loss, graph_loss = 0., 0., 0.
        loss_tmp = 0
        for chunk in range(len(idxl)): 
            smiles_idx, graph_idx = idxl[chunk] # element index in vocab dictory
            smiles_targets, graph_targets = targetsl[chunk]
            graph_pos = postition_embeddingl[chunk]
            smiles_pos = None
            hyper_adj = hyper_adjl[chunk]
            if self.train_config.model_arch in ['SMILESonly']:
                smiles_emb =  self.smiles_tok_emb(smiles_idx)
                smiles_emb = self.drop(smiles_emb)
                smiles_out = self.blocks(smiles_emb)  # Shape: (batch_size, seq_len, head_num, hidden_dim)
                smiles_logits = self.smiles_lang_model(smiles_out)
                if smiles_targets is not None:
                    smiles_logits = smiles_logits.view(-1, smiles_logits.size(-1))
                    smiles_targets = smiles_targets.view(-1)
                    true_token_lprobs = F.cross_entropy(smiles_logits, smiles_targets, ignore_index=-100)
                    smiles_loss_tmp = true_token_lprobs/len(idxl)
                if chunk < len(idxl)-1:
                    smiles_loss_tmp.backward()
                    loss += smiles_loss_tmp.detach()
                else:
                    loss += smiles_loss_tmp
                        
            elif config.model_arch in ['RoBERTa']:
                smiles_emb =  self.smiles_tok_emb(smiles_idx)
                smiles_emb = self.drop(smiles_emb)
                smiles_out = self.blocks(inputs_embeds=smiles_emb)['last_hidden_state']
                smiles_logits = self.smiles_lang_model(smiles_out)
                if smiles_targets is not None:
                    smiles_logits = smiles_logits.view(-1, smiles_logits.size(-1))
                    smiles_targets = smiles_targets.view(-1)
                    true_token_lprobs = F.cross_entropy(smiles_logits, smiles_targets, ignore_index=-100)
                    smiles_loss_tmp = true_token_lprobs/len(idxl)
                if chunk < len(idxl)-1:
                    smiles_loss_tmp.backward()
                    loss += smiles_loss_tmp.detach()
                else:
                    loss += smiles_loss_tmp
        return {'loss':loss, 'smiles_loss':smiles_loss, 'graph_loss':graph_loss}#, 'log':tensorboard_log}
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.tensor([output['loss'] for output in outputs]).mean()
        avg_smiles_loss = torch.tensor([output['smiles_loss'] for output in outputs]).mean()
        avg_graph_loss = torch.tensor([output['graph_loss'] for output in outputs]).mean()
        loss = {'training_loss': avg_loss.item(), 'smiles_loss':avg_smiles_loss.item(), 
                'graph_loss':avg_graph_loss.item()}
        self.log('training_loss', loss['training_loss'])
        self.log('smiles_loss', loss['smiles_loss'])
        self.log('graph_loss', loss['graph_loss'])

    def validation_epoch_end(self, outputs):
        pass
        
    def validation_step(self, batch, batch_idx):
        pass

class MoleculeModule(pl.LightningDataModule):
    def __init__(self,  max_len, data_path, train_args, pretrain_style, graph_padding):
        super().__init__()
        self.data_path_ori = data_path
        self.data_path = None
        self.train_args = train_args  # dict with keys {'batch_size', 'shuffle', 'num_workers', 'pin_memory'}
        print('MoleculeModule', train_args)
        self.text_encoder = Encoder(max_len, batch_size=train_args['batch_size'], vocab_path=data_path, pretrain_style=pretrain_style, graph_padding=graph_padding)


    def prepare_data(self):
        pass

    def get_vocab(self):
        #using home made tokenizer, should look into existing tokenizer
        return self.text_encoder.char2id
    
    def get_graph_vocab(self):
        return self.text_encoder.graph_char2id

    def get_cache(self):
        return self.cache_files
    
    def setup(self, stage=None):
        #using huggingface dataloader
        # create cache in tmp directory of locale mabchine under the current users name to prevent locking issues
        assert self.data_path_ori in ['pubchem-10m', 'pubchem-10k', 'pubchem-100m', 'pubchem-20m', 'ZINC-pubchem-1b']
        pubchem_smiles_path = {'train':os.path.join('../data/pretrain/', self.data_path_ori, 
                                                    '{}.txt'.format(self.data_path_ori))}
        dataset_dict = load_dataset('./pubchem_smiles_script.py', data_files=pubchem_smiles_path, cache_dir=os.path.join('/tmp',getpass.getuser(), 'pubchem_smiles'),split='train')
        
        self.pubchem= dataset_dict
        print('setup', dataset_dict.cache_files)
        self.cache_files = []

        for cache in dataset_dict.cache_files:
            tmp = '/'.join(cache['filename'].split('/')[:4])
            self.cache_files.append(tmp)

    def train_dataloader(self):
        loader = DataLoader(self.pubchem, collate_fn=self.text_encoder.process, **self.train_args)
        print('train_dataloader', len(loader))
        return loader

    def val_dataloader(self):
        pass
    def test_dataloader(self):
        pass
    
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
        epoch = trainer.current_epoch
        global_step = trainer.global_step

        if global_step % self.save_step_frequency == 0 and self.save_step_frequency > 10 and global_step > 0:

            filename = f"{self.prefix}_{epoch}_{global_step}.ckpt"
            print('CheckpointEveryNSteps: trainer.checkpoint_callback.dirpath:{p1}, filename:{p2}'.format(p1=trainer.checkpoint_callback.dirpath, p2=filename))
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)
            
class ModelCheckpointAtEpochEnd(pl.Callback):
    
    def __init__(self, every_n_epochs):
        self.every_n_epochs = every_n_epochs
    
    def on_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        metrics['epoch'] = trainer.current_epoch
        current_epoch = trainer.current_epoch

        if current_epoch % self.every_n_epochs == 0:
            loss = '{:.4f}'.format(metrics['training_loss'].item())
            smiles_loss = '{:.4f}'.format(metrics['smiles_loss'].item())
            graph_loss = '{:.4f}'.format(metrics['graph_loss'].item())
            filename = "epoch{p1}_loss{p2}_smiles{p3}_graph{p4}.ckpt".format(
                            p1=current_epoch, p2=loss, p3=smiles_loss, p4=graph_loss)
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)
        
@rank_zero_only
def remove_tree(cachefiles):
    if type(cachefiles) == type([]):
        #if cachefiles are identical remove all but one file path
        cachefiles = list(set(cachefiles))
        for cache in cachefiles:
            shutil.rmtree(cache)
    else:
        shutil.rmtree(cachefiles)

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
    get_nccl_socket_ifname()
    os.environ['NCCL_IB_CUDA_SUPPORT'] = '1'
        
def main():
    # fix_infiniband()

    config = args.parse_args()
    print('config', config)
    if config.num_nodes > 1:
        LSB_MCPU_HOSTS = os.environ["LSB_MCPU_HOSTS"].split(' ') # Parses Node list set by LSF, in format hostname proceeded by number of cores requested
        HOST_LIST = LSB_MCPU_HOSTS[::2] # Strips the cores per node items in the list
        os.environ["MASTER_ADDR"] = HOST_LIST[0] # Sets the MasterNode to the first node on the list of hosts
        os.environ["MASTER_PORT"] = "54966"
        os.environ["NODE_RANK"] = str(HOST_LIST.index(os.environ["HOSTNAME"])) #Uses the list index for node rank, master node rank must be 0
        os.environ["NCCL_DEBUG"] = "INFO" #sets NCCL debug to info, during distributed training, bugs in code show up as nccl errors
        print(os.environ["HOSTNAME"] + " MASTER_ADDR: " + os.environ["MASTER_ADDR"])
        print(os.environ["HOSTNAME"] + " MASTER_PORT: " + os.environ["MASTER_PORT"])
        print('HOST_LIST', HOST_LIST)
        print(os.environ["HOSTNAME"] + " NODE_RANK " + os.environ["NODE_RANK"])
        print(os.environ["HOSTNAME"] + " NCCL_DEBUG: " + os.environ["NCCL_DEBUG"])
        print("Using " + str(config.num_nodes) + " Nodes---------------------------------------------------------------------")
        print("Using " + str(torch.cuda.device_count()) + " GPUs---------------------------------------------------------------------")
    else:
        print("Using " + str(config.num_nodes) + " Node----------------------------------------------------------------------")
        print("Using " + str(torch.cuda.device_count()) + " GPUs---------------------------------------------------------------------")

    train_config = {'batch_size':config.n_batch, 'num_workers':config.n_workers, 'pin_memory':False}
    ## this should allow us to save a model for every x iterations and it should overwrite
    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=-1, verbose=True)
    print('checkpoint_callback', checkpoint_callback, checkpoint_callback.dirpath)
    # checkpoint_callback = pl.callbacks.ModelCheckpoint(period=1, save_top_k=-1, verbose=True)
    train_loader = MoleculeModule(max_len=config.max_len, data_path=config.train_load, train_args=train_config, pretrain_style=config.pretrain_style, graph_padding=config.graph_padding) 
    train_loader.setup()#config.debug)
    cachefiles = train_loader.get_cache()
    print('train_loader finish')
    model = LightningModule(config, train_loader.get_vocab(), train_loader.get_graph_vocab())
    
    print('model arch:', model)
    total = sum(p.numel() for p in model.parameters())
    print('Total params:{}'.format(total))

    trainer = pl.Trainer(default_root_dir=config.root_dir,
                max_epochs=config.max_epochs,
                accelerator=config.accelerator,
                num_nodes=config.num_nodes,
                gpus=config.gpus,
                auto_select_gpus=True,
                callbacks=[ModelCheckpointAtEpochEnd(config.every_n_epochs), CheckpointEveryNSteps(config.checkpoint_every)],
                checkpoint_callback=checkpoint_callback,
                resume_from_checkpoint=config.restart_path if config.restart_path != "" else None,
                accumulate_grad_batches=config.grad_acc,
                num_sanity_val_steps=10,
                val_check_interval=config.eval_every,
                weights_summary='full')
    
    # # backup scripts
    if trainer.global_rank == 0:
        save_scripts(trainer=trainer)
    
    try:
        trainer.fit(model, train_loader)
    except Exception as exp:
        print(type(exp))
        print(exp)
        traceback.print_exc()
        rank_zero_warn('We have caught an error, trying to shut down gracefully')
        remove_tree(cachefiles)

    if config.debug is True:
        pass
    else:
        rank_zero_warn('Debug mode not found eraseing cache')
        remove_tree(cachefiles)

if __name__ == '__main__':
    config = args.parse_args()
    print('config', config)
    main()
