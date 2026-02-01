## Modeified code from https://github.com/biomap-research/scFoundation

import torch
import os
import numpy as np
import random
from select_model import select_model  
import math
from torch.distributed import init_process_group

def next_16x(x):
    return int(math.ceil(x / 16) * 16)

def seed_all(seed, cuda_deterministic=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if cuda_deterministic: # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        
        
def setup_device():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "we need CUDA for DDP"
        init_process_group(backend='nccl', init_method='env://')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        
        if master_process:
            print(f"Running DDP on CUDA with {ddp_world_size}")
            
        print(f"rank: {ddp_rank} | local_rank: {ddp_local_rank} initialized")
    else:
        # vanilla, non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        # attempt to autodetect device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
            
        print('Using device: ', device)
        
        
    return {
            'ddp': ddp, 
            'ddp_rank': ddp_rank, 
            'ddp_local_rank': ddp_local_rank, 
            'ddp_world_size': ddp_world_size, 
            'master_process': master_process, 
            'device': device
        }
    
        
def convertconfig(ckpt):
    newconfig = {}
    newconfig['config']={}
    model_type = ckpt['config']['model']
    
    for key, val in ckpt['config']['model_config'][model_type].items():
        newconfig['config'][key]=val
        
    for key, val in ckpt['config']['dataset_config']['rnaseq'].items():
        newconfig['config'][key]=val
        
    if model_type == 'performergau_resolution':
        model_type = 'performer_gau'
    
    import collections
    d = collections.OrderedDict()
    for key, val in ckpt['state_dict'].items():
        d[str(key).split('model.')[1]]=val
        
    newconfig['config']['model_type']=model_type
    newconfig['model_state_dict']=d
    newconfig['config']['pos_embed']=False
    newconfig['config']['device']='cuda'
    return newconfig

def load_model_frommmf(best_ckpt_path, key='gene', default_config=None):
    model_data = torch.load(best_ckpt_path,map_location='cpu')
    model_data = model_data[key]
    model_data = convertconfig(model_data)
    if not model_data.__contains__('config'):
        print('***** No config *****')
        config={}
        config['model_type']='flash_all'
    else:
        config=model_data['config']
        # print(config)
    if not config.__contains__('qv_dim'):
        if config['model'] != 'mae_autobin':
            if config.__contains__('dim_head'):
                config['qv_dim']=config['dim_head']
            else:
                print('***** No qv_dim ***** set 64')
                config['qv_dim']= 64
    if not config.__contains__('ppi_edge'):
        config['ppi_edge']=None
    if default_config is not None:    
        config.update(default_config)
    model = select_model(config)
    model_state_dict = model_data['model_state_dict']    
    model.load_state_dict(model_state_dict, strict=False)
    model.initialize_new_parameters()
    return model, config

class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        
    def log(self, loss):
        if self.best_score is None:
            self.best_score = loss
        elif loss >= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = loss
            self.counter = 0
        return False
    
    
