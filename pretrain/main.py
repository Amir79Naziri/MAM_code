import sys
sys.path.append('../model/')
sys.path.append('../data/')
sys.path.append('../../model/')
sys.path.append('../../data/')
sys.path.append('../../../model/')
sys.path.append('../../../data/')
import argparse
from utils import *
import torch
from trainer import *
from dataset import PretrainDataset as Dataset
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torch.distributed import destroy_process_group
import warnings
warnings.filterwarnings('ignore')
import pickle
import json

def main(args):
    # args.batch_size = 8 # **********************************# for server 2
    print(args)
    device_configs = setup_device()
        
    seed_all(args.seed)
    
    assert args.masking_strategy in ["cross-attention", "random"], "Masking strategy must be either cross-attention or random"
    
    assert args.save_iteration_interval > 0, "Save iteration interval must be greater than 0"
    assert args.save_iteration_interval % args.accumulation_steps == 0, "Save iteration interval must be divisible by accumulation steps"
    
    torch.set_float32_matmul_precision('high')
    model,model_config = load_model_frommmf(args.initialized_model_dir, 
                                            default_config={
                                                "latent_dim": args.latent_dim,
                                                "num_latents": args.num_latents,
                                                "advanced_masking": args.advanced_masking,
                                                "masking_ratio": args.masking_ratio,
                                                "max_tokens": args.max_tokens,
                                                "cross_attention_heads": args.cross_attention_heads,
                                                "masking_strategy": args.masking_strategy,
                                                "cross_attention_output_involevment": args.cross_attention_output_involevment
                                                }
                                            )
    
    if args.resume:
        try:
            with open(f'{args.output_dir}/checkpoint_status.json', 'r') as f:
                checkpoint_status = json.load(f)
        except FileNotFoundError:
            print('No checkpoint status found.')
            raise FileNotFoundError   
        
        if checkpoint_status['epoch'] > 0:
            try:
                with open(f'{args.output_dir}/early_stopping_state.pkl', 'rb') as f:
                    early_stopping = pickle.load(f)
                    if early_stopping.counter >= args.early_stopping_patience:
                        sys.exit(101)
            except FileNotFoundError:
                print('No early stopping state found.')
                raise FileNotFoundError     
        else:
            early_stopping = EarlyStopping(patience=args.early_stopping_patience)
        
        if device_configs["master_process"]:
            print('Resuming training from epoch', checkpoint_status['epoch'])
            print("Checkpoint status:", checkpoint_status)
        model.load_state_dict(torch.load(args.output_dir + f'/model.ckpt'))
        
    else:
        early_stopping = EarlyStopping(patience=args.early_stopping_patience)   
        checkpoint_status = {
            'epoch': 0,
            'step': 0,
            'train_loss': 0.0
        }
        
    model.to(device_configs["device"])
    
    if args.task == "validate":
        if device_configs["master_process"]:
            print('Validating the best model...')
        model.load_state_dict(torch.load(args.output_dir + f'/best/model.ckpt'))
    
    
    if device_configs["ddp"]:
        model = DDP(model, device_ids=[device_configs["ddp_rank"]], find_unused_parameters=args.find_unused_parameters)

    # Initialize parameter groups
    param_groups = [
        {'params': [], 'lr': args.nlr},
        {'params': [], 'lr': args.olr}
    ]
    
    if args.masking_strategy == "cross-attention":
        new_layers = [
            "module.latents",
            "module.cross_attention.in_proj_weight",
            "module.cross_attention.in_proj_bias",
            "module.cross_attention.out_proj.weight",
            "module.cross_attention.out_proj.bias",
            "module.input_proj.weight",
            "module.input_proj.bias",
            "module.output_proj.weight",
            "module.output_proj.bias",
            "module.token_emb.mlp.weight",
            "module.token_emb.mlp.bias",
            "module.token_emb.mlp2.weight",
            "module.token_emb.mlp2.bias",
            "module.token_emb.emb.weight",
            "module.token_emb.emb_mask.weight",
            "module.token_emb.emb_pad.weight",
            "module.pos_emb.weight",
            "module.decoder_embed.weight",
            "module.decoder_embed.bias",
            "module.norm.weight",
            "module.norm.bias",
            "module.to_final.weight",
            "module.to_final.bias",
            "module.decoder.norm.weight",
            "module.decoder.norm.bias",
        ]
    elif args.masking_strategy == "random":
        new_layers = [
            "module.token_emb.mlp.weight",
            "module.token_emb.mlp.bias",
            "module.token_emb.mlp2.weight",
            "module.token_emb.mlp2.bias",
            "module.token_emb.emb.weight",
            "module.token_emb.emb_mask.weight",
            "module.token_emb.emb_pad.weight",
            "module.pos_emb.weight",
            "module.decoder_embed.weight",
            "module.decoder_embed.bias",
            "module.norm.weight",
            "module.norm.bias",
            "module.to_final.weight",
            "module.to_final.bias",
            "module.decoder.norm.weight",
            "module.decoder.norm.bias",
        ]

    # Populate the parameter groups
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name in new_layers:
                param_groups[0]['params'].append(param)
            else:
                param_groups[1]['params'].append(param)
            
    if device_configs["master_process"]:
        print(model)
        
    train_data = Dataset(data_path=args.train_data)
    validation_data = Dataset(data_path=args.val_data)
    
    if args.task == "validate":
        if device_configs["master_process"]:
            print('Setting batch size to at least 8 for validation')
        args.batch_size = max(8, args.batch_size)
        
        
    if device_configs["ddp"]:
        train_sampler = DistributedSampler(train_data, num_replicas=device_configs["ddp_world_size"], rank=device_configs["ddp_rank"])
        train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, sampler=train_sampler, pin_memory=True)
    else:
        train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
        
    if device_configs["ddp"]:
        validation_sampler = DistributedSampler(validation_data, num_replicas=device_configs["ddp_world_size"], rank=device_configs["ddp_rank"])
        validation_loader = DataLoader(validation_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, sampler=validation_sampler, pin_memory=True)
    else:
        validation_loader = DataLoader(validation_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
        
    
    optimizer = torch.optim.Adam(param_groups, weight_decay=args.weight_decay, amsgrad=True)
    if args.gradient_clipping > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clipping)
    criterion = torch.nn.MSELoss()
    trainer = Trainer(model, model_config, train_loader, validation_loader, 
                      optimizer, criterion, checkpoint_status, early_stopping=early_stopping, 
                      output_dir=args.output_dir, cross_attn_regularization=args.cross_attn_regularization, 
                      accumulation_steps=args.accumulation_steps, masking_strategy=args.masking_strategy, 
                      max_epochs=args.num_epochs, save_iteration_interval=args.save_iteration_interval,
                      device_configs=device_configs)
    if args.task == "train":
        trainer.train()
    else:
        trainer.validate()
        
    if device_configs["master_process"]:
        print('finished training.')
        
    if device_configs["ddp"]:
        destroy_process_group()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True', 'TRUE'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False', 'FALSE'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    parser.add_argument("--task", required=False, default='train', type=str, help="Specify the task to run")
    parser.add_argument("--train_data", required=False, 
                        default="/local/home/am/EX_M/datasets/scRNA/merged/seed_32/scFoundation/train/bone_marrow_train_processed.h5ad", 
                        type=str, help="Path to the training data")
    parser.add_argument("--val_data", required=False, 
                        default="/local/home/am/EX_M/datasets/scRNA/merged/seed_32/scFoundation/validation/bone_marrow_val_processed.h5ad", 
                        type=str, help="Path to the validation data")
    parser.add_argument("--test_data", required=False, 
                        default="/local/home/am/EX_M/datasets/scRNA/merged/seed_32/scFoundation/test/bone_marrow_test_processed.h5ad", 
                        type=str, help="Path to the test data")
    parser.add_argument("--initialized_model_dir", required=False, default="/local/home/am/EX_M/scFoundation/pretrained_model/model.ckpt", type=str, help="Path to the trained model")
    parser.add_argument("--num_epochs", required=False, default=5, type=int, help="Number of epochs to train the model")
    parser.add_argument("--batch_size", required=False, default=1, type=int, help="Batch size")
    parser.add_argument("--nlr", required=False, default=1e-4, type=float, help="New Layers Learning rate")
    parser.add_argument("--olr", required=False, default=1e-6, type=float, help="Old Layers Learning rate")
    parser.add_argument("--cross_attn_regularization", required=False, default=1e-5, type=float, help="Cross attention regularization")
    parser.add_argument("--seed", required=False, default=42, type=int, help="Seed for reproducibility")
    parser.add_argument("--output_dir", required=False, default="/local/home/am/EX_M/scDAE_mn2/", type=str, help="Path to the output directory")
    parser.add_argument("--num_workers", required=False, default=4, type=int, help="Number of workers")
    parser.add_argument("--early_stopping_patience", required=False, default=3, type=int)
    parser.add_argument("--weight_decay", required=False, default=5e-4, type=float)
    parser.add_argument("--accumulation_steps", required=False, default=5, type=int)
    parser.add_argument("--find_unused_parameters", required=False, default=False, type=str2bool)
    parser.add_argument("--resume", required=False, default=False, type=str2bool)
    parser.add_argument("--masking_strategy", required=False, default="cross-attention", type=str)
    parser.add_argument("--latent_dim", required=False, default=512, type=int)
    parser.add_argument("--num_latents", required=False, default=256, type=int)
    parser.add_argument("--advanced_masking", required=False, default=False, type=str2bool)
    parser.add_argument("--masking_ratio", required=False, default=0.3 * 0.08, type=float)
    parser.add_argument("--max_tokens", required=False, default=6000, type=int)
    parser.add_argument("--gradient_clipping", required=False, default=-1.0, type=float)
    parser.add_argument("--cross_attention_heads", required=False, default=8, type=int)
    parser.add_argument("--cross_attention_output_involevment", required=False, default='both', type=str)
    parser.add_argument("--save_iteration_interval", required=False, default=10000, type=int)

    args = parser.parse_args()
    main(args)