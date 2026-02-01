import sys
sys.path.append('../model/')
sys.path.append('../data/')
sys.path.append('../../model/')
sys.path.append('../../data/')
sys.path.append('../../../model/')
sys.path.append('../../../data/')
import argparse
from utils import *
import pickle
from mae_autobin import CellAnnoClassifierR, CellAnnoClassifierCA
import torch 
from torch import nn
import torch.optim as optim
from trainer import *
from dataset import AnnotationDataset as Dataset
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torch.distributed import destroy_process_group
import warnings
warnings.filterwarnings('ignore')
import pickle
import json


def calculate_weights(A):
    max_A = max(A)
    B = [max(max_A / a, 50) for a in A]
    total_B = sum(B)
    weights = [b / total_B for b in B]
    return torch.tensor(weights, dtype=torch.float32)


def main(args):
    args.batch_size = 32 # temporarily set batch size for server 2
    print(args)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    device_configs = setup_device()
        
    seed_all(args.seed)
    
    assert args.masking_strategy in ["cross-attention", "random"], "Masking strategy must be either cross-attention or random"
    
    assert args.save_iteration_interval > 0, "Save iteration interval must be greater than 0"
    assert args.save_iteration_interval % args.accumulation_steps == 0, "Save iteration interval must be divisible by accumulation steps"
    
    torch.set_float32_matmul_precision('high')
    
    
    base_model, base_model_config = load_model_frommmf(args.original_model_dir, 
                                                       default_config={
                                                            "latent_dim": args.latent_dim,
                                                            "num_latents": args.num_latents,
                                                            "advanced_masking": args.advanced_masking,
                                                            "masking_ratio": args.masking_ratio,
                                                            "max_tokens": args.max_tokens,
                                                            "cross_attention_heads": args.cross_attention_heads,
                                                            "masking_strategy": args.masking_strategy,
                                                            "cross_attention_output_involevment": args.cross_attention_output_involevment
                                                        })
    if args.pretrained_model_dir != 'None':
        state_dict = torch.load(args.pretrained_model_dir)
        # new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        base_model.load_state_dict(state_dict)
        del state_dict
        
    
    if args.masking_strategy == 'random':
        classification_model = CellAnnoClassifierR(n_class=args.num_classes, n_genes=args.num_genes, model=base_model, model_config=base_model_config, linear_probe=args.linear_probe)
    else:
        classification_model = CellAnnoClassifierCA(n_class=args.num_classes, n_genes=args.num_genes, model=base_model, model_config=base_model_config, frozen_masking_layers=args.frozen_masking_layers, linear_probe=args.linear_probe)
            
    
    del base_model
    
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
                    if args.task == "train" and early_stopping.counter >= args.patience:
                        sys.exit(101)
            except FileNotFoundError:
                print('No early stopping state found.')
                raise FileNotFoundError     
        else:
            early_stopping = EarlyStopping(patience=args.patience)
        
        if device_configs["master_process"]:
            print('Resuming training from epoch', checkpoint_status['epoch'])
            print("Checkpoint status:", checkpoint_status)
        classification_model.load_state_dict(torch.load(args.output_dir + f'/model.ckpt'))
    elif args.task == 'test':
        if device_configs["master_process"]:
            print(f'Testing the model from {args.output_dir + f"best/model.ckpt"}')
        classification_model.load_state_dict(torch.load(args.output_dir + f'best/model.ckpt'))
    else:
        early_stopping = EarlyStopping(patience=args.patience)   
        checkpoint_status = {
            'epoch': 0,
            'step': 0,
            'train_loss': 0.0
        }
    
    classification_model.to(device_configs["device"])
    
    if device_configs["master_process"]:
        print(classification_model)
        for name, param in classification_model.named_parameters():
            if param.requires_grad:
                print(f"Layer {name}: {param.numel()} parameters")
    
    if device_configs["ddp"]:
        classification_model = DDP(classification_model, device_ids=[device_configs["ddp_rank"]])

    
    if args.task == 'train':
        if device_configs["master_process"]:
            print("Training the model")
        
        train_data = Dataset(args.train_data)       
        if device_configs["ddp"]:
            train_sampler = DistributedSampler(train_data, num_replicas=device_configs["ddp_world_size"], rank=device_configs["ddp_rank"])
            train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, sampler=train_sampler)
        else:
            train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        
        val_data = Dataset(args.val_data)
        if device_configs["ddp"]:
            val_sampler = DistributedSampler(val_data, num_replicas=device_configs["ddp_world_size"], rank=device_configs["ddp_rank"])
            val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, sampler=val_sampler)
        else:
            val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        
        label_value_counts = [train_data.adata.obs['label'].value_counts()[i] for i in range(args.num_classes)]
        weights = calculate_weights(label_value_counts)
        criterion = nn.CrossEntropyLoss(weight=weights.to(device_configs["device"]))
        
        optimizer = optim.Adam(classification_model.parameters(), lr=args.lr, amsgrad=True, weight_decay=args.weight_decay)
        if args.gradient_clipping > 0:
            torch.nn.utils.clip_grad_norm_(classification_model.parameters(), max_norm=args.gradient_clipping)
        
        trainer = Trainer(model=classification_model, model_config=base_model_config, train_data_loader=train_loader, validation_data_loader=val_loader, 
                        test_data_loader=None, optimizer=optimizer, criterion=criterion, early_stopping=early_stopping, checkpoint_status=checkpoint_status,
                        output_dir=args.output_dir, accumulation_steps=args.accumulation_steps, masking_strategy=args.masking_strategy, save_preds=False, save_embedding=False, # save_preds and save_embedding must be False for Training
                        cross_attn_regularization=args.cross_attn_regularization, cross_attention_output_involevment=args.cross_attention_output_involevment, max_epochs=args.max_epochs, save_iteration_interval=args.save_iteration_interval, device_configs=device_configs)
        
        
        trainer.train()
                                        
        if device_configs["master_process"]:
            print('finished training.')
            
    elif args.task == 'test':
        if device_configs["master_process"]:
            print("Testing the model")
            
        if args.use_test_sample:
            for i in range(args.test_sample_number):
                if device_configs["master_process"]:
                    print('Testing sample number:', i)
                test_data = Dataset(args.test_data + f'/{args.test_sample_filename}_{i}.h5ad')       
                if device_configs["ddp"]:
                    test_sampler = DistributedSampler(test_data, num_replicas=device_configs["ddp_world_size"], rank=device_configs["ddp_rank"])
                    test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, sampler=test_sampler)
                else:
                    test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)   
                
                label_value_counts = [test_data.adata.obs['label'].value_counts()[i] for i in range(args.num_classes)]
                weights = calculate_weights(label_value_counts)
                criterion = nn.CrossEntropyLoss(weight=weights.to(device_configs["device"]))     
                
                trainer = Trainer(model=classification_model, model_config=base_model_config, train_data_loader=None, validation_data_loader=None, 
                                test_data_loader=test_loader, optimizer=None, criterion=criterion, early_stopping=None, checkpoint_status=None,
                                output_dir=args.output_dir, accumulation_steps=None, masking_strategy=args.masking_strategy, save_preds=args.save_preds, save_embedding=args.save_embedding, 
                                cross_attn_regularization=args.cross_attn_regularization, cross_attention_output_involevment=args.cross_attention_output_involevment, max_epochs=None, save_iteration_interval=None, device_configs=device_configs)
                
                trainer.test(sample_number=i)
        else:
                
            test_data = Dataset(args.test_data)       
            if device_configs["ddp"]:
                test_sampler = DistributedSampler(test_data, num_replicas=device_configs["ddp_world_size"], rank=device_configs["ddp_rank"])
                test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, sampler=test_sampler)
            else:
                test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)   
                
            label_value_counts = [test_data.adata.obs['label'].value_counts()[i] for i in range(args.num_classes)]
            weights = calculate_weights(label_value_counts)
            criterion = nn.CrossEntropyLoss(weight=weights.to(device_configs["device"]))     
                
            trainer = Trainer(model=classification_model, model_config=base_model_config, train_data_loader=None, validation_data_loader=None, 
                            test_data_loader=test_loader, optimizer=None, criterion=criterion, early_stopping=None, checkpoint_status=None,
                            output_dir=args.output_dir, accumulation_steps=None, masking_strategy=args.masking_strategy, save_preds=args.save_preds, save_embedding=args.save_embedding, 
                            cross_attn_regularization=args.cross_attn_regularization, cross_attention_output_involevment=args.cross_attention_output_involevment, max_epochs=None, save_iteration_interval=None, device_configs=device_configs)
            
            
            trainer.test()
        
        if device_configs["master_process"]:
            print('finished testing.')
            
    else:
        raise ValueError('Invalid task. Choose either train or test')
    
    
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

# torchrun --standalone --nproc_per_node=4 cell_annotation_finetune.py
# torchrun --standalone --nnodes=2 --nproc_per_node=8 cell_annotation_finetune.py
# python cell_annotation_finetune.py
if __name__=='__main__':
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
    parser.add_argument("--original_model_dir", required=False, 
                        default="/local/home/am/EX_M/scFoundation/pretrained_model/model.ckpt", 
                        type=str, help="Path to the original model")
    parser.add_argument("--pretrained_model_dir", required=False, default="/local/home/am/EX_M/scDAE_mn/model_4.pt",type=str, help="Path to the pretrained model")
    parser.add_argument("--max_epochs", required=False, default=10, type=int, help="Number of epochs to train the model")
    parser.add_argument("--batch_size", required=False, default=64, type=int, help="Batch size")
    parser.add_argument("--accumulation_steps", required=False, default=5, type=int, help="Number of accumulation steps")
    parser.add_argument("--lr", required=False, default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--seed", required=False, default=42, type=int, help="Seed for reproducibility")
    parser.add_argument("--num_classes", required=False, default=11, type=int, help="Number of classes")
    parser.add_argument("--num_genes", required=False, default=19266, type=int, help="Number of genes")
    parser.add_argument("--output_dir", required=False, default="/local/home/am/EX_M/scDAE_mn/finetuned_model_full", type=str, help="Path to the output directory")
    parser.add_argument("--num_workers", required=False, default=2, type=int, help="Number of workers")
    parser.add_argument("--device", required=False, default="cuda", type=str, help="Device to use")
    parser.add_argument("--patience", required=False, default=3, type=int, help="Patience for early stopping")
    parser.add_argument("--save_embedding", required=False, default=True, type=str2bool, help="Save embeddings")   
    parser.add_argument("--save_preds", required=False, default=True, type=str2bool, help="Save predictions")
    parser.add_argument("--resume", required=False, default=False, type=str2bool, help="Resume training")
    parser.add_argument("--masking_strategy", required=False, default='random', type=str, help="Masking strategy")
    parser.add_argument("--cross_attn_regularization", required=False, default=1e-5, type=float, help="Cross attention regularization")
    parser.add_argument("--latent_dim", required=False, default=512, type=int)
    parser.add_argument("--num_latents", required=False, default=256, type=int)
    parser.add_argument("--advanced_masking", required=False, default=False, type=str2bool)
    parser.add_argument("--masking_ratio", required=False, default=0.3 * 0.07, type=float)
    parser.add_argument("--max_tokens", required=False, default=6000, type=int)
    parser.add_argument("--gradient_clipping", required=False, default=-1.0, type=float)
    parser.add_argument("--cross_attention_heads", required=False, default=8, type=int)
    parser.add_argument("--cross_attention_output_involevment", required=False, default='both', type=str)
    parser.add_argument("--frozen_masking_layers", required=False, default=False, type=str2bool)
    parser.add_argument("--save_iteration_interval", required=False, default=10000, type=int)
    parser.add_argument("--weight_decay", required=False, default=5e-4, type=float)
    parser.add_argument("--linear_probe", required=False, default=False, type=str2bool)
    parser.add_argument("--use_test_sample", required=False, default=False, type=str2bool, help="whether to test samples or not")
    parser.add_argument("--test_sample_number", required=False, default=5, type=int, help="sample number to test")
    parser.add_argument("--test_sample_filename", required=False, default='test_sample', type=str, help="sample filename to test")

    args = parser.parse_args()
    main(args)