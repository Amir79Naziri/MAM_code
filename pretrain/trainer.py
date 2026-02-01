import torch
import torch.distributed as dist
from tqdm import tqdm
import time
import os
import json
from utils import EarlyStopping
from torch.distributed import destroy_process_group
import pickle
import itertools

class Trainer:
    def __init__(self, model, model_config, train_data_loader, validation_data_loader, optimizer, criterion,  checkpoint_status, 
                 early_stopping, output_dir, cross_attn_regularization, accumulation_steps, masking_strategy, max_epochs=100, 
                 save_iteration_interval=1000, device_configs=None):
        self.model = model
        self.model_config = model_config
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.max_epochs = max_epochs
        self.output_dir = output_dir
        self.cross_attn_regularization = cross_attn_regularization
        self.checkpoint_status = checkpoint_status
        self.masking_strategy = masking_strategy
        self.accumulation_steps = accumulation_steps
        self.device_configs = device_configs
        self.save_iteration_interval = save_iteration_interval
        
        self.early_stopping = early_stopping
        if device_configs["master_process"]:
            print(f"early stopping enabled with patience {self.early_stopping.patience}")

    def train_per_epoch(self, epoch, initial_step=0, initial_loss=0.0):
        self.model.train()
        total_loss = torch.tensor(initial_loss, device=self.device_configs['device'])
        loss_accum = torch.tensor(0.0, device=self.device_configs['device'])
        accumulation_steps = self.accumulation_steps
        t0 = time.time()
        
        if initial_step > 0:
            data_iter = itertools.islice(self.train_data_loader, initial_step, None)
        else:
            data_iter = self.train_data_loader
        
        enumerated_iter = enumerate(data_iter, start=initial_step)
        
        if self.device_configs['master_process']:
            pbar_t = tqdm(
                enumerated_iter,
                total=len(self.train_data_loader),
                initial=initial_step,
                desc=f'training -> epoch {epoch}'
            )
        else:
            pbar_t = enumerated_iter
            
            
        self.optimizer.zero_grad()
        
        # start from the initial step
        for i, (_, data) in pbar_t:
            
            data = data.to(self.device_configs['device'])
            
            with torch.autocast(device_type='cuda' if 'cuda' in self.device_configs['device'] else 'cpu', dtype=torch.bfloat16):
                # encoder_data, encoder_data_padding, encoder_position_gene_ids, decoder_data, decoder_data_padding, decoder_position_gene_ids = prepare_data(data, self.model_config, device=self.device_configs['device'])
                if self.masking_strategy == 'cross-attention':
                    output, maskings, cross_attn_output = self.model(data)
                    loss = self.criterion(output[~maskings], data[~maskings]) + self.cross_attn_regularization * torch.norm(cross_attn_output, p=2)
                elif self.masking_strategy == 'random':
                    output, maskings = self.model(data)
                    loss = self.criterion(output[~maskings], data[~maskings])
                
            loss = loss / accumulation_steps  # Normalize loss to accumulate
            loss_accum += loss.detach() 
            
            
            if self.device_configs["ddp"]:
                self.model.require_backward_grad_sync = ((i + 1) % accumulation_steps == 0)
            loss.backward()
                
            if (i + 1) % accumulation_steps == 0:  # Wait until k accumulations before stepping optimizer
                if self.device_configs["ddp"]:
                    dist.all_reduce(loss_accum, op=dist.ReduceOp.SUM)
                
                self.optimizer.step()
                
                if self.device_configs["ddp"]:
                    torch.cuda.synchronize()
                    
                t1 = time.time()
                dt = t1 - t0 # time difference in seconds
                
                if self.device_configs["master_process"]:
                    print(f"epoch {epoch} | loss: {loss_accum:.6f} | dt: {dt*1000:.2f}ms")
                    
                self.optimizer.zero_grad()
                total_loss += loss_accum
                loss_accum = torch.tensor(0.0, device=self.device_configs['device'])                    
                t0 = time.time()
                
                # save snapshot of the model
                if self.device_configs["master_process"] and (i + 1) % (self.save_iteration_interval) == 0:
                    os.makedirs(self.output_dir, exist_ok=True)
                    
                    # change name of the model file in output_dir to model_old
                    if os.path.exists(f'{self.output_dir}/model.ckpt'):
                        os.rename(f'{self.output_dir}/model.ckpt', f'{self.output_dir}/model_old_.ckpt')
                    if os.path.exists(f'{self.output_dir}/checkpoint_status.json'):
                        os.rename(f'{self.output_dir}/checkpoint_status.json', f'{self.output_dir}/checkpoint_status_old_.json')
                    
                    if self.device_configs["ddp"]:
                        model = self.model.module.state_dict()
                    else:
                        model = self.model.state_dict()
                    torch.save(model, f'{self.output_dir}/model.ckpt')
                    checkpoint_status = {
                        'epoch': epoch,
                        'step': i + 1,
                        'train_loss': total_loss.item(),
                    }
                    with open(f'{self.output_dir}/checkpoint_status.json', 'w') as f:
                        json.dump(checkpoint_status, f)
                        
                    if os.path.exists(f'{self.output_dir}/model_old.ckpt'):
                        os.remove(f'{self.output_dir}/model_old.ckpt')
                        os.rename(f'{self.output_dir}/model_old_.ckpt', f'{self.output_dir}/model_old.ckpt')
                    if os.path.exists(f'{self.output_dir}/checkpoint_status_old.json'):
                        os.remove(f'{self.output_dir}/checkpoint_status_old.json')
                        os.rename(f'{self.output_dir}/checkpoint_status_old_.json', f'{self.output_dir}/checkpoint_status_old.json')
                
        if (i + 1) % accumulation_steps != 0:
            if self.device_configs["ddp"]:
                dist.all_reduce(loss_accum, op=dist.ReduceOp.SUM)
            
            self.optimizer.step()
            
            if self.device_configs["ddp"]:
                torch.cuda.synchronize()
                
            t1 = time.time()
            dt = t1 - t0
            
            if self.device_configs["master_process"]:
                print(f"epoch {epoch} | loss: {loss_accum:.6f} | dt: {dt*1000:.2f}ms")
                
            self.optimizer.zero_grad()
            total_loss += loss_accum
            loss_accum = torch.tensor(0.0, device=self.device_configs['device'])
            t0 = time.time()
            
        if self.device_configs["ddp"]:
            world_size = self.device_configs["ddp_world_size"]
            avg_loss = total_loss / (len(self.train_data_loader) * world_size)
        else:
            avg_loss = total_loss / len(self.train_data_loader)
        return avg_loss.item()


    def validate_per_epoch(self, epoch, return_maskings=False):
        self.model.eval()
        if return_maskings and self.device_configs["ddp"]:
            raise ValueError("return_maskings is not supported with DDP")
        else:
            maskings_list = []
            sample_ids_list = []
            
        total_loss = torch.tensor(0.0, device=self.device_configs['device'])
        if self.device_configs['master_process']:
            pbar_t = tqdm(enumerate(self.validation_data_loader), total=len(self.validation_data_loader), desc=f'validation -> epoch {epoch}')
        else:
            pbar_t = enumerate(self.validation_data_loader)
        
        with torch.no_grad():
            for i, (sample_id, data) in pbar_t:
                data = data.to(self.device_configs['device'])
                
                with torch.autocast(device_type='cuda' if 'cuda' in self.device_configs['device'] else 'cpu', dtype=torch.bfloat16):
                    # encoder_data, encoder_data_padding, encoder_position_gene_ids, decoder_data, decoder_data_padding, decoder_position_gene_ids = prepare_data(data, self.model_config, device=self.device_configs['device'])
                    if self.masking_strategy == 'cross-attention':
                        output, maskings, cross_attn_output = self.model(data)
                        loss = self.criterion(output[~maskings], data[~maskings]) + self.cross_attn_regularization * torch.norm(cross_attn_output, p=2)
                    elif self.masking_strategy == 'random':
                        output, maskings = self.model(data)
                        loss = self.criterion(output[~maskings], data[~maskings])
                
                total_loss += loss.detach()
                if return_maskings:
                    maskings_list.append(maskings)
                    sample_ids_list.append(sample_id)

        if self.device_configs["ddp"]:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            world_size = self.device_configs["ddp_world_size"]
            avg_loss = total_loss / (len(self.validation_data_loader) * world_size)
        else:
            avg_loss = total_loss / len(self.validation_data_loader)
        
        if self.device_configs["master_process"]:
            print(f'Validation Epoch {epoch}: Average Loss: {avg_loss:.6f}')
        
        if return_maskings:
            maskings_list = torch.cat(maskings_list, dim=0)
            sample_ids_list = torch.cat(sample_ids_list, dim=0)
            return avg_loss.item(), maskings_list, sample_ids_list
        
        return avg_loss.item()
    
    def save_model(self, epoch):
        if self.device_configs["master_process"]:
            os.makedirs(self.output_dir, exist_ok=True)
            raw_model = self.model.module if self.device_configs['ddp'] else self.model 
            torch.save(raw_model.state_dict(), f'{self.output_dir}/model_{epoch}.pt')


    def train(self):
        best_model = None
        best_val_loss = self.early_stopping.best_score
        
        initial_step = self.checkpoint_status["step"]
        initial_loss = self.checkpoint_status["train_loss"]
        
        for epoch in tqdm(range(self.checkpoint_status["epoch"], self.max_epochs), desc='epochs'):
            train_loss = self.train_per_epoch(epoch, initial_step=initial_step, initial_loss=initial_loss)
            initial_loss = 0.0
            initial_step = 0
            
            val_loss = self.validate_per_epoch(epoch)
            
            if self.device_configs["master_process"]:
                print(f'Epoch {epoch}: Training Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}')
                os.makedirs(self.output_dir, exist_ok=True)
                os.makedirs(f'{self.output_dir}/best', exist_ok=True) 
                
                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if self.device_configs["ddp"]:
                        best_model = self.model.module.state_dict()
                    else:
                        best_model = self.model.state_dict()
                        
                        
                    torch.save(best_model, f'{self.output_dir}/best/model.ckpt')
                    losses = {
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'val_loss': val_loss
                    }
                    with open(f'{self.output_dir}/best/losses.json', 'w') as f:
                        json.dump(losses, f)
                    
                
                
                if self.device_configs["ddp"]:
                    model = self.model.module.state_dict()
                else:
                    model = self.model.state_dict()
                torch.save(model, f'{self.output_dir}/model.ckpt')
                
                self.checkpoint_status = {
                    'epoch': epoch + 1,
                    'step': 0,
                    'train_loss': 0.0, # resets the training loss for new epoch
                }
                with open(f'{self.output_dir}/checkpoint_status.json', 'w') as f:
                    json.dump(self.checkpoint_status, f)
                
                
                losses = {
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'val_loss': val_loss
                }
                with open(f'{self.output_dir}/losses_{epoch}.json', 'w') as f:
                    json.dump(losses, f)
                    
                if self.early_stopping:
                    stop = self.early_stopping.log(val_loss)
                    
                    with open(f'{self.output_dir}/early_stopping_state.pkl', 'wb') as f:
                        pickle.dump(self.early_stopping, f)
                        
                    if stop:
                        print(f'early stopping at epoch {epoch}')
                        if self.device_configs["ddp"]:
                            print(f'ignore the errors that will appear below!\n\n\n\n\n\n')
                            destroy_process_group()
                        break     
                    
            torch.cuda.empty_cache()
            
            
        return self.model
    
    def validate(self):
        val_loss, maskings, samples = self.validate_per_epoch(0, return_maskings=True)
        
        if self.device_configs["master_process"]:
            print(f'Validation Loss: {val_loss:.6f}')
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(f'{self.output_dir}/mask_retrieve', exist_ok=True) 
            
            losses = {
                    'val_loss': val_loss
            }
            with open(f'{self.output_dir}/mask_retrieve/losses.json', 'w') as f:
                json.dump(losses, f)
                
            with open(f'{self.output_dir}/mask_retrieve/maskings.pkl', 'wb') as f:
                pickle.dump(maskings, f)
                 
            with open(f'{self.output_dir}/mask_retrieve/samples.pkl', 'wb') as f:
                pickle.dump(samples, f)
                
        torch.cuda.empty_cache()
        
        
        return self.model