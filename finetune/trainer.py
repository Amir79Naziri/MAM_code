import time
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import pickle
import itertools
import json

class Trainer:
    def __init__(self, model, model_config, train_data_loader, validation_data_loader, test_data_loader, optimizer, criterion, checkpoint_status,
                 early_stopping, output_dir, accumulation_steps, cross_attn_regularization, cross_attention_output_involevment, masking_strategy, save_preds=False, save_embedding=False, 
                 max_epochs=100, save_iteration_interval=1000, device_configs=None):
        self.model = model
        self.model_config = model_config
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.test_data_loader = test_data_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.max_epochs = max_epochs
        self.output_dir = output_dir
        self.accumulation_steps = accumulation_steps
        self.save_embedding = save_embedding
        self.save_preds = save_preds
        self.cross_attn_regularization = cross_attn_regularization
        self.masking_strategy = masking_strategy
        self.device_configs = device_configs
        self.checkpoint_status = checkpoint_status
        self.save_iteration_interval = save_iteration_interval
        self.cross_attention_output_involevment = cross_attention_output_involevment
        
        self.early_stopping = early_stopping
        if device_configs["master_process"] and self.early_stopping:
            print(f"early stopping enabled with patience {self.early_stopping.patience}")
        

    def train_per_epoch(self, epoch, initial_step=0, initial_loss=0.0):
        self.model.train()
        total_loss = torch.tensor(initial_loss, device=self.device_configs['device'], dtype=torch.float32)
        loss_accum = torch.tensor(0.0, device=self.device_configs['device'], dtype=torch.float32)
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
        
        for i, (_, (inputs, labels)) in pbar_t:
            
            # encoder_data, encoder_position_gene_ids = prepare_data_finetune(inputs, device='cpu')
            # encoder_data_padding = None
            
            # encoder_data = encoder_data.to(self.device_configs["device"])
            # encoder_position_gene_ids = encoder_position_gene_ids.to(self.device_configs["device"])
            # labels = labels.to(self.device_configs["device"])
            
            inputs = inputs.to(self.device_configs["device"])
            labels = labels.to(self.device_configs["device"])
            
            with torch.autocast(device_type='cuda' if 'cuda' in self.device_configs['device'] else 'cpu', dtype=torch.bfloat16):
                
                if self.masking_strategy == 'random' or self.cross_attention_output_involevment == 'decoder':
                    outputs = self.model(inputs, return_embedding=False)
                    loss = self.criterion(outputs, labels)
                elif self.masking_strategy == 'cross-attention':
                    outputs, cross_attn_output = self.model(inputs, return_embedding=False)
                    loss = self.criterion(outputs, labels) + self.cross_attn_regularization * torch.norm(cross_attn_output, p=2)
            
            loss = loss / accumulation_steps  # Normalize loss to accumulate
            loss_accum += loss.detach()
            
            if self.device_configs["ddp"]:
                self.model.require_backward_grad_sync = ((i + 1) % accumulation_steps == 0)
            loss.backward()
                
            
            
            if (i + 1) % accumulation_steps == 0:  # Wait until k accumulations before stepping optimizer
                if self.device_configs["ddp"]:
                    # Since loss_accum is a scalar, sum across processes
                    # loss_accum_tensor = torch.tensor(loss_accum, device=self.device_configs['device'])
                    dist.all_reduce(loss_accum, op=dist.ReduceOp.SUM)
                    # loss_accum = loss_accum.item()
                
                self.optimizer.step()
                
                if self.device_configs["ddp"]:
                    torch.cuda.synchronize()
                    
                t1 = time.time()
                dt = t1 - t0  # Time difference in seconds
                
                if self.device_configs["master_process"]:
                    print(f"epoch {epoch} | loss: {loss_accum.item():.6f} | dt: {dt*1000:.2f}ms")
                    
                self.optimizer.zero_grad()
                total_loss += loss_accum
                loss_accum = torch.tensor(0.0, device=self.device_configs['device'])      # Reset loss accumulator
                t0 = time.time()
                
                # save snapshot of the model
                if self.device_configs["master_process"] and (i + 1) % (self.save_iteration_interval) == 0:
                    os.makedirs(self.output_dir, exist_ok=True)
                    
                    
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
                
        # Handle remaining gradients
        if (i + 1) % accumulation_steps != 0:
            if self.device_configs["ddp"]:
                # loss_accum_tensor = torch.tensor(loss_accum, device=self.device_configs['device'])
                dist.all_reduce(loss_accum, op=dist.ReduceOp.SUM)
                # loss_accum = loss_accum.item()
            
            self.optimizer.step()
            
            if self.device_configs["ddp"]:
                torch.cuda.synchronize()
                
            t1 = time.time()
            dt = t1 - t0
            
            if self.device_configs["master_process"]:
                print(f"epoch {epoch} | loss: {loss_accum.item():.6f} | dt: {dt*1000:.2f}ms")
                
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


    def validate_per_epoch(self, epoch, test=False):
        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.device_configs['device'])
        y_true = []
        y_pred = []
        ids = []
        embeddings_list = []
        
        if self.device_configs['master_process']:
            if test:
                pbar_t = tqdm(enumerate(self.test_data_loader), total=len(self.test_data_loader), desc=f'test')
            else:
                pbar_t = tqdm(enumerate(self.validation_data_loader), total=len(self.validation_data_loader), desc=f'validation -> epoch {epoch}')
        else:
            if test:
                pbar_t = enumerate(self.test_data_loader)
            else:
                pbar_t = enumerate(self.validation_data_loader)
        
        with torch.no_grad():
            for _, (id, (inputs, labels)) in pbar_t:
                # encoder_data, encoder_position_gene_ids = prepare_data_finetune(inputs, device='cpu')
                # encoder_data_padding = None
                
                # encoder_data = encoder_data.to(self.device_configs["device"])
                # encoder_position_gene_ids = encoder_position_gene_ids.to(self.device_configs["device"])
                # labels = labels.to(self.device_configs["device"])
                
                inputs = inputs.to(self.device_configs["device"])
                labels = labels.to(self.device_configs["device"])
                    
                with torch.autocast(device_type='cuda' if 'cuda' in self.device_configs['device'] else 'cpu', dtype=torch.bfloat16):
                    
                    if self.masking_strategy == 'random' or self.cross_attention_output_involevment == 'decoder':
                        if self.save_embedding:
                            outputs, embeddings = self.model(inputs, return_embedding=True)
                        else:
                            outputs = self.model(inputs, return_embedding=False)
                        loss = self.criterion(outputs, labels)
                    elif self.masking_strategy == 'cross-attention':
                        if self.save_embedding:
                            outputs, embeddings, cross_attn_output= self.model(inputs, return_embedding=True)
                        else:
                            outputs, cross_attn_output= self.model(inputs, return_embedding=False)
                        loss = self.criterion(outputs, labels) + self.cross_attn_regularization * torch.norm(cross_attn_output, p=2)
                
                total_loss += loss.detach()
                
                _, predicted = torch.max(outputs.data, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                
                if self.save_preds: 
                    ids.extend(id.cpu().numpy())
                    
                    if self.save_embedding:
                        embeddings_list.extend(embeddings.cpu().numpy())
            
        f1_micro = torch.tensor(f1_score(y_true, y_pred, average='micro'), dtype=torch.float32, device=self.device_configs["device"])
        precision_micro = torch.tensor(precision_score(y_true, y_pred, average='micro'), dtype=torch.float32, device=self.device_configs["device"])
        recall_micro = torch.tensor(recall_score(y_true, y_pred, average='micro'), dtype=torch.float32, device=self.device_configs["device"])
        
        f1_macro = torch.tensor(f1_score(y_true, y_pred, average='macro'), dtype=torch.float32, device=self.device_configs["device"])
        precision_macro = torch.tensor(precision_score(y_true, y_pred, average='macro'), dtype=torch.float32, device=self.device_configs["device"])
        recall_macro = torch.tensor(recall_score(y_true, y_pred, average='macro'), dtype=torch.float32, device=self.device_configs["device"])
        
        accuracy = torch.tensor(accuracy_score(y_true, y_pred), dtype=torch.float32, device=self.device_configs["device"])
        
        if self.device_configs["ddp"]:
            world_size = self.device_configs["ddp_world_size"]
            if test:
                total_size = len(self.test_data_loader) * world_size
            else:
                total_size = len(self.validation_data_loader) * world_size
            # average loss across all processes per data point
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            avg_loss = total_loss / total_size
             
            # average metrics across all processes per data batch
            dist.all_reduce(f1_micro, op=dist.ReduceOp.AVG)
            dist.all_reduce(precision_micro, op=dist.ReduceOp.AVG)
            dist.all_reduce(recall_micro, op=dist.ReduceOp.AVG) 
            dist.all_reduce(f1_macro, op=dist.ReduceOp.AVG)
            dist.all_reduce(precision_macro, op=dist.ReduceOp.AVG)
            dist.all_reduce(recall_macro, op=dist.ReduceOp.AVG)
            dist.all_reduce(accuracy, op=dist.ReduceOp.AVG)
            
            if self.save_preds:
                y_true = torch.tensor(y_true, dtype=torch.long, device=self.device_configs["device"])
                y_pred = torch.tensor(y_pred, dtype=torch.long, device=self.device_configs["device"])
                ids = torch.tensor(ids, dtype=torch.long, device=self.device_configs["device"])
            
                if self.save_embedding:
                    embeddings_list = torch.tensor(embeddings_list, dtype=torch.float32, device=self.device_configs["device"])

                # Ignore extra -1 ones
                gathered_y_trues = [torch.full_like(y_true, fill_value=-1) for _ in range(self.device_configs["ddp_world_size"])]
                gathered_y_preds = [torch.full_like(y_pred, fill_value=-1) for _ in range(self.device_configs["ddp_world_size"])]
                gathered_ids = [torch.full_like(ids, fill_value=-1) for _ in range(self.device_configs["ddp_world_size"])]
                if self.save_embedding:
                    gathered_embeddings = [torch.full_like(embeddings_list, fill_value=-1) for _ in range(self.device_configs["ddp_world_size"])]

                # Gather across all processes
                dist.all_gather(gathered_y_trues, y_true)
                dist.all_gather(gathered_y_preds, y_pred)
                dist.all_gather(gathered_ids, ids)
                if self.save_embedding:
                    dist.all_gather(gathered_embeddings, embeddings_list)

                # Concatenate the results to have complete y_true and y_pred across all processes
                gathered_y_trues = torch.cat(gathered_y_trues).cpu().numpy()
                gathered_y_preds = torch.cat(gathered_y_preds).cpu().numpy()
                gathered_ids = torch.cat(gathered_ids).cpu().numpy()
                if self.save_embedding:
                    gathered_embeddings = torch.cat(gathered_embeddings).cpu().numpy()
        else:
            if test:
                avg_loss = total_loss / len(self.test_data_loader)
            else:
                avg_loss = total_loss / len(self.validation_data_loader)
            
            if self.save_preds:
                gathered_y_preds = y_pred
                gathered_y_trues = y_true
                gathered_ids = np.array(ids)
                if self.save_embedding:
                    gathered_embeddings = np.array(embeddings_list)
        
        if self.device_configs["master_process"]:
            if test:
                print(f'Test: Average Loss: {avg_loss:.6f} | f1_micro: {f1_micro:.2f} | f1_macro: {f1_macro:.2f} | precision_micro: {precision_micro:.2f} | precision_macro: {precision_macro:.2f} | recall_micro: {recall_micro:.2f} | recall_macro: {recall_macro:.2f} | accuracy: {accuracy:.2f}')
            else:
                print(f'Validation Epoch {epoch}: Average Loss: {avg_loss:.6f} | f1_micro: {f1_micro:.2f} | f1_macro: {f1_macro:.2f} | precision_micro: {precision_micro:.2f} | precision_macro: {precision_macro:.2f} | recall_micro: {recall_micro:.2f} | recall_macro: {recall_macro:.2f} | accuracy: {accuracy:.2f}')
            
            
        metrics = {
            "f1_micro": f1_micro.item(), 
            "precision_micro": precision_micro.item(), 
            "recall_micro": recall_micro.item(), 
            "f1_macro": f1_macro.item(), 
            "precision_macro": precision_macro.item(), 
            "recall_macro": recall_macro.item(), 
            "accuracy": accuracy.item()
        }
        
        results = None
        if self.save_preds:
            results = {
                "y_true": gathered_y_trues,
                "y_pred": gathered_y_preds,
                "ids": gathered_ids
            }
        
            if self.save_embedding:
                results["embeddings"] = gathered_embeddings
                del gathered_embeddings
                
        return avg_loss.item(), metrics, results            


    def train(self):
        best_model = None
        best_val_loss = self.early_stopping.best_score
        
        initial_step = self.checkpoint_status["step"]
        initial_loss = self.checkpoint_status["train_loss"]
        
        for epoch in tqdm(range(self.checkpoint_status["epoch"], self.max_epochs), desc='epochs'):
            train_loss = self.train_per_epoch(epoch, initial_step, initial_loss)
            initial_loss = 0.0
            initial_step = 0
            
            val_loss, metrics, _ = self.validate_per_epoch(epoch)
            
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
                    metrics.update({'epoch': epoch, 'validation_loss': best_val_loss, 'train_loss': train_loss})
                
                    with open(f'{self.output_dir}/best/validation_results.txt', 'w') as f:
                        f.write(str(metrics))
                  
                          
                    
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
                
                metrics.update({'epoch': epoch, 'validation_loss': best_val_loss, 'train_loss': train_loss})
                
                with open(f'{self.output_dir}/validation_results_{epoch}.txt', 'w') as f:
                    f.write(str(metrics))
                    
                    
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
    
    def test(self, sample_number=None):
        val_loss, metrics, results = self.validate_per_epoch(0, test=True)
        if sample_number is None:
            test_dir = "test"
        else:
            test_dir = f"test_sample_{sample_number}"
            
        if self.device_configs["master_process"]:
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(f'{self.output_dir}/{test_dir}', exist_ok=True)

            metrics.update({'validation_loss': val_loss})

            with open(f'{self.output_dir}/{test_dir}/test_results.txt', 'w') as f:
                f.write(str(metrics))
            
            # save as json
            with open(f'{self.output_dir}/{test_dir}/test_results.json', 'w') as f:
                json.dump(metrics, f)
                
            if self.save_preds:
                with open(f'{self.output_dir}/{test_dir}/test_outputs.pkl', 'wb') as f:
                    pickle.dump(results, f)
                
                
        torch.cuda.empty_cache()
        
        return self.model