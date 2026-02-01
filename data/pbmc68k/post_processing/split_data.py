import scanpy as sc
import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    data_config = str(sys.argv[1])
except:
    print("Please provide the path to the data config file")
    sys.exit(1)
    
with open(data_config) as f:
    data_config = json.load(f)

adata = sc.read(data_config['path'] + '/pbmc68k.h5ad')

# Example filtering/preprocessing
adata.layers["counts_temp"] = adata.X.copy()
adata.X = adata.layers["counts"].copy()
sc.pp.filter_cells(adata, min_genes=200) 
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.highly_variable_genes(adata, n_top_genes=data_config['top_genes'], flavor='seurat_v3')
highly_variable_genes = adata.var['highly_variable']
temp_index = adata.obs.index.copy()
adata.X = adata.layers["counts_temp"].copy()
del adata.layers["counts_temp"]
adata = adata[temp_index]

adata.var.to_csv(data_config['path'] + "/genes.csv")
highly_variable_genes.to_csv(data_config['path'] + '/highly_variable_genes.csv')

######################
# Stratify only by celltype
######################
to_split_df = pd.DataFrame({'celltype': adata.obs['celltype']}, index=adata.obs.index)

random_state = data_config['seed']

# Split data by celltype
train_idx, temp_idx = train_test_split(
    to_split_df.index,
    test_size=0.2,
    random_state=random_state,
    stratify=to_split_df['celltype']
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,
    random_state=random_state,
    stratify=to_split_df.loc[temp_idx, 'celltype']
)

# Assign splits
to_split_df['set'] = None
to_split_df.loc[train_idx, 'set'] = 'train'
to_split_df.loc[val_idx, 'set'] = 'val'
to_split_df.loc[test_idx, 'set'] = 'test'

# Copy back to adata
adata.obs['set'] = to_split_df['set']

train_data = adata[adata.obs['set'] == 'train'].copy()
val_data   = adata[adata.obs['set'] == 'val'].copy()
test_data  = adata[adata.obs['set'] == 'test'].copy()

del adata, to_split_df
gc.collect()

# cell_name,cell_type,genes
def transform_data(df, max_genes=2000):
    transformed_data = []
    for row_idx in tqdm(range(df.shape[0])):
        name = df[row_idx].obs.index.values[0]
        ID = df[row_idx].obs['ID'].values[0]
        cell_type = df[row_idx].obs['celltype'].values[0]
        
        non_zero_indices = set(np.nonzero(df[row_idx].X.toarray().squeeze())[0])
        genes = df[row_idx].var.index[[idx for idx in np.argsort(df[row_idx].X.toarray().squeeze())[::-1] if idx in non_zero_indices]][:max_genes].map(lambda x: x.upper())
        
        
        numbers = df[row_idx].X.toarray().squeeze()[[idx for idx in np.argsort(df[row_idx].X.toarray().squeeze())[::-1] if idx in non_zero_indices]][:max_genes].astype(str)
        genes_with_expression = ' '.join([f"{gene}#{number}" for gene, number in zip(genes, numbers)])
        
        transformed_data.append([ID, name, cell_type, genes_with_expression])
        
        
    
    
    new_df = pd.DataFrame(transformed_data, columns=["ID", "cell_name", "cell_type", "genes"])
    return new_df

def process_batch(batch, df, max_genes):
    batch_result = []
    for row_idx in batch:
        name = df[row_idx].obs.index.values[0]
        ID = df[row_idx].obs['ID'].values[0]
        cell_type = df[row_idx].obs['celltype'].values[0]

        non_zero_indices = set(np.nonzero(df[row_idx].X.toarray().squeeze())[0])
        genes = df[row_idx].var.index[[idx for idx in np.argsort(df[row_idx].X.toarray().squeeze())[::-1] if idx in non_zero_indices]][:max_genes].map(lambda x: x.upper())

        numbers = df[row_idx].X.toarray().squeeze()[[idx for idx in np.argsort(df[row_idx].X.toarray().squeeze())[::-1] if idx in non_zero_indices]][:max_genes].astype(str)
        genes_with_expression = ' '.join([f"{gene}#{number}" for gene, number in zip(genes, numbers)])
        batch_result.append([ID, name, cell_type, genes_with_expression])
        
            
            
    return batch_result

def transform_data_multi_thread(df, max_genes=2000, num_threads=4):
    transformed_data = []
    batch_size = len(df) // num_threads

    batches = [range(i * batch_size, min((i + 1) * batch_size, len(df))) for i in range(num_threads)]
    

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_batch, batch, df, max_genes) for batch in batches]
        for future in tqdm(as_completed(futures), total=len(futures)):
            transformed_data.extend(future.result())
            
    
    
    new_df = pd.DataFrame(transformed_data, columns=["ID", "cell_name", "cell_type", "genes"])
    
    return new_df

import os; os.makedirs(data_config['path'] + f'/seed_{random_state}', exist_ok=True)

print("*"*10 + " Saving raw val data " + "*"*10)
val_data.obs['ID'] = range(val_data.shape[0])
val_data.write(data_config['path'] + f'/seed_{random_state}/pbmc68k_val.h5ad')

if data_config['only_highly_variable']:
    val_data = val_data[:, highly_variable_genes]

print("*"*10 + " Transforming val data " + "*"*10)
transformed_val_data = transform_data_multi_thread(val_data, num_threads=40, max_genes=data_config['top_genes'])
del val_data
gc.collect();

transformed_val_data.to_csv(data_config['path'] + f'/seed_{random_state}/pbmc68k_val_with_expression{"_highly_variable" if data_config["only_highly_variable"] else ""}.csv', index=False)

del transformed_val_data
gc.collect();


print("*"*10 + " Saving raw test data " + "*"*10)
test_data.obs['ID'] = range(test_data.shape[0])
test_data.write(data_config['path'] + f'/seed_{random_state}/pbmc68k_test.h5ad')

if data_config['only_highly_variable']:
    test_data = test_data[:, highly_variable_genes]

print("*"*10 + " Transforming test data " + "*"*10)
transformed_test_data = transform_data_multi_thread(test_data, num_threads=40, max_genes=data_config['top_genes'])
del test_data
gc.collect();

transformed_test_data.to_csv(data_config['path'] + f'/seed_{random_state}/pbmc68k_test_with_expression{"_highly_variable" if data_config["only_highly_variable"] else ""}.csv', index=False)

del transformed_test_data
gc.collect();


print("*"*10 + " Saving raw train data " + "*"*10)
train_data.obs['ID'] = range(train_data.shape[0])
train_data.write(data_config['path'] + f'/seed_{random_state}/pbmc68k_train.h5ad')

if data_config['only_highly_variable']:
    train_data = train_data[:, highly_variable_genes]

print("*"*10 + " Transforming train data " + "*"*10)
transformed_train_data = transform_data_multi_thread(train_data, num_threads=40, max_genes=data_config['top_genes'])
del train_data
gc.collect();

transformed_train_data.to_csv(data_config['path'] + f'/seed_{random_state}/pbmc68k_train_with_expression{"_highly_variable" if data_config["only_highly_variable"] else ""}.csv', index=False)

del transformed_train_data
gc.collect();
print("done!")