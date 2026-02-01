import scanpy as sc
import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


data_config = {
    "path": "/local/home/am/EX_M/datasets/scRNA/heart",
    "seed": 32,
    "only_highly_variable": True,
    "top_genes": 2000
}

random_state = data_config['seed']
# try:
#     data_config = str(sys.argv[1])
# except:
#     print("Please provide the path to the data config file")
#     sys.exit(1)
    
# with open(data_config) as f:
#     data_config = json.load(f)

# change the path to the location of the data
adata = sc.read(data_config['path'] + '/heart_preprocessed.h5ad')

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

adata.obs["celltype"] = adata.obs["cell_type"]

print("*"*10 + " Merged data " + "*"*10)
print(adata.obs['age.order'].value_counts())
# print(adata.obs['sample'].value_counts())
print(adata.obs['sex'].value_counts())
print(adata.obs['celltype'].value_counts())
# print(adata.obs['donor_id'].value_counts())
# print(adata.obs['tissue'].value_counts())
# print(adata.obs['disease'].value_counts())

to_split_df = pd.DataFrame({
    'celltype': adata.obs['celltype'],
    'age.order': adata.obs['age.order'],
    'sex': adata.obs['sex'],
    'donor_id': range(adata.shape[0]),
    # 'tissue': adata.obs['tissue'],
    })
# ---- Step 1: Create a combined strata label per observation ----
to_split_df['strata'] = (
    to_split_df['celltype'].astype(str) + '_' +
    # to_split_df['age.order'].astype(str) + '_' +
    to_split_df['sex'].astype(str) 
)

# ---- Step 2: Compute a representative strata per donor ----
def safe_mode(series):
    """Return the mode if available, else the first element."""
    return series.mode().iat[0] if not series.mode().empty else series.iat[0]

donor_strata = (
    to_split_df.groupby('donor_id', observed=False)['strata']
    .agg(safe_mode)
)

unique_donors = donor_strata.index.values

# ---- Step 3: Split donors into train / val / test ----
train_donors, temp_donors = train_test_split(
    unique_donors,
    train_size=0.8,
    random_state=random_state,
    stratify=donor_strata
)

temp_strata = donor_strata[temp_donors]

val_donors, test_donors = train_test_split(
    temp_donors,
    train_size=0.5,
    random_state=random_state,
    stratify=temp_strata
)

# ---- Step 4: Assign split labels to cells ----
def assign_set(donor_id):
    if donor_id in train_donors:
        return 'train'
    elif donor_id in val_donors:
        return 'val'
    else:
        return 'test'

to_split_df['set'] = to_split_df['donor_id'].map(assign_set)

print("*"*10 + " Train data " + "*"*10)
print(to_split_df[to_split_df['set'] == 'train']['celltype'].value_counts())
print(to_split_df[to_split_df['set'] == 'train']['age.order'].value_counts())
# print(to_split_df[to_split_df['set'] == 'train']['sample'].value_counts())
print(to_split_df[to_split_df['set'] == 'train']['sex'].value_counts())
# print(to_split_df[to_split_df['set'] == 'train']['tissue'].value_counts())
# print(to_split_df[to_split_df['set'] == 'train']['disease'].value_counts())

print("*"*10 + " Test data " + "*"*10)
print(to_split_df[to_split_df['set'] == 'test']['celltype'].value_counts())
print(to_split_df[to_split_df['set'] == 'test']['age.order'].value_counts())
# print(to_split_df[to_split_df['set'] == 'test']['sample'].value_counts())
print(to_split_df[to_split_df['set'] == 'test']['sex'].value_counts())
# print(to_split_df[to_split_df['set'] == 'test']['tissue'].value_counts())
# print(to_split_df[to_split_df['set'] == 'test']['disease'].value_counts())
print("*"*10 + " Val data " + "*"*10)
print(to_split_df[to_split_df['set'] == 'val']['celltype'].value_counts())
print(to_split_df[to_split_df['set'] == 'val']['age.order'].value_counts())
# print(to_split_df[to_split_df['set'] == 'val']['sample'].value_counts())
print(to_split_df[to_split_df['set'] == 'val']['sex'].value_counts())
# print(to_split_df[to_split_df['set'] == 'val']['tissue'].value_counts())
# print(to_split_df[to_split_df['set'] == 'val']['disease'].value_counts())

adata.obs['set'] = to_split_df['set']

train_data = adata[adata.obs['set'] == 'train'].copy()
val_data = adata[adata.obs['set'] == 'val'].copy()
test_data = adata[adata.obs['set'] == 'test'].copy()

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
val_data.write(data_config['path'] + f'/seed_{random_state}/heart_val.h5ad')

if data_config['only_highly_variable']:
    val_data = val_data[:, highly_variable_genes]

print("*"*10 + " Transforming val data " + "*"*10)
transformed_val_data = transform_data_multi_thread(val_data, num_threads=40, max_genes=data_config['top_genes'])
del val_data
gc.collect();

transformed_val_data.to_csv(data_config['path'] + f'/seed_{random_state}/heart_val_with_expression{"_highly_variable" if data_config["only_highly_variable"] else ""}.csv', index=False)

del transformed_val_data
gc.collect();


print("*"*10 + " Saving raw test data " + "*"*10)
test_data.obs['ID'] = range(test_data.shape[0])
test_data.write(data_config['path'] + f'/seed_{random_state}/heart_test.h5ad')

if data_config['only_highly_variable']:
    test_data = test_data[:, highly_variable_genes]

print("*"*10 + " Transforming test data " + "*"*10)
transformed_test_data = transform_data_multi_thread(test_data, num_threads=40, max_genes=data_config['top_genes'])
del test_data
gc.collect();

transformed_test_data.to_csv(data_config['path'] + f'/seed_{random_state}/heart_test_with_expression{"_highly_variable" if data_config["only_highly_variable"] else ""}.csv', index=False)

del transformed_test_data
gc.collect();


print("*"*10 + " Saving raw train data " + "*"*10)
train_data.obs['ID'] = range(train_data.shape[0])
train_data.write(data_config['path'] + f'/seed_{random_state}/heart_train.h5ad')

if data_config['only_highly_variable']:
    train_data = train_data[:, highly_variable_genes]

print("*"*10 + " Transforming train data " + "*"*10)
transformed_train_data = transform_data_multi_thread(train_data, num_threads=40, max_genes=data_config['top_genes'])
del train_data
gc.collect();

transformed_train_data.to_csv(data_config['path'] + f'/seed_{random_state}/heart_train_with_expression{"_highly_variable" if data_config["only_highly_variable"] else ""}.csv', index=False)

del transformed_train_data
gc.collect();
print("done!")