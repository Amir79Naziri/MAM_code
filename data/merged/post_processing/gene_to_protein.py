from bioservices import Ensembl
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import sys
import json


try:
    data_config = str(sys.argv[1])
except:
    print("Please provide the path to the data config file")
    sys.exit(1)
    
with open(data_config) as f:
    data_sources = json.load(f)

data_sources['path'] = '../../../../EX_M/datasets/scRNA/merged'

ensembl = Ensembl()
genes = pd.read_csv(data_sources['path'] + '/genes.csv', index_col=0)


def fetch_data(symbols):
    try:
        response = ensembl.post_lookup_by_symbol(symbols=symbols, species="homo_sapiens", expand=True)
        return response, symbols
    except Exception as e:
        print(f"Failed to fetch data for {symbols}: {e}")
        return None

# Function to split gene list into chunks
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Number of symbols per thread
symbols_per_thread = 30

# Total number of threads
num_workers = 10

# This function manages the distribution of gene symbols across threads and monitors progress
def process_genes(gene_symbols):
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Create a list of tasks submitted to the executor
        tasks = [executor.submit(fetch_data, chunk) for chunk in chunks(gene_symbols, symbols_per_thread)]
        all_responses = {}

        # Use tqdm to track the progress of tasks
        for task in tqdm(as_completed(tasks), total=len(tasks), desc="Processing Genes", unit="batch"):
            try:
                response, _ = task.result()
                if response:
                    all_responses.update(response)
            except Exception as e:
                print(f"Error processing task: {e}")

        return all_responses

# Process all gene symbols
all_gene_responses = process_genes(genes.index.tolist())


def flatten_data(gene_responses):
    rows = []
    for idx, (gene_symbol, gene_info) in enumerate(gene_responses.items()):
        
        row = {
            'id': gene_info.get('id'),
            'gene_symbol': gene_symbol,
            'gene_biotype': gene_info.get('biotype'),
            'n_cells': genes[genes.index == gene_symbol].n_cells.values[0] if gene_symbol in genes.index else None,
        }
           
        rows.append(row)
    return pd.DataFrame(rows)

# Process the collected gene responses into a DataFrame
gene_df = flatten_data(all_gene_responses)
gene_df.to_csv(data_sources['path'] + '/gene_to_protein.csv', index=False)
