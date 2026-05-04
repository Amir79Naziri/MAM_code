# MAM: Multinomial Attention Masking

**Multinomial Attention Masking (MAM)** is an intelligent, content-aware masking module designed to optimize the training of large-scale AI models on single-cell genomics (**scRNA-seq**) data. It replaces traditional uniform random masking with a biologically informed selection process.

## The Problem: Sparsity and Noise
In natural language processing, most words carry meaning. In scRNA-seq data, however, up to **90% of the matrix consists of zeros** (due to technical "dropout" or gene inactivity). 

Standard random masking frequently forces a model to spend its "brainpower" predicting that a zero is a zero—a trivial task that ignores the complex biological signals hidden in the remaining 10% of the data. This leads to inefficient training and, on large datasets, can cause the model's internal representations to collapse.

---

## How MAM Works
MAM shifts the training focus from random noise to **biological signal** through a three-stage process.

### 1. The Latent Attention "Scan"
MAM introduces a set of **learnable latent vectors** (acting as "expert biological scanners"). During training, these latents perform a **cross-attention** operation over the cell's gene expression profile. Instead of treating all genes as equally important, the latents identify which genes are currently expressing "high-signal" information relevant to that specific cell's state.



### 2. Importance-Based Masking
The module aggregates the attention weights into an **Importance Score** for every gene. 
*   **Weighted Selection:** Rather than picking the top-$k$ genes, MAM uses **Multinomial Sampling**. 
*   **Smart Challenge:** Genes with high biological significance are much more likely to be masked. This forces the model to solve "harder" problems—reconstructing critical markers of cell identity—resulting in a much deeper understanding of the underlying biology.

### 3. Contextual Injection
MAM doesn't just decide what to hide; it also creates a **global summary** of the cell it just scanned. This summary is "injected" back into the main backbone of the foundation model via lightweight **adapters**. This acts as a constant "hint" to the model, providing a stable biological context that prevents representations from drifting or collapsing during the intensive pretraining phase.
