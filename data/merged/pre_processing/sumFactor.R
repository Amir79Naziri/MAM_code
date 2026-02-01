library(Matrix)
library(scran)
library(SingleCellExperiment)

# Load the matrix from the .mtx file
data_mat <- readMM("../data/matrix.mtx")

# Ensure the matrix is treated as sparse
data_mat <- as(data_mat, "CsparseMatrix")

# Create a SingleCellExperiment object
sce <- SingleCellExperiment(assays = list(counts = data_mat))

# Load group assignments from .csv file
input_groups <- read.csv("../data/input_groups.csv", header = TRUE, stringsAsFactors = FALSE)
# Ensure the group data is a factor if it's not
input_groups <- factor(input_groups$groups)

# Store groups in the SingleCellExperiment object
colData(sce)$groups = input_groups

print('started...')
# Compute size factors
size_factors = computeSumFactors(sce, clusters = colData(sce)$groups, min.mean = 0.1)
print(size_factors)

# Ensure size_factors is a numeric vector
# size_factors_vector <- as.numeric(size_factors)

# # Convert to data frame for saving
# size_factors_df <- data.frame(size_factor = size_factors_vector)

# # Write to CSV
writeRDS(size_factors, "../data/size_factors.rds")

