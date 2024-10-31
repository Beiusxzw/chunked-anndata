# Tensorstore Extension for AnnData 

This extension provides support for reading and writing [AnnData](https://anndata.readthedocs.io/en/latest/) objects using the [Tensorstore](https://google.github.io/tensorstore/)

## Design Goals

- **The challenge**: Large single-cell anndata object are often too large to fit in [RAM](https://en.wikipedia.org/wiki/Random-access_memory) and save in disk. 
- **The solution**: [Tensorstore](https://google.github.io/tensorstore/) provides a way to read and write data in a variety of formats, including [Zarr](https://zarr.dev/). This extension provides a way to read anndata objects with specific rows (cells) and columns (genes) from a Zarr store. **You will not need to load the entire anndata object into memory to access a subset of the data.**
- **Caveats**: This extension is still in development and may not support all features of AnnData objects.
- **Caveats**: This extension is not optimized for read/write speed.



## Installation

```bash
pip install anndata-tensorstore
```

## Usage

### Writing an AnnData object to a Tensorstore

```python
import anndata
import anndata_tensorstore as ats

anndata = anndata.read_h5ad("path/to/large_anndata.h5ad")
ats.save(anndata, "path/to/large_anndata.ats", is_raw_count=True)
```


### Reading an AnnData object from a Tensorstore

```python
import os
import anndata
import anndata_tensorstore as ats

# Load the entire data from the storage
adata = ats.load("path/to/large_anndata.ats")

# Load the anndata object from the storage, specifying the rows and columns to load
var = pd.read_parquet(os.path.join("path/to/large_anndata.ats", ats.ATS_FILE_NAME.var))
obs = pd.read_parquet(os.path.join("path/to/large_anndata.ats", ats.ATS_FILE_NAME.obs))

# Option 1: Load the partial data from the storage, specifying the rows and columns to load
adata = ats.load(
    "path/to/large_anndata.ats",
    obs_indices=slice(0, 1000),                     # the specification of columns and rows can either be
    var_indices=var.index.isin(["gene1", "gene2"])  # a slice object or a boolean array
)

# Option 2: Load the partial data from the storage, specifying the rows and columns to load
adata = ats.load(
    "path/to/large_anndata.ats",
    obs_names=['barcode1','barcode2'],    # the specification of columns and rows can either be
    var_names=["gene1", "gene2"]          # the index names of the obs and var dataframes
)

# Option 3: Load the partial data from the storage, specifying the rows and columns to load
adata = ats.load(
    "path/to/large_anndata.ats",
    obs_indices=[0, 1],    # the specification of columns and rows can either be
    var_indices=[0, 1]     # a list of indices
)

# Option 4: Load the partial data from the storage, specifying the rows and columns to load
adata = ats.load(
    "path/to/large_anndata.ats",
    obs_selection=[("obs_column_name", ["cell_type_1", "cell_type_2"])],
    var_selection=[("var_column_name", ["gene1", "gene2"])]
    # the specification of columns and rows can either be
    # a list of tuples where the first element is the column name and the 
    # second element is a list of values
)



```

## Development and Future Work

- [ ] reduce storage size
- [ ] support more AnnData features
- [ ] support more tensorstore features
