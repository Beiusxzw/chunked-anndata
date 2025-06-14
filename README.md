# Chunked Saving/Loading AnnData 

This extension provides support for reading and writing [AnnData](https://anndata.readthedocs.io/en/latest/) objects.

## Design Goals

- **The challenge**: Large single-cell anndata object are often too large to fit in [RAM](https://en.wikipedia.org/wiki/Random-access_memory) and save in disk. 
- **The solution**: [Tensorstore](https://google.github.io/tensorstore/) provides a way to read and write data in a variety of formats, including [Zarr](https://zarr.dev/). This extension provides a way to read anndata objects with specific rows (cells) and columns (genes) from a Zarr store. **You will not need to load the entire anndata object into memory to access a subset of the data.**
- **Caveats**: This extension is still in development and may not support all features of AnnData objects, but it is designed to be efficient for large datasets.
- **Caveats**: This extension is not optimized for read/write speed.


## ![anndata_tensorstor](./docs/source/_static/images/anndata_tensorstor.jpeg)Installation

```bash
pip install chunked_anndata
```

## Usage

### Writing an AnnData object to a chunked anndata storage

```python
import anndata
import chunked_anndata

adata = anndata.read_h5ad("path/to/large_anndata.h5ad")
chunked_anndata.save(adata, "path/to/large_anndata.ats", is_raw_count=True)
```


### Reading an AnnData object from a chunked anndata storage

```python
import os
import anndata
import chunked_anndata

# Load the entire data from the storage
adata = chunked_anndata.load("path/to/large_anndata.ats")

# Load the anndata object from the storage, specifying the rows and columns to load
var = pd.read_parquet(os.path.join("path/to/large_anndata.ats", chunked_anndata.ATS_FILE_NAME.var))
obs = pd.read_parquet(os.path.join("path/to/large_anndata.ats", chunked_anndata.ATS_FILE_NAME.obs))

# Option 1: Load the partial data from the storage, specifying the rows and columns to load
adata = chunked_anndata.load(
    "path/to/large_anndata.ats",
    obs_indices=slice(0, 1000),                     # the specification of columns and rows can either be
    var_indices=var.index.isin(["gene1", "gene2"])  # a slice object or a boolean array
)

# Option 2: Load the partial data from the storage, specifying the rows and columns to load
adata = chunked_anndata.load(
    "path/to/large_anndata.ats",
    obs_names=['barcode1','barcode2'],    # the specification of columns and rows can either be
    var_names=["gene1", "gene2"]          # the index names of the obs and var dataframes
)

# Option 3: Load the partial data from the storage, specifying the rows and columns to load
adata = chunked_anndata.load(
    "path/to/large_anndata.ats",
    obs_indices=[0, 1],    # the specification of columns and rows can either be
    var_indices=[0, 1]     # a list of indices
)

# Option 4: Load the partial data from the storage, specifying the rows and columns to load
adata = chunked_anndata.load(
    "path/to/large_anndata.ats",
    obs_selection=[("obs_column_name", ["cell_type_1", "cell_type_2"])],
    var_selection=[("var_column_name", ["gene1", "gene2"])]
    # the specification of columns and rows can either be
    # a list of tuples where the first element is the column name and the 
    # second element is a list of values
)
```

### Concatenating AnnData object to an existing Tensorstore

If you have a new AnnData object that you want to add to an existing chunked anndata storage, you can use the `concat` function. This is useful for appending new data without needing to rewrite the entire dataset. This is especially useful for large datasets where you want to incrementally add new data.

```python
import chunked_anndata
import anndata

adata_new = anndata.read_h5ad("path/to/new_anndata.h5ad")
chunked_anndata.concat(
    adata_new,
    "path/to/large_anndata.ats",
    is_raw_count=True
)

```


## Development and Future Work

- [ ] reduce storage size
- [ ] support more AnnData features
- [ ] support more tensorstore features
