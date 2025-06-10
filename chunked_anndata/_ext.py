import os
import tensorstore as ts
import pandas as pd
import numpy as np
import scipy
import pickle
from anndata import AnnData
import json
from typing import List, Union, Tuple, Optional
import geopandas
# import multiscale_spatial_image

import tqdm
from abc import ABC, ABCMeta
from enum import Enum, EnumMeta, unique
from functools import wraps
from typing import Any, Callable

import xarray as xr
from spatial_image import to_spatial_image

from ._utils import typed
from ._version import version as package_version


class PrettyEnum(Enum):
    """Enum with a pretty :meth:`__str__` and :meth:`__repr__`."""

    @property
    def v(self) -> Any:
        """Alias for :attr`value`."""
        return self.value

    def __repr__(self) -> str:
        return f"{self.value!r}"

    def __str__(self) -> str:
        return f"{self.value!s}"

class ModeEnum(str, PrettyEnum, metaclass=EnumMeta):
    """Enum with a pretty :meth:`__str__` and :meth:`__repr__`."""

@unique
class ATS_FILE_NAME(ModeEnum):
    X = 'X'
    obs = 'obs.parquet'
    var = 'var.parquet'
    obsm = 'obsm'
    varm = 'varm'
    uns = 'uns'
    layers = 'layers'
    raw = 'raw'
    config = 'storage.config'


@unique 
class DTYPE(PrettyEnum):
    # check for edianess
    float16 = np.dtype(np.float16).str
    float32 = np.dtype(np.float32).str
    float64 = np.dtype(np.float64).str
    int8 = np.dtype(np.int8).str
    int16 = np.dtype(np.int16).str
    int32 = np.dtype(np.int32).str
    int64 = np.dtype(np.int64).str
    uint8 = np.dtype(np.uint8).str
    uint16 = np.dtype(np.uint16).str
    uint32 = np.dtype(np.uint32).str
    uint64 = np.dtype(np.uint64).str


def check_input_path(input_path):
    if input_path.startswith("./"):
        input_path = os.path.abspath(input_path)
    if input_path.endswith("/"):
        input_path = input_path[:-1]
    return input_path

def save_X(
    X: Union[scipy.sparse.spmatrix, np.ndarray], 
    output_path: Union[str, os.PathLike], 
    chunk_size: int = 1024, 
    dtype: DTYPE = None,
):
    """
    Save a matrix to a tensorstore.

    :param X: The matrix to save.
    :param output_path: The path to the tensorstore.
    :param chunk_size: The chunk size to write.
    :param dtype: The data type to save.
    """

    dataset = ts.open({
        'driver': 'zarr',
        'kvstore': {
            'driver': 'file',
            'path': output_path,
        },
        'metadata': {
            'dtype': X.dtype.str if dtype is None else dtype,
            'shape': X.shape
        },
    }, create=True, delete_existing=True).result()
    if scipy.sparse.issparse(X):
        for e in range(0,X.shape[0],chunk_size):
            towrite = X[e:e+chunk_size].toarray()
            if dtype is not None:
                towrite = towrite.astype(dtype)
            write_future = dataset[e:min(X.shape[0], e+chunk_size), :].write(towrite)
            write_result = write_future.result()
    else:
        write_future = dataset.write(X)
        write_result = write_future.result()

def concat_X(
    X2: Union[scipy.sparse.spmatrix, np.ndarray], 
    output_path: Union[str, os.PathLike],
    chunk_size: int = 100,
    dim: Optional[int] = None
):
    dataset = ts.open({
        'driver': 'zarr',
        'kvstore': {
            'driver': 'file',
            'path': output_path,
        }
    }).result()

    if (dim is not None and dim == 0) or (dataset.shape[0] == X2.shape[0] and dataset.shape[1] != X2.shape[1]):
        # Determine the original and new shapes
        original_rows = dataset.shape[0]
        new_rows = original_rows + X2.shape[0]
        new_shape = (new_rows, dataset.shape[1])
        
        print(f"Resizing dataset from {dataset.shape} to {new_shape}")

        # Resize the dataset to accommodate additional rows
        dataset = dataset.resize(exclusive_max=new_shape).result()

        # Write X2 to the expanded portion of the dataset
        if scipy.sparse.issparse(X2):
            for e in range(0, X2.shape[0], chunk_size):
                write_future = dataset[original_rows + e : original_rows + e + chunk_size, :].write(X2[e : e + chunk_size].toarray())
                write_result = write_future.result()
        else:
            write_future = dataset[original_rows:new_rows, :].write(X2)
            write_result = write_future.result()
            
    elif (dim is not None and dim == 1) or (dataset.shape[0] != X2.shape[0] and dataset.shape[1] == X2.shape[1]):
        # Determine the original and new shapes
        original_cols = dataset.shape[1]
        new_cols = original_cols + X2.shape[1]
        new_shape = (dataset.shape[0], new_cols)

        print(f"Resizing dataset from {dataset.shape} to {new_shape}")
        # Resize the dataset to accommodate additional columns
        dataset = dataset.resize(exclusive_max=new_shape).result()

        # Write X2 to the expanded portion of the dataset
        if scipy.sparse.issparse(X2):
            for e in range(0, X2.shape[1], chunk_size):
                write_future = dataset[:, original_cols + e : original_cols + e + chunk_size].write(X2[:, e : e + chunk_size].toarray())
                write_result = write_future.result()
        else:
            write_future = dataset[:, original_cols:new_cols].write(X2)
            write_result = write_future.result()
    else:
        raise ValueError("The shapes of the two matrices do not match or dim is not specified")

def load_X(
    input_path: Union[str, os.PathLike], 
    obs_indices: Union[slice, np.ndarray] = None,
    var_indices: Union[slice, np.ndarray] = None,
    show_progress: bool = False,
    chunk_size: int = 1024,
    to_sparse: bool = True,
    sparse_format: Callable = scipy.sparse.csr_matrix
) -> Union[np.ndarray, scipy.sparse.spmatrix]:
    """
    Load a matrix from a tensorstore.

    :param input_path: The path to the tensorstore.
    :param obs_indices: The row indices to load.
    :param var_indices: The column indices to load.
    :param show_progress: Whether to show a progress bar.
    :param chunk_size: The chunk size to read.
    :param to_sparse: Whether to return a sparse matrix.
    :param sparse_format: The sparse matrix format to use
    
    :return: The matrix.
    """
    input_path = check_input_path(input_path)
    dataset = ts.open({
        'driver': 'zarr',
        'kvstore': {
            'driver': 'file',
            'path': input_path,
        }
    }, create=False).result()
    
    if obs_indices is None and var_indices is None:
        handler = dataset
    elif obs_indices is not None and var_indices is None:
        handler = dataset[obs_indices, :]
    elif obs_indices is None and var_indices is not None:
        handler = dataset[:, var_indices]
    else:
        handler = dataset[obs_indices, :][:, var_indices]

    Xs = []
    if show_progress:
        pbar = tqdm.tqdm(
            total=handler.shape[0],
            desc="Loading data", 
            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
            position=0, 
            leave=True
        )
    for e in range(0, handler.shape[0], chunk_size):
        x = handler[e:min(handler.shape[0], e+chunk_size)].read().result()
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        if to_sparse:
            Xs.append(sparse_format(x))
        else:
            Xs.append(x)
        if show_progress:
            pbar.update(chunk_size)
    if show_progress:
        pbar.close()
    if to_sparse:
        X = scipy.sparse.vstack(Xs)
    else:
        X = np.vstack(Xs)
    return X

def save_np_array_to_tensorstore(
    Z: np.ndarray, 
    output_path: Union[str, os.PathLike]
):
    """
    Save a numpy array to a tensorstore.

    :param Z: The numpy array to save.
    :param output_path: The path to the tensorstore.
    """
    dataset = ts.open({
        'driver': 'zarr',
        'kvstore': {
            'driver': 'file',
            'path': output_path,
        },
        'metadata': {
            'dtype': Z.dtype.str,
            'shape': Z.shape
        },
    }, create=True).result()
    
    write_future = dataset.write(Z)
    write_result = write_future.result()


def load_np_array_from_tensorstore(
    input_path: Union[str, os.PathLike], 
    obs_indices: Union[slice, np.ndarray] = None
) -> np.ndarray:
    """
    Load a numpy array from a tensorstore.

    :param input_path: The path to the tensorstore.
    :param obs_indices: The row indices to load.
    """
    input_path = check_input_path(input_path)
    dataset = ts.open({
        'driver': 'zarr',
        'kvstore': {
            'driver': 'file',
            'path': input_path,
        }
    }, create=False).result()

    if obs_indices is None:
        return dataset.read().result()
    else:
        return dataset[obs_indices].read().result()

def save_csr_matrix_to_tensorstore(
    X: scipy.sparse.csr_matrix,
    output_path: Union[str, os.PathLike],
):
    save_X(X.indices, os.path.join(output_path, 'indices'))
    save_X(X.indptr, os.path.join(output_path, 'indptr'))
    save_X(X.data, os.path.join(output_path, 'data'))
    save_X(np.array(X.shape), os.path.join(output_path, 'shape'))
    
def load_csr_matrix_from_tensorstore(
    input_path: Union[str, os.PathLike],
):
    indices = load_np_array_from_tensorstore(os.path.join(input_path, 'indices'))
    indptr = load_np_array_from_tensorstore(os.path.join(input_path, 'indptr'))
    data = load_np_array_from_tensorstore(os.path.join(input_path, 'data'))
    shape = load_np_array_from_tensorstore(os.path.join(input_path, 'shape'))
    return scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)

def save_chunked_sparse_matrix_to_tensorstore(
    X: scipy.sparse.csr_matrix,
    output_path: Union[str, os.PathLike],
    chunk_size: int = 32768,
    dtype: Optional[np.dtype] = None
):
    for e in range(0, X.shape[0], chunk_size):
        towrite = X[e:e+chunk_size]
        if dtype is not None:
            towrite.data = towrite.data.astype(dtype)
        save_csr_matrix_to_tensorstore(towrite, os.path.join(output_path, f'{e // chunk_size}'))
        
def load_chunked_sparse_matrix_from_tensorstore(
    input_path: Union[str, os.PathLike],
    obs_indices: Optional[Union[slice, np.ndarray]] = None,
    var_indices: Optional[Union[slice, np.ndarray]] = None,
    chunk_size: int = 32768,
):
    Xs = []
    es = np.array(sorted(list(map(int, os.listdir(input_path)))))
    assert np.all(es == np.arange(es[0], es[-1] + 1))
    
    if var_indices is not None:
        if isinstance(var_indices, slice):
            var_indices = np.arange(var_indices.start, var_indices.stop)
        elif var_indices.dtype == bool:
            var_indices = np.where(var_indices)[0]
        var_indices = np.sort(var_indices)
        
    if obs_indices is not None:
        if isinstance(obs_indices, slice):
            obs_indices = np.arange(obs_indices.start, obs_indices.stop)
        elif obs_indices.dtype == bool:
            obs_indices = np.where(obs_indices)[0]

        obs_indices = np.sort(obs_indices)
        obs_indices_to_chunk = obs_indices // chunk_size
        obs_indices_to_chunk_grouped = {
            e: obs_indices[obs_indices_to_chunk == e] - (e * chunk_size) for e in np.unique(obs_indices_to_chunk)
        }
    
        for e, indices in obs_indices_to_chunk_grouped.items():
            X = load_csr_matrix_from_tensorstore(os.path.join(input_path, f'{e}'))[indices, :]
            if var_indices is not None:
                X = X[:, var_indices]
            Xs.append(X)
    else:
        for e in es:
            X = load_csr_matrix_from_tensorstore(os.path.join(input_path, f'{e}'))
            if var_indices is not None: 
                X = X[:, var_indices]
            Xs.append(X)
    
    return scipy.sparse.vstack(Xs)  
    
def concat_chunked_sparse_matrix_to_tensorstore(
    X2: scipy.sparse.csr_matrix,
    output_path: Union[str, os.PathLike],
    chunk_size: int = 32768,
    dtype: Optional[np.dtype] = None
):
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Output path {output_path} does not exist.")
    if not os.path.exists(os.path.join(output_path, '0')):
        raise FileNotFoundError(f"Output path {output_path} does not contain any data.")
    es = np.array(sorted(list(map(int, os.listdir(output_path)))))
    tail = es[-1]
    X1 = load_csr_matrix_from_tensorstore(
        os.path.join(output_path, f'{tail}'),
    )
    if X1.shape[1] != X2.shape[1]:
        raise ValueError(f"Number of columns in {output_path} ({X1.shape[1]}) does not match number of columns in X2 ({X2.shape[1]}).")
    if X1.shape[0] > chunk_size:
        raise ValueError(f"Number of rows in {output_path} ({X1.shape[0]}) is greater than chunk size ({chunk_size}).")
    
    if X1.shape[0] < chunk_size:
        new_X1 = scipy.sparse.vstack([X1, X2[:chunk_size - X1.shape[0], :]])
        save_csr_matrix_to_tensorstore(
            new_X1,
            os.path.join(output_path, f'{tail}'),
        )
        start = chunk_size - X1.shape[0]
    else:
        start = 0
    for e in range(start, X2.shape[0], chunk_size):
        towrite = X2[e:e+chunk_size]
        if dtype is not None:
            towrite.data = towrite.data.astype(dtype)
        save_csr_matrix_to_tensorstore(towrite, os.path.join(output_path, f'{tail + 1 + e // chunk_size}'))
    

def check_is_parquet_serializable(obj: pd.DataFrame):
    for c in obj.columns:
        if obj[c].dtype == 'O':
            obj[c] = obj[c].astype(str)
        elif obj[c].dtype == 'category':
            obj[c] = obj[c].astype(str)


def save_anndata_to_tensorstore(
    adata: AnnData,
    output_path: str,
    chunk_size: Optional[int] = None,
    is_raw_count: bool = False,
    sparse_storage: bool = True
):
    """
    Save an AnnData object to a tensorstore.

    :param adata: The AnnData object to save.
    :param output_path: The path to the tensorstore.
    :param is_raw_count: Whether the AnnData object is raw count data. 
                         If True, the raw count matrix will be saved as uint32 for memory efficiency.
    :param sparse_storage: Whether to save the matrix into sparse format.

    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    if chunk_size is None:
        if adata.shape[0] > 1024 and adata.shape[0] < 2 ** 14:
            chunk_size = 1024
        elif adata.shape[0] >= 2 ** 14 and adata.shape[0] < 2 ** 16:
            chunk_size = 2048
        elif adata.shape[0] >= 2 ** 16 and adata.shape[0] < 2 ** 18:
            chunk_size = 4096
        elif adata.shape[0] >= 2 ** 18 and adata.shape[0] < 2 ** 20:
            chunk_size = 8192
        elif adata.shape[0] >= 2 ** 20:
            chunk_size = 16384
            
    config = dict(
        version=package_version,
        sparse_storage=sparse_storage,
        chunk_size=chunk_size,
        obs_is_gpd=isinstance(adata.obs, geopandas.GeoDataFrame) if hasattr(adata, 'obs') else False
    )
    with open(os.path.join(output_path, 'storage.config'),'w+') as f:
        json.dump(config, f)
    
    if hasattr(adata, 'X') and adata.X is not None:
        if sparse_storage:
            save_chunked_sparse_matrix_to_tensorstore(
                adata.X,
                os.path.join(output_path, 'X'),
                chunk_size = chunk_size,    
                dtype=DTYPE.uint32 if is_raw_count else None
            )
        else:
            save_X(adata.X, os.path.join(output_path, 'X'), dtype=DTYPE.uint32 if is_raw_count else None)

    if hasattr(adata, 'obs') and adata.obs is not None:
        check_is_parquet_serializable(adata.obs)
        adata.obs.to_parquet(os.path.join(output_path, 'obs.parquet'))

    if hasattr(adata, 'var') and adata.var is not None:
        check_is_parquet_serializable(adata.var)
        adata.var.to_parquet(os.path.join(output_path, 'var.parquet'))

    if hasattr(adata, 'obsm') and adata.obsm is not None:
        for k, v in adata.obsm.items():
            save_np_array_to_tensorstore(v, os.path.join(output_path, f'obsm/{k}'))

    if hasattr(adata, 'varm') and adata.varm is not None:
        for k, v in adata.varm.items():
            save_np_array_to_tensorstore(v, os.path.join(output_path, f'varm/{k}'))

    if hasattr(adata, 'obsp') and adata.obsp is not None:
        for k,v in adata.obsp.items():
            if scipy.sparse.issparse(v):
                save_csr_matrix_to_tensorstore(v, os.path.join(output_path, f'obsp/{k}'))
            else:
                save_np_array_to_tensorstore(v, os.path.join(output_path, f'obsp/{k}'))

    if not os.path.exists(os.path.join(output_path, 'uns')):
        os.makedirs(os.path.join(output_path, 'uns'))

    if hasattr(adata, 'uns') and adata.uns is not None:
        for k, v in adata.uns.items():
            if isinstance(v, pd.DataFrame):
                check_is_parquet_serializable(v)
                v.to_parquet(os.path.join(output_path, f'uns/{k}.parquet'))
            if k == 'spatial':
                os.mkdirs(os.path.join(output_path, 'spatial'))
                spatial_id = list(v.keys())[0]
                if 'images' in v[spatial_id].keys():
                    for img_id, img in v[spatial_id]['images'].items():
                        save_spatial_image(img, os.path.join(output_path, f'spatial/images/{img_id}.zarr'))
                if 'scalefactors' in v[spatial_id].keys():
                    with open(os.path.join(output_path, 'spatial/scalefactors.json'), 'w+') as f:
                        json.dump(v[spatial_id]['scalefactors'], f)
            else:
                with open(os.path.join(output_path, f'uns/{k}.pickle'), 'wb') as f:
                    pickle.dump(v, f)

    if adata.layers is not None:
        for k, v in adata.layers.items():
            if sparse_storage:
                save_chunked_sparse_matrix_to_tensorstore(
                    v, 
                    os.path.join(output_path, f'layers/{k}'),
                    chunk_size = chunk_size
                )
            else:
                save_X(v, os.path.join(output_path, f'layers/{k}'))

    if adata.raw is not None:
        save_anndata_to_tensorstore(adata.raw, os.path.join(output_path, 'raw'))


def load_anndata_from_tensorstore(
    input_path: str, 
    obs_indices: Optional[Union[slice, np.ndarray]] = None,
    var_indices: Optional[Union[slice, np.ndarray]] = None,
    obs_selection: Optional[List[Tuple[str, Any]]] = None,
    var_selection: Optional[List[Tuple[str, Any]]] = None,
    obs_names: Optional[List[str]] = None,
    var_names: Optional[List[str]] = None,
    sparse_storage: bool = True,
    as_sparse: bool = True,
    chunk_size: int = 1024,
    load_layers_as_X: Optional[str] = None,
    obs_is_gpd: bool = False
):
    """
    Load an AnnData object from a tensorstore.

    :param input_path: The path to the tensorstore.
    :param obs_indices: The row indices to load.
    :param var_indices: The column indices to load.
    :param obs_selection: The row selection to load.
    :param var_selection: The column selection to load.
    :param obs_names: The observation names to load.
    :param var_names: The variable names to load.
    :param sparse_storage: Whether the matrix is stored in sparse format.
    :param as_sparse: Whether to return the matrix as a sparse matrix.
    :param chunk_size: The chunk size to read.
    :param load_layers_as_X: The layer to load as the X matrix. 
        If specified, the X matrix will be loaded from the layers, and other layers will be ignored.
    :param obs_is_gpd: Whether the observation is a GeoPandas DataFrame.
    """

    if os.path.exists(os.path.join(input_path, 'storage.config')):
        with open(os.path.join(input_path, 'storage.config')) as f:
            config = json.load(f)
        if 'sparse_storage' in config.keys():
            sparse_storage = config['sparse_storage']
        if 'chunk_size' in config.keys():
            chunk_size = config['chunk_size']
        if 'obs_is_gpd' in config.keys():
            obs_is_gpd = config['obs_is_gpd']
    
    if obs_names is not None:
        if obs_is_gpd:
            obs = geopandas.read_parquet(os.path.join(input_path, 'obs.parquet'))
        else:
            obs = pd.read_parquet(os.path.join(input_path, 'obs.parquet'))
        if obs_indices is not None:
            print("Warning: obs_names will override obs_indices")
        obs_indices = obs.index.isin(obs_names)

    if var_names is not None:
        var = pd.read_parquet(os.path.join(input_path, 'var.parquet'))
        if var_indices is not None:
            print("Warning: var_names will override var_indices")
        var_indices = var.index.isin(var_names)
    
    if var_indices is None and var_selection is not None:
        var_indices = np.zeros(_X.shape[1], dtype=bool)
        for k, v in var_selection:
            if isinstance(v, list):
                var_indices = var_indices | np.array(pd.Series(var[k]).isin(v))
            else:
                var_indices = var_indices | np.array((var[k] == v).values)


    if obs_indices is None and obs_selection is not None:
        obs = pd.read_parquet(os.path.join(input_path, 'obs.parquet'))
        obs_indices = np.zeros(obs.shape[0], dtype=bool)
        for k, v in obs_selection:
            if isinstance(v, list):
                obs_indices = obs_indices | np.array(pd.Series(obs[k]).isin(v))
            else:
                obs_indices = obs_indices | np.array((obs[k] == v).values)
    
    if load_layers_as_X is not None:
        if sparse_storage:
            _X = load_chunked_sparse_matrix_from_tensorstore(
                os.path.join(input_path, 'layers', load_layers_as_X), 
                obs_indices, 
                var_indices,
                chunk_size = chunk_size
            )
        else:
            _X = load_X(os.path.join(input_path, 'layers', load_layers_as_X), obs_indices, var_indices)
        if as_sparse:
            _X = scipy.sparse.csr_matrix(_X)
    else:
        if sparse_storage:
            _X = load_chunked_sparse_matrix_from_tensorstore(
                os.path.join(input_path, 'X'), 
                obs_indices, 
                var_indices,
                chunk_size = chunk_size
            )
        else:
            _X = load_X(os.path.join(input_path, 'X'), obs_indices, var_indices)
        if as_sparse:
            _X = scipy.sparse.csr_matrix(_X)

    if os.path.exists(os.path.join(input_path, 'obs.parquet')):
        obs = pd.read_parquet(os.path.join(input_path, 'obs.parquet'))
        # if obs_indices is a slice
        if isinstance(obs_indices, slice):
            _obs = obs.iloc[obs_indices]
        # if obs_indices is a array
        elif isinstance(obs_indices, np.ndarray):
            # if obs_indices is a boolean array
            if obs_indices.dtype == bool:
                _obs = obs.loc[obs_indices]
            elif obs_indices.dtype ==  int:
                _obs = obs.iloc[obs_indices]
            else:
                raise ValueError(f'Invalid obs_indices of type {type(obs_indices)} with value {type(obs_indices)}')
        elif isinstance(obs_indices, list):
            if isinstance(obs_indices[0], str) or isinstance(obs_indices[0], bool):
                _obs = obs.loc[obs_indices]
            elif isinstance(obs_indices[0], int):
                _obs = obs.iloc[obs_indices]
            else:
                raise ValueError(f'Invalid obs_indices of type {type(obs_indices[0])} with value {type(obs_indices[0])}')
        elif obs_indices is None:
            _obs = obs
        else:
            raise ValueError(f'Invalid obs_indices of type {type(obs_indices)}')
    
    if os.path.exists(os.path.join(input_path, 'var.parquet')):
        var = pd.read_parquet(os.path.join(input_path, 'var.parquet'))
        # if var_indices is a slice
        if isinstance(var_indices, slice):
            _var = var.iloc[var_indices]
        # if var_indices is a array
        elif isinstance(var_indices, np.ndarray):
            # if var_indices is a boolean array
            if var_indices.dtype == bool:
                _var = var.loc[var_indices]
            elif var_indices.dtype ==  int:
                _var = var.iloc[var_indices]
            else:
                raise ValueError(f'Invalid var_indices of type {type(var_indices)} with value {type(var_indices)}')
        elif isinstance(var_indices, list):
            if isinstance(var_indices[0], str) or isinstance(var_indices[0], bool):
                _var = var.loc[var_indices]
            elif isinstance(var_indices[0], int):
                _var = var.iloc[var_indices]
            else:
                raise ValueError(f'Invalid var_indices of type {type(var_indices[0])} with value {type(var_indices[0])}')
        elif var_indices is None:
            _var = var
        else:
            raise ValueError(f'Invalid var_indices of type {type(var_indices)}')
        
    _obsm = None
    if os.path.exists(os.path.join(input_path, 'obsm')):
        _obsm = {}
        for f in os.listdir(os.path.join(input_path, 'obsm')):
            _obsm[f] = load_np_array_from_tensorstore(os.path.join(input_path, 'obsm', f), obs_indices)

    _varm = None
    if os.path.exists(os.path.join(input_path, 'varm')):
        _varm = {}
        for f in os.listdir(os.path.join(input_path, 'varm')):
            _varm[f] = load_np_array_from_tensorstore(os.path.join(input_path, 'varm', f), var_indices)

    _obsp = None
    if os.path.exists(os.path.join(input_path, 'obsp')):
        _obsp = {}
        for f in os.listdir(os.path.join(input_path, 'obsp')):
            try:
                _obsp[f] = load_csr_matrix_from_tensorstore(os.path.join(input_path, 'obsp', f))
            except:
                _obsp[f] = load_np_array_from_tensorstore(os.path.join(input_path, 'obsp', f))
            if obs_indices is not None:
                _obsp[f] = _obsp[f][obs_indices, :][:, obs_indices]        

    _uns = None
    if os.path.exists(os.path.join(input_path, 'uns')):
        _uns = {}
        for fname in os.listdir(os.path.join(input_path, 'uns')):
            if fname.endswith('.parquet'):
                _uns[fname[:-8]] = pd.read_parquet(os.path.join(input_path, 'uns', fname))
            elif fname.endswith('.pickle'):
                with open(os.path.join(input_path, 'uns', fname), 'rb') as f:
                    _uns[fname] = pickle.load(f)

    _layers = None
    if load_layers_as_X is None:
        if os.path.exists(os.path.join(input_path, 'layers')):
            _layers = {}
            for f in os.listdir(os.path.join(input_path, 'layers')):
                if sparse_storage:
                    _layers[f] = load_chunked_sparse_matrix_from_tensorstore(os.path.join(input_path, 'layers', f), obs_indices, var_indices)
                else:
                    _layers[f] = load_X(os.path.join(input_path, 'layers', f), obs_indices, var_indices)
                if as_sparse:
                    _layers[f] = scipy.sparse.csr_matrix(_layers[f])
    
    adata = AnnData(
        X=_X,
        obs=_obs,
        var=_var,
        obsm=_obsm,
        varm=_varm,
        uns=_uns,
        layers=_layers
    )
    if os.path.exists(os.path.join(input_path, 'raw')):
        adata.raw = load_anndata_from_tensorstore(
            os.path.join(input_path, 'raw'), 
            obs_indices, 
            var_indices
        )

    return adata

def concat_anndata_to_tensorstore(
    adata2: AnnData,
    output_path: str,
    chunk_size: Optional[int] = None,
    is_raw_count: bool = False,
    sparse_storage: bool = True,
    obs_join: str = 'outer',
):
    """
    Concatenate an AnnData object to an existing tensorstore, without loading 

    :param adata2: The AnnData object to concatenate.
    :param output_path: The path to the tensorstore.
    :param chunk_size: The chunk size to write.
    :param is_raw_count: Whether the AnnData object is raw count data. 
                         If True, the raw count matrix will be saved as uint32 for memory efficiency.
    :param sparse_storage: Whether to save the matrix into sparse format.
    :param obs_join: The type of join to use for the observation metadata (adata.obs)

    :note: This function overwrite the existing data in the tensorstore.
    """
    
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Output path {output_path} does not exist.")
    
    if not os.path.exists(os.path.join(output_path, 'storage.config')):
        raise FileNotFoundError(f"Output path {output_path} does not contain any data.")
    
    with open(os.path.join(output_path, 'storage.config')) as f:
        config = json.load(f)

    if 'sparse_storage' in config.keys():
        sparse_storage = config['sparse_storage']

    if 'chunk_size' in config.keys():
        chunk_size = config['chunk_size']

    var2 = adata2.var
    obs2 = adata2.obs

    if os.path.exists(os.path.join(output_path, 'var.parquet')):
        var1 = pd.read_parquet(os.path.join(output_path, 'var.parquet'))

        if var1.columns.tolist() != var2.columns.tolist():
            raise ValueError(f"Variable names in {output_path} do not match variable names in adata2.")
        

    if os.path.exists(os.path.join(output_path, 'obs.parquet')):
        if 'obs_is_gpd' in config.keys():
            obs_is_gpd = config['obs_is_gpd']

        if obs_is_gpd:
            obs1 = geopandas.read_parquet(os.path.join(output_path, 'obs.parquet'))
        else:
            obs1 = pd.read_parquet(os.path.join(output_path, 'obs.parquet'))
        new_obs = pd.concat([obs1, obs2], ignore_index=True, join=obs_join)
        check_is_parquet_serializable(new_obs)
        new_obs.to_parquet(os.path.join(output_path, 'obs.parquet'))

    
    if hasattr(adata2, 'X') and adata2.X is not None:
        if sparse_storage:
            concat_chunked_sparse_matrix_to_tensorstore(
                adata2.X, 
                os.path.join(output_path, 'X'),
                chunk_size = chunk_size,
                dtype= DTYPE.uint32 if is_raw_count else None
            )
        else:
            concat_X(
                adata2.X, 
                os.path.join(output_path, 'X'),
                chunk_size = chunk_size,
                dtype= DTYPE.uint32 if is_raw_count else None
            )

    if adata2.layers is not None:
        for k, v in adata2.layers.items():
            if os.path.exists(os.path.join(output_path, 'layers', k)):
                if sparse_storage:
                    concat_chunked_sparse_matrix_to_tensorstore(
                        v, 
                        os.path.join(output_path, f'layers/{k}'),
                        chunk_size = chunk_size
                    )
                else:
                    concat_X(
                        v, 
                        os.path.join(output_path, f'layers/{k}'),
                        chunk_size = chunk_size
                    )

    if os.path.exists(os.path.join(output_path, 'obsm')):
        os.rmdir(os.path.join(output_path, 'obsm'))
    
    if os.path.exists(os.path.join(output_path, 'varm')):
        os.rmdir(os.path.join(output_path, 'varm'))


def save_spatial_image(
    img: Union[xr.core.dataarray.DataArray, np.ndarray],
    output_path: Union[str, os.PathLike]
):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if isinstance(img, np.ndarray):
        img = to_spatial_image(img)
    img.to_zarr(output_path)

def load_spatial_image(
    input_path: Union[str, os.PathLike],
    x_indices: Optional[Union[slice, np.ndarray]] = None,
    y_indices: Optional[Union[slice, np.ndarray]] = None
) -> np.ndarray:
    return xr.open_dataarray(input_path, chunks='auto').isel(x=x_indices, y=y_indices).to_numpy()