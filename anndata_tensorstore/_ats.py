import anndata
from fastparquet import ParquetFile
import os

from ._ext import save_anndata_to_tensorstore, load_anndata_from_tensorstore, ATS_FILE_NAME


class AnndataTensorStore:
    
    @staticmethod
    def view(path: str):
        obs_info = ParquetFile(os.path.join(path, ATS_FILE_NAME.obs)).info
        var_info = ParquetFile(os.path.join(path, ATS_FILE_NAME.var)).info
        print("AnnDataTensorStore with n_obs x n_vars = {} x {}\n".format(obs_info['rows'], var_info['rows']) + \
                "    obs: {}\n".format(', '.join(obs_info['columns'][:-1])) + \
                "    var: {}\n".format(', '.join(var_info['columns'][:-1]))
        )
