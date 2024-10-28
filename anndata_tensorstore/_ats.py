import anndata
from fastparquet import ParquetFile
import os
import multiscale_spatial_image

from ._ext import save_anndata_to_tensorstore, load_anndata_from_tensorstore, ATS_FILE_NAME


class AnndataTensorStore:
    def __init__(self, anndata: anndata.AnnData):
        self.anndata = anndata

    def save(self, path):
        save_anndata_to_tensorstore(self.anndata, path)

    @staticmethod
    def load(path):
        return load_anndata_from_tensorstore(path)
    
    @staticmethod
    def view(path):
        obs_info = ParquetFile(os.path.join(path, ATS_FILE_NAME.obs)).info
        var_info = ParquetFile(os.path.join(path, ATS_FILE_NAME.var)).info
        print("AnnDataTensorStore with n_obs x n_vars = {} x {}\n".format(obs_info['rows'], var_info['rows']) + \
                "    obs: {}\n".format(', '.join(obs_info['columns'][:-1])) + \
                "    var: {}\n".format(', '.join(var_info['columns'][:-1]))
        )
