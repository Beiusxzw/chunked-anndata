import setuptools 
from src._version import version

version = version
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="chunked-anndata",
    version=version,
    url="https://github.com/xueziwei/chunked-anndata",
    author="Ziwei Xue",
    author_email="xueziweisz@gmail.com",
    description="Save and Load anndata in chunked format for random access",
    long_description=long_description,
    long_description_content_type='text/plain',
    packages=setuptools.find_packages(exclude=[
        "*docs*",
    ]),
    install_requires=[
        'anndata',
        'tensorstore',
        'pandas',
        'numpy',
        'scipy',
        'anndata',
        'geopandas',
        'fastparquet',
        'xarray',
        'zarr',
        'dask',
        'spatial_image',
    ],
    include_package_data=False,
)
