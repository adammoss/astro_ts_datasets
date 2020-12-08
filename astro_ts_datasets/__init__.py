"""Tensorflow datasets of astronomical time series."""
import astro_ts_datasets.checksums
import astro_ts_datasets.spcc
import astro_ts_datasets.plasticc

builders = [
    'spcc',
    'plasticc'
]

__version__ = '0.1.0'
