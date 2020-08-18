"""Tensorflow datasets of astronomical time series."""
import astro_ts_datasets.checksums
import astro_ts_datasets.spcc
import astro_ts_datasets.spcc_sn1a
import astro_ts_datasets.plasticc

builders = [
    'spcc',
    'spcc_sn1a',
    'plasticc'
]

__version__ = '0.1.0'
