"""Module containing the Supernova Photometric Classification Challenge 2010."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from collections.abc import Sequence
import logging

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

from .util import AstroTsDatasetBuilder, AstroTsDatasetInfo

RESOURCES = os.path.join(
    os.path.dirname(__file__), 'resources', 'spcc')

_CITATION = """
@article{kessler2010results,
  title={Results from the supernova photometric classification challenge},
  author={Kessler, Richard and Bassett, Bruce and Belov, Pavel and Bhatnagar, Vasudha and Campbell, Heather and Conley, Alex and Frieman, Joshua A and Glazov, Alexandre and Gonz{\'a}lez-Gait{\'a}n, Santiago and Hlozek, Ren{\'e}e and others},
  journal={Publications of the Astronomical Society of the Pacific},
  volume={122},
  number={898},
  pages={1415},
  year={2010},
  publisher={IOP Publishing}
}
"""

_DESCRIPTION = """
"""


class SPCCDataReader(Sequence):
    """Reader class for SPCC dataset."""

    static_features = [
        'host_photoz', 'host_photoz_error', 'mwebv'
    ]
    ts_features = [
        'desg_flux', 'desg_flux_error',
        'desr_flux', 'desr_flux_error',
        'desi_flux', 'desi_flux_error',
        'desz_flux', 'desz_flux_error',
    ]

    # Remove instances without any timeseries
    blacklist = [
    ]

    class_keys = {
        1: 0,
        2: 1,
        21: 2,
        22: 3,
        23: 4,
        3: 5,
        32: 6,
        33: 7
    }

    # Time quantisation in days
    time_quantisation = 1.0

    def __init__(self, data_files, metadata_file):
        """Load instances from the SPCC challenge.

        Args:
            data_path: Path containing the records.
            metadata_file: File containing the metadata definitions

        """
        metadata = pd.read_csv(metadata_file, header=0, sep=',')
        self.metadata = metadata[~metadata['object_id'].isin(self.blacklist)]
        self.data = pd.concat([pd.read_csv(data_file, header=0, sep=',') for data_file in data_files])

    def _quantise_time(self, values):
        return self.time_quantisation * np.round(values / self.time_quantisation)

    def __getitem__(self, index):
        """Get instance at position index of metadata file."""
        instance = self.metadata.iloc[index]
        static = instance[self.static_features]
        object_id = instance['object_id']
        # Read data
        timeseries = self._read_timeseries(object_id)
        time = timeseries['time']
        values = timeseries[self.ts_features]

        return object_id, {
            'static': static,
            'time': time,
            'timeseries': values,
            'targets': {
                'class':
                    self.class_keys[instance['class']]
            },
            'metadata': {
                'object_id': object_id
            }
        }

    def _read_timeseries(self, object_id):
        data = self.data[self.data['object_id'] == object_id].copy()
        data['time'] = self._quantise_time(data['time'])
        timeseries = data.pivot_table(index='time', columns='parameter', values='value')
        timeseries = timeseries.reindex(columns=self.ts_features).reset_index()
        return timeseries

    def __len__(self):
        """Return number of instances in the dataset."""
        return len(self.metadata)


class SPCC(AstroTsDatasetBuilder):
    """Dataset of the SPCC."""

    VERSION = tfds.core.Version('1.0.10')
    has_metadata = True
    has_timeseries = True
    default_target = 'class'

    def _info(self):
        return AstroTsDatasetInfo(
            builder=self,
            targets={
                'class':
                    tfds.features.ClassLabel(num_classes=8),
            },
            default_target='class',
            static_names=SPCCDataReader.static_features,
            timeseries_names=SPCCDataReader.ts_features,
            description=_DESCRIPTION,
            homepage='',
            citation=_CITATION
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Return SplitGenerators."""
        paths = dl_manager.download({
            'train_ts': 'https://storage.googleapis.com/spcc/simgen_training_set.csv.gz',  # noqa: E501
            'train_meta': 'https://storage.googleapis.com/spcc/simgen_training_set_metadata.csv.gz',  # noqa: E501
            'test_ts': 'https://storage.googleapis.com/spcc/simgen_test_set.csv.gz',  # noqa: E501
            'test_meta': 'https://storage.googleapis.com/spcc/simgen_test_set_metadata.csv.gz',  # noqa: E501
        })

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    'data_files': [paths['train_ts']],
                    'metadata_file': paths['train_meta']
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    'data_files': [paths['test_ts']],
                    'metadata_file': paths['test_meta']
                }
            )
        ]

    def _generate_examples(self, data_files, metadata_file):
        """Yield examples."""
        reader = SPCCDataReader(data_files, metadata_file)
        for instance in reader:
            yield instance
