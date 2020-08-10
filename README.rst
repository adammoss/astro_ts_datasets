==============================
Astronomy time series datasets
==============================

This module contains the implementation of multiple astronomy time series datasets
following the tensorflow dataset API.

Currently implemented datasets are:

- ``spcc`` (Supernova Photometric Classification Challenge)
- ``plasticc`` (Photometric Lsst Astronomical Time-series Classification Challenge)

This is based on the medical datasets repository https://github.com/ExpectationMax/medical_ts_datasets

Example usage
-------------

In order to get a tensorflow dataset representation of one of the datasets simply
import ``tensorflow_datasets`` and this module.  The datasets can then be accessed
like any other tensorflow dataset.

.. code-block:: python

    import tensorflow_datasets as tfds
    import astro_ts_datasets

    spcc_dataset = tfds.load(name='spcc', split='train')


Instance structure
------------------

Each instance in the dataset is represented as a nested directory of the following
structure:

- ``static``: Static variables such as photometric redshift
- ``time``: Scalar time variable containing the observation time
- ``values``: Observation values of time series, these by default contain `NaN` for
  modalities which were not observed for the given timepoint.
- ``targets``: Directory of potential target values, the available endpoints are
  dataset specific.
- ``metadata``: Directory of metadata on an individual object

Supervised dataset
------------------

If the load method is called with the flag ``as_supervised=True``, it will
return a dataset which can readily be used together with keras. Here each
instance is represented by a (X, y) tuple and the X tuple contains the
following 4 elements: ``time``, ``values``, ``measurements`` (indicators if
a value was measured or not) and ``length``.
