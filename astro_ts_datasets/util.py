"""Utility functions and classes used by medical ts datasets."""
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core.features import FeaturesDict, Tensor


class AstroTsDatasetInfo(tfds.core.DatasetInfo):
    """DatasetINfo for astro time series datasets."""

    time_dtype = tf.float32
    static_dtype = tf.float32
    timeseries_dtype = tf.float32
    object_id_dtype = tf.uint32

    @tfds.core.api_utils.disallow_positional_args
    def __init__(self, builder, targets, default_target,
                 static_names=None, timeseries_names=None,
                 description=None, homepage=None, citation=None):
        """Dataset info for medical time series datasets.

        Ensures all datasets follow a similar structure and can be used
        (almost) interchangably.

        Args:
            builder: Builder class associated with this dataset info.
            targets: Dictionary of endpoints.
            static_names: Names of the static measurements.
            timeseries_names: Names of the time series measurements.
            description: Dataset description.
            homepage: Homepage of dataset.
            citation: Citation of dataset.

        """
        self.has_static = static_names is not None
        self.has_timeseries = timeseries_names is not None
        self.default_target = default_target

        metadata = tfds.core.MetadataDict()
        features_dict = {
            'time': Tensor(shape=(None,), dtype=self.time_dtype)
        }
        static_is_categorical = []
        timeseries_is_categorical = []
        if self.has_static:
            metadata['static_names'] = static_names
            static_is_categorical.extend(
                ['=' in static_name for static_name in static_names])
            features_dict['static'] = Tensor(
                shape=(len(static_names),),
                dtype=self.static_dtype)
        if self.has_timeseries:
            metadata['timeseries_names'] = timeseries_names
            timeseries_is_categorical.extend(
                ['=' in name for name in timeseries_names])
            features_dict['timeseries'] = Tensor(
                shape=(None, len(timeseries_names),),
                dtype=self.timeseries_dtype)

        metadata['static_categorical_indicator'] = static_is_categorical
        metadata['timeseries_categorical_indicator'] = timeseries_is_categorical

        features_dict['targets'] = targets
        features_dict['metadata'] = {'object_id': self.object_id_dtype}
        features_dict = FeaturesDict(features_dict)
        # TODO: If we are supposed to return raw values, we cannot make
        # a supervised dataset
        if builder.output_raw:
            supervised_keys = None
        else:
            supervised_keys = ("combined", "target")

        super().__init__(
            builder=builder,
            description=description, homepage=homepage, citation=citation,
            features=features_dict,
            supervised_keys=supervised_keys,
            metadata=metadata
        )


class AstroTsDatasetBuilder(tfds.core.GeneratorBasedBuilder):
    """Builder class for astro time series datasets."""

    def __init__(self, output_raw=False, add_measurements_and_lengths=True,
                 **kwargs):
        self.output_raw = output_raw
        self.add_measurements_and_lengths = add_measurements_and_lengths
        super().__init__(**kwargs)

    def _as_dataset(self, **kwargs):
        """Evtl. transform categorical covariates into one-hot encoding."""
        dataset = super()._as_dataset(**kwargs)
        if self.output_raw:
            return dataset

        has_static = self.info.has_static
        collect_ts = []
        if self.has_timeseries:
            collect_ts.append('timeseries')

        def preprocess_output(instance):
            if has_static:
                static = instance['static']
            else:
                static = None

            time = instance['time']
            time_series = tf.concat(
                [instance[mod_type] for mod_type in collect_ts], axis=-1)

            if self.add_measurements_and_lengths:
                measurements = tf.math.is_finite(time_series)
                length = tf.shape(time)[0]
                return {
                    'combined': (
                        static,
                        time,
                        time_series,
                        measurements,
                        length
                    ),
                    'target': instance['targets'][self.default_target]
                }
            else:
                return {
                    'combined': (static, time, time_series),
                    'target': instance['targets'][self.default_target]
                }

        return dataset.map(
            preprocess_output,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

