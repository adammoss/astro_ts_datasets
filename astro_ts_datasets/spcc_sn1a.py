from .spcc import SPCCDataReader, SPCC


class SPCCSn1aReader(SPCCDataReader):
    class_keys = {
        1: 1,
        2: 0,
        21: 0,
        22: 0,
        23: 0,
        3: 0,
        32: 0,
        33: 0
    }


class SPCCSn1a(SPCC):

    def _generate_examples(self, data_files, metadata_file):
        """Yield examples."""
        reader = SPCCSn1aReader(data_files, metadata_file)
        for instance in reader:
            yield instance
