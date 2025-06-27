import os

from loguru import logger

from data_juicer.utils.constant import Fields, HashKeys


class RayExporter:
    """The Exporter class is used to export a ray dataset to files of specific
    format."""

    # TODO: support config for export, some export methods require additional args
    _SUPPORTED_FORMATS = {
        "json",
        "jsonl",
        "parquet",
        "csv",
        "tfrecords",
        "webdataset",
        "lance",
        # 'images',
        # 'numpy',
    }

    def __init__(self, export_path, keep_stats_in_res_ds=True, keep_hashes_in_res_ds=False):
        """
        Initialization method.

        :param export_path: the path to export datasets.
        :param keep_stats_in_res_ds: whether to keep stats in the result
            dataset.
        :param keep_hashes_in_res_ds: whether to keep hashes in the result
            dataset.
        """
        self.export_path = export_path
        self.keep_stats_in_res_ds = keep_stats_in_res_ds
        self.keep_hashes_in_res_ds = keep_hashes_in_res_ds
        self.export_format = self._get_export_format(export_path)

    def _get_export_format(self, export_path):
        """
        Get the suffix of export path and check if it's supported.
        We only support ["jsonl", "json", "parquet"] for now.

        :param export_path: the path to export datasets.
        :return: the export data format.
        """
        suffix = os.path.splitext(export_path)[-1].strip(".")
        if not suffix:
            logger.warning(
                f'export_path "{export_path}" does not have a suffix. '
                f'We will use "jsonl" as the default export type.'
            )
            suffix = "jsonl"

        export_format = suffix
        if export_format not in self._SUPPORTED_FORMATS:
            raise NotImplementedError(
                f'export data format "{export_format}" is not supported '
                f"for now. Only support {self._SUPPORTED_FORMATS}."
            )
        return export_format

    def _export_impl(self, dataset, export_path, columns=None):
        """
        Export a dataset to specific path.

        :param dataset: the dataset to export.
        :param export_path: the path to export the dataset.
        :param columns: the columns to export.
        :return:
        """
        feature_fields = dataset.columns() if not columns else columns
        removed_fields = []
        if not self.keep_stats_in_res_ds:
            extra_fields = {Fields.stats, Fields.meta}
            removed_fields.extend(list(extra_fields.intersection(feature_fields)))
        if not self.keep_hashes_in_res_ds:
            extra_fields = {
                HashKeys.hash,
                HashKeys.minhash,
                HashKeys.simhash,
                HashKeys.imagehash,
                HashKeys.videohash,
            }
            removed_fields.extend(list(extra_fields.intersection(feature_fields)))

        if len(removed_fields):
            dataset = dataset.drop_columns(removed_fields)

        if self.export_format in {"json", "jsonl"}:
            return dataset.write_json(export_path, force_ascii=False)
        else:
            return getattr(dataset, f"write_{self.export_format}")(export_path)

    def export(self, dataset, columns=None):
        """
        Export method for a dataset.

        :param dataset: the dataset to export.
        :param columns: the columns to export.
        :return:
        """
        self._export_impl(dataset, self.export_path, columns)
