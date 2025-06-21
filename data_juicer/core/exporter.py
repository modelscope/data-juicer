import os
from multiprocessing import Pool

from loguru import logger

from data_juicer.utils.constant import Fields, HashKeys


class Exporter:
    """The Exporter class is used to export a dataset to files of specific
    format."""

    KiB = 2**10  # 1024
    MiB = 2**20  # 1024*1024
    GiB = 2**30  # 1024*1024*1024
    TiB = 2**40  # 1024*1024*1024*1024

    def __init__(self,
                 export_path,
                 export_shard_size=0,
                 export_in_parallel=True,
                 num_proc=1,
                 export_ds=True,
                 keep_stats_in_res_ds=False,
                 keep_hashes_in_res_ds=False,
                 export_stats=True):
        """
        Initialization method.

        :param export_path: the path to export datasets.
        :param export_shard_size: the size of each shard of exported
            dataset. In default, it's 0, which means export the dataset
            to a single file.
        :param num_proc: number of process to export the dataset.
        :param export_ds: whether to export the dataset contents.
        :param keep_stats_in_res_ds: whether to keep stats in the result
            dataset.
        :param keep_hashes_in_res_ds: whether to keep hashes in the result
            dataset.
        :param export_stats: whether to export the stats of dataset.
        """
        self.export_path = export_path
        self.export_shard_size = export_shard_size
        self.export_in_parallel = export_in_parallel
        self.export_ds = export_ds
        self.keep_stats_in_res_ds = keep_stats_in_res_ds
        self.keep_hashes_in_res_ds = keep_hashes_in_res_ds
        self.export_stats = export_stats
        self.suffix = self._get_suffix(export_path)
        self.num_proc = num_proc
        self.max_shard_size_str = ''

        # get the string format of shard size
        if self.export_shard_size // Exporter.TiB:
            self.max_shard_size_str = '%.2f TiB' % (self.export_shard_size /
                                                    Exporter.TiB)
        elif self.export_shard_size // Exporter.GiB:
            self.max_shard_size_str = '%.2f GiB' % (self.export_shard_size /
                                                    Exporter.GiB)
        elif self.export_shard_size // Exporter.MiB:
            self.max_shard_size_str = '%.2f MiB' % (self.export_shard_size /
                                                    Exporter.MiB)
        elif self.export_shard_size // Exporter.KiB:
            self.max_shard_size_str = '%.2f KiB' % (self.export_shard_size /
                                                    Exporter.KiB)
        else:
            self.max_shard_size_str = '%.2f Bytes' % (self.export_shard_size)

        # we recommend users to set a shard size between MiB and TiB.
        if 0 < self.export_shard_size < Exporter.MiB:
            logger.warning(f'The export_shard_size [{self.max_shard_size_str}]'
                           f' is less than 1MiB. If the result dataset is too '
                           f'large, there might be too many shard files to '
                           f'generate.')
        if self.export_shard_size >= Exporter.TiB:
            logger.warning(f'The export_shard_size [{self.max_shard_size_str}]'
                           f' is larger than 1TiB. It might generate large '
                           f'single shard file and make loading and exporting '
                           f'slower.')

    def _get_suffix(self, export_path):
        """
        Get the suffix of export path and check if it's supported.

        We only support ["jsonl", "json", "parquet"] for now.

        :param export_path: the path to export datasets.
        :return: the suffix of export_path.
        """
        suffix = export_path.split('.')[-1].lower()
        support_dict = self._router()
        if suffix not in support_dict:
            raise NotImplementedError(f'Suffix of export path ['
                                      f'{export_path}] is not supported '
                                      f'for now. Only support '
                                      f'{list(support_dict.keys())}.')
        return suffix

    def _export_impl(self, dataset, export_path, suffix, export_stats=True):
        """
        Export a dataset to specific path.

        :param dataset: the dataset to export.
        :param export_path: the path to export the dataset.
        :param suffix: suffix of export path.
        :param export_stats: whether to export stats of dataset.
        :return:
        """
        if export_stats:
            # export stats of datasets into a single file.
            logger.info('Exporting computed stats into a single file...')
            export_columns = []
            if Fields.stats in dataset.features:
                export_columns.append(Fields.stats)
            if Fields.meta in dataset.features:
                export_columns.append(Fields.meta)
            if len(export_columns):
                ds_stats = dataset.select_columns(export_columns)
                stats_file = export_path.replace('.' + suffix, '_stats.jsonl')
                Exporter.to_jsonl(
                    ds_stats,
                    stats_file,
                    num_proc=self.num_proc if self.export_in_parallel else 1)

        if self.export_ds:
            # fetch the corresponding export method according to the suffix
            if not self.keep_stats_in_res_ds:
                extra_fields = {Fields.stats, Fields.meta}
                feature_fields = set(dataset.features.keys())
                removed_fields = extra_fields.intersection(feature_fields)
                dataset = dataset.remove_columns(removed_fields)
            if not self.keep_hashes_in_res_ds:
                extra_fields = {
                    HashKeys.hash,
                    HashKeys.minhash,
                    HashKeys.simhash,
                    HashKeys.imagehash,
                    HashKeys.videohash,
                }
                feature_fields = set(dataset.features.keys())
                removed_fields = extra_fields.intersection(feature_fields)
                dataset = dataset.remove_columns(removed_fields)
            export_method = Exporter._router()[suffix]
            if self.export_shard_size <= 0:
                # export the whole dataset into one single file.
                logger.info('Export dataset into a single file...')
                export_method(
                    dataset,
                    export_path,
                    num_proc=self.num_proc if self.export_in_parallel else 1)
            else:
                # compute the dataset size and number of shards to split
                if dataset._indices is not None:
                    dataset_nbytes = dataset.data.nbytes * len(
                        dataset._indices) / len(dataset.data)
                else:
                    dataset_nbytes = dataset.data.nbytes
                num_shards = int(dataset_nbytes / self.export_shard_size) + 1
                num_shards = min(num_shards, len(dataset))

                # split the dataset into multiple shards
                logger.info(f'Split the dataset to export into {num_shards} '
                            f'shards. Size of each shard <= '
                            f'{self.max_shard_size_str}')
                shards = [
                    dataset.shard(num_shards=num_shards,
                                  index=i,
                                  contiguous=True) for i in range(num_shards)
                ]
                len_num = len(str(num_shards)) + 1
                num_fmt = f'%0{len_num}d'

                # regard the export path as a directory and set file names for
                # each shard
                dirname = os.path.dirname(os.path.abspath(self.export_path))
                basename = os.path.basename(self.export_path).split('.')[0]
                os.makedirs(dirname, exist_ok=True)
                filenames = [
                    os.path.join(
                        dirname, f'{basename}-{num_fmt % index}-of-'
                        f'{num_fmt % num_shards}'
                        f'.{self.suffix}') for index in range(num_shards)
                ]

                # export dataset into multiple shards using multiprocessing
                logger.info(f'Start to exporting to {num_shards} shards.')
                pool = Pool(self.num_proc)
                for i in range(num_shards):
                    pool.apply_async(export_method,
                                     args=(
                                         shards[i],
                                         filenames[i],
                                     ))
                pool.close()
                pool.join()

    def export(self, dataset):
        """
        Export method for a dataset.

        :param dataset: the dataset to export.
        :return:
        """
        self._export_impl(dataset, self.export_path, self.suffix,
                          self.export_stats)

    def export_compute_stats(self, dataset, export_path):
        """
        Export method for saving compute status in filters
        """
        keep_stats_in_res_ds = self.keep_stats_in_res_ds
        self.keep_stats_in_res_ds = True
        self._export_impl(dataset,
                          export_path,
                          self.suffix,
                          export_stats=False)
        self.keep_stats_in_res_ds = keep_stats_in_res_ds

    @staticmethod
    def to_jsonl(dataset, export_path, num_proc=1, **kwargs):
        """
        Export method for jsonl target files.

        :param dataset: the dataset to export.
        :param export_path: the path to store the exported dataset.
        :param num_proc: the number of processes used to export the dataset.
        :param kwargs: extra arguments.
        :return:
        """
        dataset.to_json(export_path, force_ascii=False, num_proc=num_proc)

    @staticmethod
    def to_json(dataset, export_path, num_proc=1, **kwargs):
        """
        Export method for json target files.

        :param dataset: the dataset to export.
        :param export_path: the path to store the exported dataset.
        :param num_proc: the number of processes used to export the dataset.
        :param kwargs: extra arguments.
        :return:
        """
        dataset.to_json(export_path,
                        force_ascii=False,
                        num_proc=num_proc,
                        lines=False)

    @staticmethod
    def to_parquet(dataset, export_path, **kwargs):
        """
        Export method for parquet target files.

        :param dataset: the dataset to export.
        :param export_path: the path to store the exported dataset.
        :param kwargs: extra arguments.
        :return:
        """
        dataset.to_parquet(export_path)

    # suffix to export method
    @staticmethod
    def _router():
        """
        A router from different suffixes to corresponding export methods.

        :return: A dict router.
        """
        return {
            'jsonl': Exporter.to_jsonl,
            'json': Exporter.to_json,
            'parquet': Exporter.to_parquet,
        }
