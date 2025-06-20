import json
import os

import jsonlines as jl
from tqdm import tqdm

from data_juicer.core.sandbox.data_pool_manipulators import (
    BaseDataPoolManipulator,
    check_io_paths,
)
from data_juicer.utils.mm_utils import SpecialTokens


class COCOCaptionToDJConversion(BaseDataPoolManipulator):
    def run(self):
        """
        Convert InternVL COCO Caption datasets to DJ format.

        Input:
            - dataset_path: N InternVL COCO Caption datasets
            - export_path: path to export the result datasets
        Output:
            - N InternVL COCO Caption datasets in DJ format, with two new fields added: "text" for only caption with
                special tokens and "images" with a list of images. They are named following the rule
                "<original_name>_dj_fmt.jsonl"
        """
        # read inputs
        input_dataset_paths = self.data_pool_cfg.get("dataset_path", [])
        export_path = self.data_pool_cfg.get("export_path", None)
        # check I/O paths
        existing_input_paths, export_path = check_io_paths(input_dataset_paths, export_path)

        output_paths = []
        for src_path in tqdm(existing_input_paths, desc="Converting to Data-Juicer format"):
            basename = os.path.splitext(os.path.basename(src_path))[0]
            output_path = os.path.join(export_path, f"{basename}_dj_fmt.jsonl")
            with jl.open(src_path, "r") as reader:
                with jl.open(output_path, "w") as writer:
                    for s in reader:
                        image = s["image"]
                        text = s["conversations"][1]["value"]
                        s["images"] = [image]
                        s["text"] = f"{SpecialTokens.image} {text}"
                        writer.write(s)
            output_paths.append(output_path)

        return output_paths


class COCOCaptionMetaGeneration(BaseDataPoolManipulator):
    def run(self):
        """
        Generate meta file for InternVL COCO Caption datasets.

        Input:
            - dataset_path: N InternVL COCO Caption datasets
            - data_root_path: data root path to store the InternVL COCO Caption datasets, where there should be deeper
                directory starting with "data/coco/xxx"
            - export_path: path to export the result datasets
        Output:
            - N meta file paths. They are named following the rule "<original_name>.json". Refer to:
                https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/data/coco_caption.json
        """
        # read inputs
        input_dataset_paths = self.data_pool_cfg.get("dataset_path", [])
        export_path = self.data_pool_cfg.get("export_path", None)
        data_root_path = self.data_pool_cfg.get("data_root_path", None)
        # check I/O paths
        existing_input_paths, export_path = check_io_paths(input_dataset_paths, export_path)

        default_meta = {
            "root": "data/coco/",
            "annotation": "data/coco/annotations/coco_karpathy_train_567k.jsonl",
            "data_augment": False,
            "repeat_time": 1,
            "length": 566747,
        }

        output_paths = []
        for src_path in tqdm(existing_input_paths, desc="Generating meta files"):
            basename = os.path.basename(src_path)
            meta_key = os.path.splitext(basename)[0]
            ds = []
            default_meta["annotation"] = os.path.relpath(os.path.abspath(src_path), data_root_path)
            with jl.open(src_path, "r") as reader:
                for s in reader:
                    ds.append(s)
            default_meta["length"] = len(ds)
            output_path = os.path.join(export_path, f"{meta_key}_meta.json")
            with open(output_path, "w") as fout:
                json.dump({meta_key: default_meta}, fout)
            output_paths.append(output_path)

        return output_paths
