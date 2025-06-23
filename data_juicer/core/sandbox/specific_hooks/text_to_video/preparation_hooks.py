import json
import os

from tqdm import tqdm

from data_juicer.core.sandbox.data_pool_manipulators import (
    BaseDataPoolManipulator,
    check_io_paths,
)
from data_juicer.utils.mm_utils import SpecialTokens


class DJToEasyAnimateVideoConversion(BaseDataPoolManipulator):

    def run(self):
        """
        Convert Video datasets in DJ format to the target format for Video Dataset in EasyAnimate.
        DJ format:
        {
            "videos": ["/path/to/video.mp4"],
            "text": "<__dj__video> the caption of this video"
        }
        -->
        Video Dataset:
        {
            "file_path": "/path/to/video.mp4",
            "text": "the caption of this video"
        }

        Input:
            - dataset_path: N Video datasets in DJ format
            - export_path: path to export the result datasets
        Output:
            - N Video datasets for EasyAnimate, with one new field added: "file_path" for the video path, and one field
                modified: remove special tokens in "text". They are named following the rule
                "<original_name>_ea_fmt.json"
        """
        # read inputs
        input_dataset_paths = self.data_pool_cfg.get("dataset_path", [])
        export_path = self.data_pool_cfg.get("export_path", None)
        # check I/O paths
        existing_input_paths, export_path = check_io_paths(input_dataset_paths, export_path)

        output_paths = []
        for src_path in tqdm(existing_input_paths, desc="Converting to EasyAnimate video dataset format"):
            basename = os.path.splitext(os.path.basename(src_path))[0]
            output_path = os.path.join(export_path, f"{basename}_ea_fmt.json")
            with open(src_path, "r") as fin:
                ori_fmt = json.load(fin)
                with open(output_path, "w") as fout:
                    res_fmt = []
                    for s in ori_fmt:
                        s["text"] = s["text"].replace(SpecialTokens.video, "").strip()
                        if len(s["videos"]) == 0:
                            continue
                        s["file_path"] = s["videos"][0]
                        res_fmt.append(s)
                    json.dump(res_fmt, fout)
            output_paths.append(output_path)

        return output_paths
