import sys
from typing import Optional, Tuple, Union

from pydantic import PositiveFloat, PositiveInt

from data_juicer import cuda_device_count
from data_juicer.ops.filter.video_motion_score_filter import \
    VideoMotionScoreFilter
from data_juicer.utils.lazy_loader import LazyLoader

from ..base_op import OPERATORS, UNFORKABLE

torch = LazyLoader('torch', 'torch')
tvm = LazyLoader('tvm', 'torchvision.models')
tvt = LazyLoader('tvt', 'torchvision.transforms')

OP_NAME = 'video_motion_score_raft_filter'


@UNFORKABLE.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class VideoMotionScoreRaftFilter(VideoMotionScoreFilter):
    """Filter to keep samples with video motion scores within a specified range.
    This operator utilizes the RAFT (Recurrent All-Pairs Field Transforms)
    model from torchvision to predict optical flow between video frames.

    For further details, refer to the official torchvision documentation:
    https://pytorch.org/vision/main/models/raft.html

    The original paper on RAFT is available here:
    https://arxiv.org/abs/2003.12039
    """

    _accelerator = 'cuda'
    _default_kwargs = {}

    def __init__(self,
                 min_score: float = 1.0,
                 max_score: float = sys.float_info.max,
                 sampling_fps: PositiveFloat = 2,
                 size: Union[PositiveInt, Tuple[PositiveInt],
                             Tuple[PositiveInt, PositiveInt], None] = None,
                 max_size: Optional[PositiveInt] = None,
                 divisible: PositiveInt = 8,
                 relative: bool = False,
                 any_or_all: str = 'any',
                 *args,
                 **kwargs):
        super().__init__(min_score, max_score, sampling_fps, size, max_size,
                         divisible, relative, any_or_all, *args, **kwargs)

    def setup_model(self, rank=None):
        self.model = tvm.optical_flow.raft_large(
            weights=tvm.optical_flow.Raft_Large_Weights.DEFAULT,
            progress=False)
        if self.use_cuda():
            rank = rank if rank is not None else 0
            rank = rank % cuda_device_count()
            self.device = f'cuda:{rank}'
        else:
            self.device = 'cpu'
        self.model.to(self.device)
        self.model.eval()

        self.transforms = tvt.Compose([
            tvt.ToTensor(),
            tvt.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
            tvt.Lambda(lambda img: img.flip(-3).unsqueeze(0)),  # BGR to RGB
        ])

    def compute_flow(self, prev_frame, curr_frame):
        curr_frame = self.transforms(curr_frame).to(self.device)
        if prev_frame is None:
            flow = None
        else:
            with torch.inference_mode():
                flows = self.model(prev_frame, curr_frame)
            flow = flows[-1][0].cpu().numpy().transpose(
                (1, 2, 0))  # 2, H, W -> H, W, 2
        return flow, curr_frame
