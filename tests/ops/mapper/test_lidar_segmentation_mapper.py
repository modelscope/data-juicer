import unittest
import os

from data_juicer.core import NestedDataset as Dataset
from data_juicer.ops.mapper.lidar_segmentation_mapper import LiDARSegmentationMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class LiDARSegmentationMapperTest(DataJuicerTestCaseBase):
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    lidar_test1 = os.path.join(data_path, 'lidar_test1.bin')
    lidar_test2 = os.path.join(data_path, 'lidar_test2.bin')
    lidar_test3 = os.path.join(data_path, 'lidar_test3.bin')

    model_cfg_path = "cylinder3d_8xb2-laser-polar-mix-3x_semantickitti.py"
    model_path = "cylinder3d_8xb2-amp-laser-polar-mix-3x_semantickitti_20230425_144950-372cdf69.pth"
 

    def test_cpu(self):
        source = [
            {
                'lidar': self.lidar_test1
            },
            {
                'lidar': self.lidar_test2
            },
            {
                'lidar': self.lidar_test3
            }
        ]
        
        op = LiDARSegmentationMapper(
            model_name="cylinder3d",
            model_cfg_path=self.model_cfg_path,
            model_path=self.model_path,
        )
        
        dataset = Dataset.from_list(source)
        dataset = dataset.map(op.process, batch_size=2, with_rank=False)

        print(dataset)


    def test_cuda(self):
        source = [
            {
                'lidar': self.lidar_test1
            },
            {
                'lidar': self.lidar_test2
            },
            {
                'lidar': self.lidar_test3
            }
        ]
        
        op = LiDARSegmentationMapper(
            model_name="cylinder3d",
            model_cfg_path=self.model_cfg_path,
            model_path=self.model_path,
        )
        
        dataset = Dataset.from_list(source)
        dataset = dataset.map(op.process, batch_size=2, with_rank=True)

        print(dataset)


if __name__ == "__main__":
    unittest.main()