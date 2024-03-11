#!/bin/bash
#SBATCH --job-name=parallel_data_processing
#SBATCH --ntasks=
#SBATCH --nodes=
#SBATCH --time=
#SBATCH --partition=your_partition_name
#SBATCH --output=parallel_data_processing_%j.out
#SBATCH --error=parallel_data_processing_%j.err
#SBATCH --exclusive

# set data-juicer and config file path
datajuicer_path=  # 替换为 data-juicer 的实际路径
config_path=  # 替换为实际的配置文件路径


cd $datajuicer_path

readarray -t nodes <<< "$(sinfo --noheader --states=idle,mixed --format=%n)"

PARTITION_SCRIPT=./scripts/dlc/partition_data_dlc.py

# set dataset path
JSON_FILE_PATH= #替换为实际的数据集路径

# split_dataset
python $PARTITION_SCRIPT $JSON_FILE_PATH "${nodes[@]}"

# run on nodes

for node in "${nodes[@]}"; do
    echo $node
    nohup srun --nodes=1 --ntasks=1 -w $node scripts/dlc/run_on_dlc.sh > output_$node.log 2>&1 &
done
