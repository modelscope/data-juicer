GPUS=$1
python3 -m torch.distributed.launch --nproc_per_node=$GPUS main.py ${@:2}