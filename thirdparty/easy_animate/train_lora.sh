##########################################################
# pretrained model path
PRETRAINED_MODEL_NAME_OR_PATH=$1
TRANSFORMER_PATH=$2

# dataset path
DATASET_NAME=$3
DATASET_META_NAME=$4

# training config
SAMPLE_SIZE=$5
MIXED_PRECISION=$6
BATCH_SIZE_PER_GPU=$7
GRADIENT_ACCUMULATION_STEPS=$8
NUM_TRAIN_EPOCHS=$9
DATALOADER_NUM_WORKERS=${10}

# saving config
OUTPUT_DIR=${11}
CHECKPOINTING_STEPS=5000
VALIDATION_STEPS=500
VALIDATION_PROMPTS="A soaring drone footage captures the majestic beauty of a coastal cliff, its red and yellow stratified rock faces rich in color and against the vibrant turquoise of the sea. Seabirds can be seen taking flight around the cliff\'s precipices. As the drone slowly moves from different angles, the changing sunlight casts shifting shadows that highlight the rugged textures of the cliff and the surrounding calm sea. The water gently laps at the rock base and the greenery that clings to the top of the cliff, and the scene gives a sense of peaceful isolation at the fringes of the ocean. The video captures the essence of pristine natural beauty untouched by human structures."

# tracer config
PROJECT_NAME=${12}
EXPERIMENT_NAME=${13}
##########################################################


accelerate launch --mixed_precision=$MIXED_PRECISION train_lora.py \
  --config_path "config/easyanimate_video_motion_module_v1.yaml" \
  --pretrained_model_name_or_path=$PRETRAINED_MODEL_NAME_OR_PATH \
  --transformer_path=$TRANSFORMER_PATH \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --sample_size=$SAMPLE_SIZE \
  --sample_n_frames=16 \
  --sample_stride=2 \
  --train_batch_size=$BATCH_SIZE_PER_GPU \
  --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
  --num_train_epochs=$NUM_TRAIN_EPOCHS \
  --dataloader_num_workers=$DATALOADER_NUM_WORKERS \
  --checkpointing_steps=$CHECKPOINTING_STEPS \
  --validation_prompts="$VALIDATION_PROMPTS" \
  --output_dir=$OUTPUT_DIR \
  --validation_steps=$VALIDATION_STEPS \
  --learning_rate=2e-05 \
  --seed=42 \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --report_to="wandb" \
  --tracker_project_name=$PROJECT_NAME \
  --tracker_experiment_name=$EXPERIMENT_NAME
