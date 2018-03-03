 
set -e


# Where the dataset is saved to.
DATASET_DIR=../tmp/flowers

# Download the dataset
python download_and_convert_data.py \
  --dataset_name=flowers \
  --dataset_dir=${DATASET_DIR}
  
  
# Where the checkpoint and logs will be saved to.
TRAIN_DIR=../tmp/densenet-model


# Run training.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=flowers \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=densenet \
  --preprocessing_name=inception \
  --max_number_of_steps=30 \
  --batch_size=128 \
  --save_interval_secs=120 \
  --save_summaries_secs=120 \
  --log_every_n_steps=100 \
  --optimizer=sgd \
  --learning_rate=0.01 \
  --learning_rate_decay_factor=0.1 \
  --num_epochs_per_decay=200 \
  --clone_on_cpu=true\
  --weight_decay=0.004

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=flowers \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=densenet