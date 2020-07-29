#!/bin/bash
#SBATCH --job-name=bert          # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes. a task is an instance of a running program
#SBATCH --cpus-per-task=2        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=04:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email on job start, end and fail
#SBATCH --mail-user=junmingh@princeton.edu

# usage: sbatch /home/junmingh/virus/code-bert/virus_classifier.sh
# https://researchcomputing.princeton.edu/computational-hardware/tiger/tutorials
# https://hpcc.usc.edu/support/documentation/slurm/
# statistics from previous experiments (State: COMPLETED): total mem 8GB, total CPU 2, total GPU >= 1, total time 1-4 hr
echo "Running on" $HOSTNAME
if [[ $HOSTNAME = tiger* ]]; then
  export BERT_BASE_DIR=/home/junmingh/media/code/uncased_L-12_H-768_A-12
  module load cudnn/cuda-10.0/7.5.0
  cd /home/junmingh/virus/experiment
  python /home/junmingh/virus/code-bert/virus_classifier.py \
    --task_name=tweet-pairwise \
    --do_train=true \
    --do_eval=true \
    --data_dir=/home/junmingh/virus/data-labeled-tweet \
    --vocab_file=${BERT_BASE_DIR}/vocab.txt \
    --bert_config_file=${BERT_BASE_DIR}/bert_config.json \
    --init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
    --max_seq_length=128 \
    --train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=3.0 \
    --output_dir=/home/junmingh/virus/tmp/


#  ~/anaconda3/bin/conda activate tf
#  python3 /home/junmingh/virus/code-bert/virus_classifier.py \
#    --task_name=tweet-pairwise \
#    --do_train=true \
#    --do_eval=true \
#    --data_dir=/home/junmingh/virus/data-labeled-tweet \
#    --vocab_file=${BERT_BASE_DIR}/vocab.txt \
#    --bert_config_file=${BERT_BASE_DIR}/bert_config.json \
#    --init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
#    --max_seq_length=128 \
#    --train_batch_size=32 \
#    --learning_rate=2e-5 \
#    --num_train_epochs=3.0 \
#    --output_dir=/home/junmingh/virus/tmp/
#  ~/anaconda3/bin/conda deactivate

#  python ./classifier.py \
#    --labeled_data_for_train=/home/junmingh/media/data/2019-08-24-cross/topic-7-assignment-annotator-chesley.pickle2 \
#    --labeled_data_for_eval=/home/junmingh/media/data/2019-08-24-cross/topic-7-assignment-annotator-aggregate.pickle2  \
#    --train_set_percentage=0.75 \
#    --eval_set_percentage=0.25 \
#    --eda_whether=True \
#    --num_aug=9 \
#    --alpha=0.1 \
#    --task_name=china_impact \
#    --do_train=true \
#    --do_predict=true \
#    --do_eval=false \
#    --data_dir=/home/junmingh/media/experiment/2019-08-24-cross/topic-7-sentiment-train-chesley-emilyyin-pred-unlabeled-2019-09-18-11-22-08 \
#    --vocab_file=/home/junmingh/media/code/uncased_L-12_H-768_A-12/vocab.txt \
#    --bert_config_file=/home/junmingh/media/code/uncased_L-12_H-768_A-12/bert_config.json \
#    --init_checkpoint=/home/junmingh/media/code/wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt \
#    --max_seq_length=64 \
#    --train_batch_size=16 \
#    --learning_rate=2e-05 \
#    --num_train_epochs=3 \
#    --output_dir=/home/junmingh/media/tmp/fine-tuned/topic-7-sentiment-train-chesley-emilyyin-pred-unlabeled-2019-09-18-11-22-08 \

elif [ "$HOSTNAME" = "ccc-virus-bert" ]; then
  export TPU_NAME=ccc-virus-bert-tpu
  export PYTHONPATH="${PYTHONPATH}:/usr/share/tpu/models"
  export BERT_BASE_DIR=gs://cloud-tpu-checkpoints/bert/uncased_L-12_H-768_A-12
  ctpu up --tpu-only --name=${TPU_NAME} --zone=us-central1-b --tpu-size=v3-8 --tf-version=1.15.3 -noconf #--preemptible
  #export GLUE_DIR=$HOME/glue_data
  #export TASK_NAME=MRPC
  export STORAGE_BUCKET=gs://ccc-virus-bert
  #python3 /home/mail_huangjunming_com/bert/run_classifier.py \
  #	--task_name=${TASK_NAME} \
  #	--do_train=true \
  #	--do_eval=true \
  #	--data_dir=${GLUE_DIR}/${TASK_NAME} \
  #	--vocab_file=${BERT_BASE_DIR}/vocab.txt \
  #	--bert_config_file=${BERT_BASE_DIR}/bert_config.json \
  #	--init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
  #	--max_seq_length=128 \
  #	--train_batch_size=32 \
  #	--learning_rate=2e-5 \
  #	--num_train_epochs=3.0 \
  #	--output_dir=${STORAGE_BUCKET}/${TASK_NAME}-output/ \
  #	--use_tpu=True \
  #	--tpu_name=${TPU_NAME}
  python3 /home/mail_huangjunming_com/virus/code-bert/virus_classifier.py \
    --task_name=tweet-pairwise \
    --do_train=true \
    --do_eval=true \
    --do_predict=true \
    --data_dir=/home/mail_huangjunming_com/virus/data-labeled-tweet \
    --vocab_file=${BERT_BASE_DIR}/vocab.txt \
    --bert_config_file=${BERT_BASE_DIR}/bert_config.json \
    --init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
    --max_seq_length=128 \
    --train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=3.0 \
    --output_dir=${STORAGE_BUCKET}/$(date +'%Y-%m-%d-%H-%M-%S')/ \
    --use_tpu=True \
    --tpu_name=${TPU_NAME}
  ctpu pause --tpu-only --name=${TPU_NAME} --zone=us-central1-b -noconf

  # note: vocab_file/bert_config_file/init_checkpoint/output_dir must be Google Cloud Storage. Apparently a TPU trained SavedModel models can be saved only to the Google Cloud Storage. https://towardsdatascience.com/trials-and-tribulations-using-keras-on-colab-and-tpu-69378762468d

else
    printf "This script must be run on tiger or google cloud.\n"
fi




