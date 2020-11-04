echo "bash 2020-11-03-unittest.sh"
if [ "$HOSTNAME" = "ccc-virus-deeplearning" ] || [ "$HOSTNAME" = "ccc-virus-bert" ]; then
  if [ "$HOSTNAME" = "ccc-virus-deeplearning" ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate py37tf115
  fi
  python3 -V
  python3 -c 'import tensorflow; print("tensorflow", tensorflow.__version__)'

  export TPU_NAME=ccc-virus-bert-tpu
  export PYTHONPATH="${PYTHONPATH}:/usr/share/tpu/models"
  ctpu up --tpu-only --name=ccc-virus-bert-tpu --zone=us-central1-b --tpu-size=v3-8 --tf-version=1.15.3 -noconf  --require-permissions #--preemptible
  python3 /home/junmingh/virus/code-bert/tweet_classifier.py \
    --task_name=tweet-pointwise \
    --do_train=false \
    --do_eval=false \
    --do_predict=true \
    --test_file=/home/junmingh/virus/test_en0.tsv \
    --output_file=/home/junmingh/virus/test_en0_scores.tsv \
    --data_dir=/home/junmingh/virus/ \
    --vocab_file=gs://cloud-tpu-checkpoints/bert/uncased_L-12_H-768_A-12/vocab.txt \
    --bert_config_file=gs://cloud-tpu-checkpoints/bert/uncased_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=gs://ccc-virus-bert/2020-10-24-bert/topic-f-train-karla-eval-phillip-lang-en-pointwise-0-5436-1/ \
    --max_seq_length=128 \
    --train_batch_size=32 \
    --learning_rate=2e-05 \
    --num_train_epochs=4.0 \
    --output_dir=gs://ccc-virus-bert/2020-11-03-bert/test_en0/ \
    --use_tpu=True \
    --tpu_name=ccc-virus-bert-tpu \
    --identifier=/home/junmingh/virus/test_en0
  python3 /home/junmingh/virus/code-bert/tweet_classifier.py \
    --task_name=tweet-pointwise \
    --do_train=false \
    --do_eval=false \
    --do_predict=true \
    --test_files='/home/junmingh/virus/test_en1.tsv|/home/junmingh/virus/test_en2.tsv' \
    --output_files='/home/junmingh/virus/test_en1_scores.tsv|/home/junmingh/virus/test_en2_scores.tsv' \
    --data_dir=/home/junmingh/virus/ \
    --vocab_file=gs://cloud-tpu-checkpoints/bert/uncased_L-12_H-768_A-12/vocab.txt \
    --bert_config_file=gs://cloud-tpu-checkpoints/bert/uncased_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=gs://ccc-virus-bert/2020-10-24-bert/topic-f-train-karla-eval-phillip-lang-en-pointwise-0-5436-1/ \
    --max_seq_length=128 \
    --train_batch_size=32 \
    --learning_rate=2e-05 \
    --num_train_epochs=4.0 \
    --output_dir=gs://ccc-virus-bert/2020-11-03-bert/test_en1/ \
    --use_tpu=True \
    --tpu_name=ccc-virus-bert-tpu \
    --identifier=/home/junmingh/virus/test_en1
  ctpu pause --tpu-only --name=ccc-virus-bert-tpu --zone=us-central1-b -noconf
fi

