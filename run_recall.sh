#!/bin/bash
#set -x

echo "running run.sh..."
#export FLAGS_enable_parallel_graph=1
#export FLAGS_eager_delete_tensor_gb=0
#export FLAGS_sync_nccl_allreduce=1
#export FLAGS_fraction_of_gpu_memory_to_use=0.95
#export PATH="$PWD_DIR/$PYDIR/bin/:$PATH"
#export PYTHONPATH="$PWD_DIR/$PYDIR/lib/python2.7/site-packages/:$PYTHONPATH"
#export LD_LIBRARY_PATH="/home/work/cudnn/cudnn_v7/cuda/lib64:$LD_LIBRARY_PATH"
#export LD_LIBRARY_PATH="/home/work/cuda-9.0/lib64/:$LD_LIBRARY_PATH"
#export LD_LIBRARY_PATH="$PWD_DIR/nccl_2.3.7-1+cuda9.0_x86_64/lib:$LD_LIBRARY_PATH"
#export GLOG_v=1
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH

TEST_PATH=table_bkv_data_round1
MODEL_PATH=./checkpoint/baike_twin_round1_7384
CHECKPOINT_PATH=output

lr=1e-5
batch_size=8
node=1
epoch=2
train_exampls=10000
train_exampls=162186
save_steps=$[$train_exampls/$batch_size/$node]
data_size=$[$save_steps*$batch_size*$node]
new_save_steps=$[$save_steps*$epoch/4]
new_save_steps=2000

echo "trian begin1...."
#export CPU_NUM=8
export CUDA_VISIBLE_DEVICES=0
#/root/anaconda3/envs/paddle_cpu/bin/python ./ernie/run_classifier_768.py \
/root/anaconda3/envs/paddle_116/bin/python ./ernie/run_classifier_recall.py \
                   --is_distributed false \
                   --use_recompute true \
                   --use_mix_precision true \
                   --use_cross_batch false \
                   --use_cuda true \
                   --verbose true \
                   --do_train true \
                   --do_val false \
                   --do_test true \
                   --batch_size ${batch_size} \
                   --train_data_size ${data_size} \
                   --train_set ./data/recall_train_tbkv_twin_round1.joint_2pos_4neg_tag \
                   --dev_set ${TEST_PATH}/top1_twin_test.top50 \
                   --test_set ${TEST_PATH}/tbkv_twin_test.top50 \
                   --test_save ${CHECKPOINT_PATH}/tbkv_score \
                   --use_fast_executor true \
                   --checkpoints ${CHECKPOINT_PATH} \
                   --save_steps ${new_save_steps} \
                   --validation_steps ${new_save_steps} \
                   --learning_rate ${lr} \
                   --epoch ${epoch} \
                   --q_max_seq_len 32 \
                   --p_max_seq_len 384 \
                   --init_pretraining_params ${MODEL_PATH} \
                   --vocab_path ernie/config/base/vocab.txt \
                   --ernie_config_path ernie/config/base/ernie_config.json \
                   --weight_decay  0.0 \
                   --warmup_proportion 0.1 \
                   --skip_steps 10 \
                   --num_iteration_per_drop_scope 1 \
                   --for_cn true \
                   --random_seed 1
