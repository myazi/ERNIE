#!/bin/bash
set -x

# echo "running run.sh..."
#export FLAGS_enable_parallel_graph=1
#export FLAGS_eager_delete_tensor_gb=0
#export FLAGS_sync_nccl_allreduce=1
#export FLAGS_fraction_of_gpu_memory_to_use=0.95
#export GLOG_v=1
#export LD_LIBRARY_PATH=/usr/local/cuda:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/root/anaconda3/envs/pytorch/lib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/usr/local/cuda-10.2/:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH

CHECKPOINT_PATH=output

#MODEL_PATH=./checkpoint/step_102516
MODEL_PATH=./checkpoint/tbkv_joint_spo_7601

VOCAB_PATH=ernie/config/base/vocab.txt
CONFIG_PATH=ernie/config/base/ernie_config.json

#vocab size=5w (chr_word_mix)
#VOCAB_PATH=ernie/config/base_5w/vocab.txt
#CONFIG_PATH=ernie/config/base_5w/ernie_config.json

TRAIN_PATH=./data/v5_baike_kv_qtp_V2_5_1_add_sample_process_test5_model_2_3_diff_all_sort_rule_sample_sort
TEST_PATH=./data/v5_baike_kv_qtp_V2_5_1_add_sample_process_test5_model_2_3_diff_all_sort_rule_sample_sort_1K

TEST_PATH1=./data/test_v6_kv.qtp

###将训练集拆部分出来做验证集
#DEV_PATH=${TASK_DATA_PATH}/train_dev
#all_samples=`wc -l ${TRAIN_PATH} | awk -F " " '{print $1}'`
#dev_samples=$[${all_samples}/10]
#shuf ${TRAIN_PATH} > ${TRAIN_PATH}_shuf 
#head -n ${dev_samples} ${TRAIN_PATH}_shuf > ${DEV_PATH}
#cp ${DEV_PATH} ./table_metric/train_dev.label
#tail -n +$[${dev_samples}+1] ${TRAIN_PATH}_shuf > ${TRAIN_PATH}
#rm -rf ${TRAIN_PATH}_shuf
DEV_PATH=./data/v5_baike_kv_qtp_V2_5_1_add_sample_process_test5_model_2_3_diff_all_sort_rule_sample_sort_1K

lr=1e-5
batch_size=8
node=500
epoch=0
train_exampls=`wc -l ${TRAIN_PATH} | awk -F " " '{print $1}'`
echo ${train_exampls}
#save_steps=$[$train_exampls/$batch_size/$node]
#data_size=$[$save_steps*$batch_size*$node]
#new_save_steps=$[$save_steps*$epoch/1]
#all_steps=$[$train_samples*$epoch/$batch_size/$node]
new_save_steps=1000
all_steps=10000

#export CPU_NUM=8
export CUDA_VISIBLE_DEVICES=0
#/root/anaconda3/envs/paddle_cpu/bin/python -u ernie/run_classifier.py \
/root/anaconda3/envs/paddle_116/bin/python -u ernie/run_classifier_orgin.py \
                   --use_cuda true \
                   --verbose true \
                   --do_train true \
                   --do_val false \
                   --do_test true \
                   --use_ema false \
                   --batch_size ${batch_size} \
                   --init_pretraining_params ${MODEL_PATH} \
                   --train_set ${TRAIN_PATH} \
                   --dev_set 28420.test_v8 \
                   --test_set $TEST_PATH \
                   --test_save ${TEST_PATH}_score \
                   --checkpoints ${CHECKPOINT_PATH} \
                   --save_steps ${new_save_steps} \
                   --validation_steps ${new_save_steps} \
                   --weight_decay 0.01 \
                   --warmup_proportion 0.0 \
                   --epoch $epoch \
                   --max_seq_len 384 \
                   --margin 1.0 \
                   --vocab_path ${VOCAB_PATH} \
                   --ernie_config_path ${CONFIG_PATH} \
                   --learning_rate ${lr} \
                   --end_learning_rate 0 \
                   --skip_steps 100\
                   --num_iteration_per_drop_scope 1 \
                   --num_labels 2 \
                   --only_pointwise false \
                   --only_pairwise false \
                   --for_cn true \
                   --random_seed 1 \
                   --is_classify true \
                   --is_regression false #\
                   #1>>output/train.log 2>&1

for((step=$new_save_steps;step<${all_steps};step+=$new_save_steps));do
    echo "predicting step "${step}
    /root/anaconda3/envs/paddle_116/bin/python -u ernie/run_classifier_orgin.py >> output/test.log \
                   --use_cuda true \
                   --verbose true \
                   --do_train false \
                   --do_val false \
                   --do_test true \
                   --use_ema false \
                   --metric acc_and_f1 \
                   --batch_size 8 \
                   --train_set ${TRAIN_PATH} \
                   --dev_set ${TEST_PATH} \
                   --test_set ${TEST_PATH1}\
                   --test_save ${TEST_PATH1}.score.${step} \
                   --vocab_path ${VOCAB_PATH} \
                   --checkpoints output/step_${step} \
                   --init_checkpoint output/step_${step} \
                   --max_seq_len 384 \
                   --ernie_config_path ${CONFIG_PATH} \
                   --num_iteration_per_drop_scope 1 \
                   --num_labels 2 \
                   --random_seed 1
done
TASK_NAME=test_v6_kv.qtp
sh table_metric/run_metric.sh ${TASK_NAME} 0.90 > output/res.base
sh table_metric/run_metric.sh ${TASK_NAME} 0.95 >> output/res.base
sh table_metric/run_metric.sh ${TASK_NAME} 0.96 >> output/res.base
