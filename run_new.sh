#!/bin/bash
set -x

echo "running run.sh..."
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

MODEL_PATH=./checkpoint/step_102516
#MODEL_PATH=./checkpoint/tbkv_joint_spo_7601

#VOCAB_PATH=ernie/config/base/vocab.txt
#CONFIG_PATH=ernie/config/base/ernie_config.json

#vocab size=5w (chr_word_mix)
VOCAB_PATH=ernie/config/base_5w/vocab.txt
CONFIG_PATH=ernie/config/base_5w/ernie_config.json

TRAIN_PATH=./data/rel_v3.top50.all_process_sort_json
TEST_PATH=./data/relv3_merge_sort_process_sort_v6_sort_1K

TEST_PATH1=./data/relv3_merge_sort_process_sort_v6_sort_1K

###将训练集拆部分出来做验证集
#DEV_PATH=${TASK_DATA_PATH}/train_dev
#all_samples=`wc -l ${TRAIN_PATH} | awk -F " " '{print $1}'`
#dev_samples=$[${all_samples}/10]
#shuf ${TRAIN_PATH} > ${TRAIN_PATH}_shuf 
#head -n ${dev_samples} ${TRAIN_PATH}_shuf > ${DEV_PATH}
#cp ${DEV_PATH} ./table_metric/train_dev.label
#tail -n +$[${dev_samples}+1] ${TRAIN_PATH}_shuf > ${TRAIN_PATH}
#rm -rf ${TRAIN_PATH}_shuf
DEV_PATH=./data/relv3_merge_sort_process_sort_v6_sort_1K

lr=1e-5
batch_size=8
node=1
epoch=2
train_exampls=`wc -l ${TRAIN_PATH} | awk -F " " '{print $1}'`
echo ${train_exampls}
save_steps=$[$train_exampls/$batch_size/$node]
data_size=$[$save_steps*$batch_size*$node]
new_save_steps=$[$save_steps*$epoch/1]
#all_steps=$[$train_samples*$epoch/$batch_size/$node]
new_save_steps=3333
all_steps=20000

#export CPU_NUM=8
export CUDA_VISIBLE_DEVICES=0
#/root/anaconda3/envs/paddle_cpu/bin/python -u ernie/run_classifier.py \
/root/anaconda3/envs/paddle_116/bin/python -u ernie/run_classifier_new.py \
                   --use_cuda true \
                   --verbose true \
                   --do_train true \
                   --do_val false \
                   --do_test false \
                   --use_ema false \
                   --batch_size ${batch_size} \
                   --init_pretraining_params ${MODEL_PATH} \
                   --train_set ${TRAIN_PATH} \
                   --dev_set xxx.test_v8 \
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
                   --skip_steps 1000\
                   --num_iteration_per_drop_scope 1 \
                   --num_labels 1 \
                   --only_pointwise false \
                   --only_pairwise false \
                   --for_cn true \
                   --random_seed 1 \
                   --is_classify true \
                   --is_regression false #\
                   #1>>output/train.log 2>&1
for((step=$new_save_steps;step<20000;step+=$new_save_steps));do
    echo "predicting step "${step} >>output/test.log
    /root/anaconda3/envs/paddle_116/bin/python -u ernie/run_classifier_new.py \
                    --use_cuda true \
                    --verbose true \
                    --do_train false \
                    --do_val false \
                    --do_test true \
                    --use_ema false \
                    --batch_size ${batch_size} \
                    --init_checkpoint output/step_${step} \
                    --checkpoints output/ema_step_${step} \
                    --train_set xxx.train_v8 \
                    --dev_set xxx.test_v8 \
                    --test_set ${TEST_PATH1}\
                    --test_save ${TEST_PATH1}.score.${step} \
                    --max_seq_len 384 \
                    --is_regression true \
                    --is_classify false \
                    --vocab_path ${VOCAB_PATH} \
                    --ernie_config_path ${CONFIG_PATH} \
                    --skip_steps 100 \
                    --num_iteration_per_drop_scope 1 \
                    --num_labels 1 \
                    --for_cn true \
                    --random_seed 1 \
                    #1>>output/test.log 2>&1
done
TASK_NAME=relv3_merge_sort_process_sort_v6_sort_1K
sh table_metric/run_metric.sh ${TASK_NAME} 0.90 > output/res.base_music
sh table_metric/run_metric.sh ${TASK_NAME} 0.95 >> output/res.base_music
sh table_metric/run_metric.sh ${TASK_NAME} 0.96 >> output/res.base_music
