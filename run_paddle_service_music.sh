#export FLAGS_eager_delete_tensor_gb=0
#export FLAGS_sync_nccl_allreduce=1
#export CUDA_VISIBLE_DEVICES=6
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH

CHECKPOINT_PATH=V2_5_1_test6_rule_sample_sort_ema_infer
#CHECKPOINT_PATH=ema_step_1461
CHECKPOINT_PATH=./data/ema_step_1461_infer ## 1) 进行infer时，模型不带infer后缀，2) 使用带后缀的infer模型进行服务部署
DATA_PATH=./data/
MODEL_PATH=./ernie/config/base_5w

cat run_paddle_service_music.sh >${CHECKPOINT_PATH}.log

#export CPU_NUM=8
#/root/anaconda3/envs/paddle_cpu/bin/python -u ernie_server/paddle_service.py \
/root/anaconda3/envs/paddle_116/bin/python -u ernie/paddle_service_music.py \
                   --ernie_version 1.0 \
                   --use_cuda true \
                   --batch_size 16 \
                   --ernie_config_path ${MODEL_PATH}/ernie_config.json \
                   --num_labels 1 \
                   --data_dir ${DATA_PATH} \
                   --predict_set ${DATA_PATH}/top1test.format \
                   --vocab_path ${MODEL_PATH}/vocab.txt \
                   --init_checkpoint ${CHECKPOINT_PATH} \
                   --max_seq_len 384 \
                   --use_ema false \
                   --do_prediction False \
                   --save_inference_model_path ${CHECKPOINT_PATH}_infer \
                   --server_port 8082 > ${CHECKPOINT_PATH}.log 2>&1 &
