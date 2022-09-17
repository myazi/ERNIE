#export FLAGS_eager_delete_tensor_gb=0
#export FLAGS_sync_nccl_allreduce=1
#export CUDA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH

# MODEL_PATH=vip_triple_cos_title005_7075
CHECKPOINT_PATH=./data/tbkv_twin_round1_7636
MODEL_PATH=./ernie/config/base

/root/anaconda3/envs/paddle_116/bin/python -u ernie/paddle_service_recall.py \
       --use_cuda true                                                                  \
       --use_fp16 ${USE_FP16:-"false"}                                                  \
       --batch_size 32                                                         \
       --init_checkpoint ${CHECKPOINT_PATH} \
       --test_save test_out.tsv \
       --output_item 1 \
       --output_file_name query_emb \
       --read_id 1 \
       --vocab_path ${MODEL_PATH}/vocab.txt \
       --q_max_seq_len 32                                                               \
       --p_max_seq_len 384                                                              \
       --ernie_config_path ${MODEL_PATH}/ernie_config.json \
       --num_labels 2                                                                   \
       --for_cn true                                                                   \
       --save_part all \
       --save_inference_model_path ${CHECKPOINT_PATH}_infer_all \
       --server_port 8053 >server_log 2>server_log & 

/root/anaconda3/envs/paddle_116/bin/python -u ernie/paddle_service_recall.py \
       --use_cuda true                                                                  \
       --use_fp16 ${USE_FP16:-"false"}                                                  \
       --batch_size 32                                                         \
       --init_checkpoint ${CHECKPOINT_PATH} \
       --test_save test_out.tsv \
       --output_item 1 \
       --output_file_name query_emb \
       --read_id 1 \
       --vocab_path ${MODEL_PATH}/vocab.txt \
       --q_max_seq_len 32                                                               \
       --p_max_seq_len 384                                                              \
       --ernie_config_path ${MODEL_PATH}/ernie_config.json \
       --num_labels 2                                                                   \
       --for_cn true                                                                   \
       --save_part query \
       --save_inference_model_path ${CHECKPOINT_PATH}_infer_q \
       --server_port 8054 >server_log 2>server_log &

/root/anaconda3/envs/paddle_116/bin/python -u ernie/paddle_service_recall.py \
       --use_cuda true                                                                  \
       --use_fp16 ${USE_FP16:-"false"}                                                  \
       --batch_size 32                                                         \
       --init_checkpoint ${CHECKPOINT_PATH} \
       --test_save test_out.tsv \
       --output_item 1 \
       --output_file_name query_emb \
       --read_id 1 \
       --vocab_path ${MODEL_PATH}/vocab.txt \
       --q_max_seq_len 32                                                               \
       --p_max_seq_len 384                                                              \
       --ernie_config_path ${MODEL_PATH}/ernie_config.json \
       --num_labels 2                                                                   \
       --for_cn true                                                                   \
       --save_part para \
       --save_inference_model_path ${CHECKPOINT_PATH}_infer_p \
       --server_port 8055 >>server_log 2>>server_log &

LOG_PATH="./log"
if [ ! -z "$1" ]; then
    port=$1
else
    port=8086
fi

port=8086
#export FLAGS_eager_delete_tensor_gb=0
#export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH

##注意query和para模型，需要改_p、_q的地方
##改服务端口

#export CPU_NUM=8
#/root/anaconda3/envs/paddle_cpu/bin/python -u ernie_server_infer_q/paddle_service.py \
/root/anaconda3/envs/paddle_116/bin/python -u ernie/paddle_service_recall.py \
       --use_cuda true                                                                  \
       --use_fp16 ${USE_FP16:-"false"}                                                  \
       --batch_size 16                                                         \
       --init_checkpoint ${CHECKPOINT_PATH}_infer_p \
       --test_save test_out.tsv \
       --output_item 1 \
       --output_file_name query_emb \
       --read_id false \
       --vocab_path ${MODEL_PATH}/vocab.txt \
       --q_max_seq_len 32                                                               \
       --p_max_seq_len 384                                                              \
       --ernie_config_path ${MODEL_PATH}/ernie_config.json \
       --num_labels 2                                                                   \
       --for_cn true                                                                   \
       --server_port ${port} > ${LOG_PATH}/server_log_p 2>>${LOG_PATH}/server_log_p &


LOG_PATH="./log"
if [ ! -z "$1" ]; then
    port=$1
else
    port=8087
fi

port=8087
#export FLAGS_eager_delete_tensor_gb=0
#export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH

##注意query和para模型，需要改_p、_q的地方
##改服务端口

#export CPU_NUM=8
#/root/anaconda3/envs/paddle_cpu/bin/python -u ernie_server_infer_q/paddle_service.py \
/root/anaconda3/envs/paddle_116/bin/python -u ernie/paddle_service_recall.py \
       --use_cuda true                                                                  \
       --use_fp16 ${USE_FP16:-"false"}                                                  \
       --batch_size 16                                                         \
       --init_checkpoint ${CHECKPOINT_PATH}_infer_q \
       --test_save test_out.tsv \
       --output_item 1 \
       --output_file_name query_emb \
       --read_id false \
       --vocab_path ${MODEL_PATH}/vocab.txt \
       --q_max_seq_len 32                                                               \
       --p_max_seq_len 384                                                              \
       --ernie_config_path ${MODEL_PATH}/ernie_config.json \
       --num_labels 2                                                                   \
       --for_cn true                                                                   \
       --server_port ${port} > ${LOG_PATH}/server_log_q 2>>${LOG_PATH}/server_log_q &
