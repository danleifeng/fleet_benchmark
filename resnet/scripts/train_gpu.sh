#!/bin/bash
#sleep 10m
set -x
export FLAGS_sync_nccl_allreduce=1
export FLAGS_cudnn_exhaustive_search=1
export FLAGS_conv_workspace_size_limit=7000 #MB

#export GLOG_v=1
#export GLOG_vmodule=conv_cudnn_op=10,conv_cudnn_helper=10
export GLOG_logtostderr=1
export FLAGS_eager_delete_tensor_gb=0
export NCCL_DEBUG=INFO

ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.7 /usr/lib/x86_64-linux-gnu/libcudnn.so
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64/:/usr/lib/x86_64-linux-gnu/:/usr/lib64/:/usr/local/lib/:$LD_LIBRARY_PATH
export PADDLE_PSERVER_PORT=9184

MODEL=ResNet50 #VGG16
MODEL_SAVE_PATH="output/"

# training params
NUM_EPOCHS=2
BATCH_SIZE=32
LR=0.1
LR_STRATEGY=piecewise_decay

# data params
DATA_PATH="/ImageNet"
TOTAL_IMAGES=1281167
CLASS_DIM=1000
IMAGE_SHAPE=3,224,224


#gpu params
FUSE=True
NCCL_COMM_NUM=1
NUM_THREADS=2
USE_HIERARCHICAL_ALLREDUCE=False
NUM_CARDS=1
FP16=False #whether to use float16 

if [[ ${FUSE} == "True" ]]; then
    export FLAGS_fuse_parameter_memory_size=16 #MB
    export FLAGS_fuse_parameter_groups_size=50
fi

pip install kubernetes
apt-get install -f -y libglib2.0-0
apt-get install -f -y libsm6
apt-get install -f -y libxrender1
apt-get install -f -y libxext-dev


current_ip=`hostname -i`
ips = "`python utils/k8s_tools.py fetch_ips mpi_role_type=worker`"
echo "ips: ${ips}"
distributed_args=""
if [[ ${ips} != "" ]]; then
    distributed_args="--cluster_node_ips=${ips} --node_ip=${current_ip}"
    #distributed_args="--cluster_node_ips=${ips}"
fi

if [[ ${NUM_CARDS} == "1" ]]; then
    distributed_args="${distributed_args} --selected_gpus 0"
fi

#sleep 1h
python -m paddle.distributed.launch ${distributed_args} --log_dir log \
       ./train_with_fleet.py \
       --model=${MODEL} \
       --batch_size=${BATCH_SIZE} \
       --total_images=${TOTAL_IMAGES} \
       --data_dir=${DATA_PATH} \
       --class_dim=${CLASS_DIM} \
       --image_shape=${IMAGE_SHAPE} \
       --model_save_dir=${MODEL_SAVE_PATH} \
       --with_mem_opt=False \
       --lr_strategy=${LR_STRATEGY} \
       --lr=${LR} \
       --num_epochs=${NUM_EPOCHS} \
       --l2_decay=1e-4 \
       --scale_loss=1.0 \
       --fuse=${FUSE} \
       --num_threads=${NUM_THREADS} \
       --nccl_comm_num=${NCCL_COMM_NUM} \
       --use_hierarchical_allreduce=${USE_HIERARCHICAL_ALLREDUCE} \
       --fp16=${FP16}

cat log/workerlog.0
