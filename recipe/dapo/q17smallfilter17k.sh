#!/usr/bin/env bash
set -x

export NCCL_BLOCKING_WAIT=1  # 强制同步等待，避免异步超时误判
export NCCL_TIMEOUT=3600000  # 超时时间设为60分钟（单位：ms）
# 启动训练时设置，生成NCCL通信日志
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576  # 1MB缓存，足够记录单步操作
export NCCL_DEBUG=INFO  # 打印NCCL调试日志


SHARED_PATH="./exp_tmp/$AIP_RUN_ID"
ADDRESS_FILE="$SHARED_PATH/master_addr.txt"


project_name='DAPOnormalq3_1_7_1e-6filter17kopdis'
exp_name='dscalernormal1e-6filter17kopdis'

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 20))
enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

train_prompt_bsz=256
n_resp_per_prompt=16
train_prompt_mini_bsz=32

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10

gen_prompt_bsz=$((train_prompt_bsz * 3))
# Ray
# RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
# WORKING_DIR=${WORKING_DIR:-"${PWD}"}
# RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
# Paths
#RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=./Qwen3-1.7
DIST_MODEL_PATH=Qwen/Qwen3-8B
#./DeepSeek_R1_Distill_Qwen_1_5B
#./Qwen3-1.7
#${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3-30B-A3B-Base"}
CKPTS_DIR=./ckpts/${project_name}/${exp_name}
#${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=./still30k/train.parquet
#${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
TEST_FILE=./dapo17k/train.parquet 
#${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7
export WANDB_API_KEY="b9cb4c5167aae55b9cdee22c4627d684ca2256fa"
# Performance Related Parameter
sp_size=2
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 4))
offload=True
gen_tp=1
fsdp_size=-1
cd ./onpolicy/verl


export HYDRA_FULL_ERROR=1


if [ "$RANK" -eq 0 ]; then
    # 写操作
    # 确保共享路径存在
    mkdir -p "$SHARED_PATH"
    echo "MASTER_ADDR: $MASTER_ADDR"
    FIRST_MASTER_ADDR=$(hostname -i)
    echo "$FIRST_MASTER_ADDR" > "$ADDRESS_FILE"
    echo "Master address written by RANK $RANK"
else
    # 读操作
    echo "RANK $RANK waiting for master address..."
    ls $ADDRESS_FILE
    echo $ADDRESS_FILE
    while [ ! -f "$ADDRESS_FILE" ]; do
        sleep 1  # 等待写操作完成
    done
    FIRST_MASTER_ADDR=$(cat "$ADDRESS_FILE")
    echo "RANK $RANK received master address: $FIRST_MASTER_ADDR"
fi
if [ "$RANK" -eq 0 ]; then
    ray start --head --port=8679 --node-ip-address=$podName.job-$SERVICE_NAME  --num-gpus=8 --object-store-memory=799511627776 #--num-cpus=20
    sleep 60s
else
    sleep 30s
    ray start --address=$FIRST_MASTER_ADDR:8679 --node-ip-address=$podName.job-$SERVICE_NAME  --num-gpus=8 --object-store-memory=799511627776  #--num-cpus=20
    ray status
fi
sleep 40
which python

ray status
if [ "$RANK" -eq 0 ]; then
    python  -m recipe.dapo.main_dapo  \
        data.train_files="${TRAIN_FILE}" \
        data.val_files="${TEST_FILE}" \
        data.prompt_key=prompt \
        data.truncation='left' \
        data.max_prompt_length=${max_prompt_length} \
        data.max_response_length=${max_response_length} \
        data.train_batch_size=${train_prompt_bsz} \
        actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
        algorithm.adv_estimator=${adv_estimator} \
        algorithm.use_kl_in_reward=${use_kl_in_reward} \
        algorithm.kl_ctrl.kl_coef=${kl_coef} \
        actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
        actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
        actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
        actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
        actor_rollout_ref.actor.clip_ratio_c=10.0 \
        data.gen_batch_size=${gen_prompt_bsz} \
        algorithm.filter_groups.enable=${enable_filter_groups} \
        algorithm.filter_groups.metric=${filter_groups_metric} \
        algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        +actor_rollout_ref.ref.model.path="${DIST_MODEL_PATH}" \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
        actor_rollout_ref.actor.optim.weight_decay=0.1 \
        actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
        actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.grad_clip=1.0 \
        actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.rollout.max_num_batched_tokens=$(((max_prompt_length + max_response_length) * 2)) \
        actor_rollout_ref.rollout.temperature=${temperature} \
        actor_rollout_ref.rollout.top_p=${top_p} \
        actor_rollout_ref.rollout.top_k=${top_k} \
        actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
        actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
        actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.n=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
        actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
        reward_model.reward_manager=dapo \
        +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
        +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
        +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
        +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
        +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
        trainer.logger='["console","wandb"]' \
        trainer.project_name="${project_name}" \
        trainer.experiment_name="${exp_name}" \
        trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
        trainer.nnodes="${NNODES}" \
        trainer.val_before_train=False \
        trainer.test_freq=-1 \
        trainer.max_actor_ckpt_to_keep=1 \
        trainer.save_freq=20 \
        trainer.total_epochs=2 \
        trainer.default_local_dir="${CKPTS_DIR}" \
        trainer.resume_mode=auto \
        trainer.log_val_generations=10
fi

