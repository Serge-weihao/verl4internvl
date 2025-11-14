set -x
#mpirun -np 4 -map-by node 
export WANDB_API_KEY="xxxx"
source ./miniconda3/bin/activate verl_qwentest1
cd ./verl_internvl_bszsp
#hmod -R u+w /lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas
#export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
SHARED_PATH="./exp_tmp/$AIP_RUN_ID"
ADDRESS_FILE="$SHARED_PATH/master_addr.txt"
# pip install vllm/dist/vllm-xxx-cp310-cp310-linux_x86_64.whl
# python3 -m pip install --upgrade optree==0.13.0 
# python -m pip uninstall flash-attn -y
# MAX_JOBS=64 python -m pip install flash-attn==2.7.4.post1 --no-build-isolation 
#sleep 200s
use_dynamic_bsz=True
#clip_ratio_low=0.2
#clip_ratio_high=0.28

clip_ratio_low=0.0003 # as recommended by the paper, see Sec. 5.1
clip_ratio_high=0.0004 # as recommended by the paper, see Sec. 5.1

max_prompt_length=$((1024 * 4))
max_response_length=$((1024 * 3))
enable_overlong_buffer=False
overlong_buffer_len=0
overlong_penalty_factor=0

loss_mode="gspo"
#loss_agg_mode="token-mean"
loss_agg_mode="seq-mean-token-mean"

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=12
train_prompt_bsz=128
gen_prompt_bsz=$((train_prompt_bsz * 3))
if [ "$RANK" -eq 0 ]; then
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
    ray start --head --port=8679 --node-ip-address=$podName.job-$SERVICE_NAME --num-gpus=8 --num-cpus=20 --object-store-memory=799511627776
    sleep 60s
else
    sleep 30s
    ray start --address=$FIRST_MASTER_ADDR:8679 --node-ip-address=$podName.job-$SERVICE_NAME --num-gpus=8 --num-cpus=20 --object-store-memory=799511627776
    ray status
fi
sleep 40
which python
#export PYTHONPATH=xxx/vllm:$PYTHONPATH

ray status
if [ "$RANK" -eq 0 ]; then
    python  -m recipe.dapo.main_dapo  \
        actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
        algorithm.adv_estimator=grpo  \
        ++actor_rollout_ref.actor.loss_mode=${loss_mode}     \
        trainer.default_local_dir=./ckpt/interngspo \
        +model.trust_remote_code=True \
        data.train_files="['./rldata/train_mmk12_cast/train.parquet','./rldata/gsm8kim/train.parquet']"  \
        data.val_files="[./rldata/geo3k/test.parquet]"  \
        data.train_batch_size=${train_prompt_bsz}  \
        data.max_prompt_length=${max_prompt_length}   \
        data.max_response_length=${max_response_length}  \
        data.gen_batch_size=${gen_prompt_bsz} \
        actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
        actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
        actor_rollout_ref.actor.clip_ratio_c=10.0 \
        algorithm.filter_groups.enable=${enable_filter_groups} \
        algorithm.filter_groups.metric=${filter_groups_metric} \
        algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
        reward_model.reward_manager=dapo \
        reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
        reward_model.overlong_buffer.len=${overlong_buffer_len} \
        reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
        actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
        data.filter_overlong_prompts=True  \
        data.truncation='error'  \
        data.image_key=images \
        actor_rollout_ref.model.path=OpenGVLab/InternVL2_5-4B-MPO \
        actor_rollout_ref.ref.strategy=fsdp \
        actor_rollout_ref.actor.strategy=fsdp \
        actor_rollout_ref.actor.optim.lr=1e-6  \
        actor_rollout_ref.model.use_remove_padding=True   \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
        actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_bsz}     \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2    \
        actor_rollout_ref.actor.use_kl_loss=True     \
        actor_rollout_ref.actor.kl_loss_coef=0.000     \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl     \
        actor_rollout_ref.actor.entropy_coeff=0 \
        ++actor_rollout_ref.model.trust_remote_code=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True     \
        actor_rollout_ref.actor.fsdp_config.param_offload=True     \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True     \
        ++actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2     \
        ++actor_rollout_ref.rollout.tensor_model_parallel_size=2     \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.rollout.name=vllm     \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6     \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.rollout.n=48     \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=2 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8     \
        actor_rollout_ref.ref.fsdp_config.param_offload=True     \
        algorithm.kl_ctrl.kl_coef=0.000    \
        trainer.critic_warmup=0     \
        trainer.logger=['console','wandb']     \
        ++trainer.val_before_train=False     \
        trainer.project_name='interngspo'     \
        trainer.experiment_name='interngspo'     \
        trainer.n_gpus_per_node=8     \
        trainer.nnodes=2      \
        trainer.max_actor_ckpt_to_keep=1 \
        trainer.save_freq=10    \
        trainer.test_freq=-1     \
        trainer.rollout_data_dir="./interngspo"\
        trainer.total_epochs=4
fi
