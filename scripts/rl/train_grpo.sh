#!/bin/bash
set -x
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate openrlhf


# Start the Ray head node with the dashboard accessible.
# The --dashboard-host flag ensures the dashboard is available on all interfaces.
ray start --head --port=8000 \
  --dashboard-port=8001 \
  --dashboard-agent-listen-port=8002 \
  --ray-client-server-port=8003

# Wait a few seconds to ensure the head node is fully initialized.
sleep 5

# Define the Ray address.
# The default head node will be available at 127.0.0.1 on port 8266.
RAY_ADDRESS="http://127.0.0.1:8001"
echo "Ray head node is running at $RAY_ADDRESS"

# Now submit your job to the Ray cluster.
ray job submit --address="$RAY_ADDRESS" \
   --runtime-env-json='{"working_dir": "openrlhf"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --colocate_actor_ref \
   --pretrain MODEL_NAME \
   --remote_rm_url REWARD_FUNCTION_PATH \
   --save_path ../saves/rl/MODEL_NAME \
   --ckpt_path ../saves/rl/MODEL_NAME \
   --save_steps 10 \
   --logging_steps 1 \
   --eval_steps 50 \
   --micro_train_batch_size 8 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 16 \
   --rollout_batch_size 512 \
   --num_episodes 20 \
   --max_epochs 1 \
   --prompt_max_len 2048 \
   --generate_max_len 6000 \
   --max_len 8192 \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate 1e-6 \
   --critic_learning_rate 5e-6 \
   --init_kl_coef 0.01 \
   --prompt_data DATA_PATH \
   --input_key question \
   --label_key answer \
   --apply_chat_template \
   --normalize_reward \
   --packing_samples \
   --adam_offload \
   --gradient_checkpointing \
   --flash_attn \
   --load_checkpoint \
   --ring_attn_size 1 \
   --ring_head_stride 1 \
   --n_samples_per_prompt 16 \
   --enable_prefix_caching \
   --advantage_estimator group_norm \
   --freezing_actor_steps -1 \
   --use_kl_loss \
   --kl_estimator k3 \
   --temperature 1.0 \
   --top_p 1.0 \
   --gamma 1.0 \
   --lambd 1.0 \
