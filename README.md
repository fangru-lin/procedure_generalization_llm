## procedure_generalization

This repository contains the code for the paper **“Can Large Language Models Generalize Procedures Across Representations?”** ([arxiv:2602.03542](https://arxiv.org/abs/2602.03542)).

### Repository structure

- **`data/`**: AsyncHow (NL, Graph, Code) and other datasets used in the paper.
- **`scripts/sft/`**: Supervised fine-tuning (SFT) and STaR/distillation pipelines.
  - `sft.sh`: vanilla full-parameter SFT on NL / Graph / Code using LLaMA-Factory.
  - `star.sh`: Self-Taught Reasoner (STaR) bootstrapping (10 rounds) on AsyncHow-NL.
  - `asynchow_distillation.sh`: distillation from `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` to Qwen2.5-1.5B.
  - `py_scripts/`: vLLM wrapper, rejection sampler, AsyncHow evaluation utilities.
- **`scripts/rl/`**: GRPO training and reward functions.
  - `train_grpo.sh`: GRPO training with OpenRLHF on AsyncHow (Graph→NL curriculum etc.).
  - `reward_function/`: task-specific verifiable reward functions (AsyncHow, math, physics).

### Environment

Create and activate a Conda environment (names in scripts assume `procedure` for SFT and `openrlhf` for RL):

```bash
conda create -n procedure python=3.10
conda activate procedure
pip install -r requirements.txt  # if provided, otherwise install LLaMA-Factory, vllm, datasets, etc.
```

For RL:

```bash
conda create -n openrlhf python=3.10
conda activate openrlhf
pip install openrlhf ray[tune]  # plus standard HF/torch stack
```

### Running SFT baselines

From the repo root:

```bash
cd scripts/sft
bash sft.sh
```

This script:

- Trains Qwen2.5-{1.5,3,7}B-Instruct on:
  - `asynchow` (NL),
  - `asynchow_graph` (Graph),
  - `asynchow_python` (Code),
- Uses `examples/train_full/general_template.yaml` (2 epochs, full-parameter SFT),
- Evaluates with `py_scripts/asynchow_eval.py` at temperature 0 on all representations.

### Running STaR

```bash
cd scripts/sft
bash star.sh
```

This script:

- Starts from `Qwen/Qwen2.5-1.5B-Instruct` on AsyncHow-NL.
- Uses `vllm_causallms.py` + `rejection_trainer.py` to:
  - Sample k=4 candidate CoTs per prompt (T=1),
  - Filter to correct generations,
  - Build SFT datasets for 10 STaR rounds.
- Each round trains a new checkpoint with LLaMA-Factory using the same `general_template.yaml`.

### Running distillation

```bash
cd scripts/sft
bash asynchow_distillation.sh
```

This script:

- Uses **`deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`** as teacher.
- Calls `vllm_causallms.py` with:
  - `n` starting at 4, `max_iter=5`, `iter_increase_n_samples=2` (effective n schedule ≈ {4,8,16,32,64}).
- Uses `rejection_trainer.py` to construct a distilled SFT dataset.
- Trains `Qwen/Qwen2.5-1.5B-Instruct` on that dataset with LLaMA-Factory.
- Evaluates the distilled student with `asynchow_eval.py`.

### Running GRP

```bash
cd scripts/rl
bash train_grpo.sh
```

This script:

- Starts a local Ray cluster and launches OpenRLHF `train_ppo_ray` with:
  - `num_episodes=20`, `max_epochs=1` (single epoch, 20 episodes),
  - `train_batch_size=128`, `rollout_batch_size=512`,
  - `prompt_max_len=2048`, `generate_max_len=6000`,
  - `init_kl_coef=0.01`, `n_samples_per_prompt=16` (k=16).
- Uses `reward_function/reward_func_asynchow.py` (and others) implementing 0/1 verifiable rewards as in the paper.

### Datasets and representations

- AsyncHow-NL, AsyncHow-Graph, AsyncHow-Code, and NL-AAVE variants are stored under `data/asynchow/`.
- Math and physics dataset live under `data/math` and `data/physics` (see paper for exact splits).
