Install [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), name the environment as 'openrlhf', then ground this folder in the root directory for relevant scripts.

## Reward Function

The reward function can be found in `reward_function`. Use `reward_func_asynchow.py` for asynchow data, and `reward_func_math.py` and `reward_func_physics.py` for math and physics data respectively.

## Training

Modify the `MODEL_NAME`, `REWARD_FUNCTION_PATH`, `DATA_PATH` in the script `train_grpo.sh`, then run the script.