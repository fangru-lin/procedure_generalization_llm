export VLLM_WORKER_MULTIPROC_METHOD=spawn

conda activate procedure

python ./py_scripts/asynchow_eval.py \
    --pretrained PRETRAIN_DIR \
    --dtype auto \
    --temperature 0.0 \
    --n 1 \
    --output_dir $BASE_OUTPUT_DIR \
    --record_likelihoods False \
    --chat_template True \
    --model_name MODEL_NAME \
    --train_set TRAIN_SET \
    --splits test