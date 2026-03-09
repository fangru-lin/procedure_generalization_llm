export VLLM_WORKER_MULTIPROC_METHOD=spawn
export FORCE_TORCHRUN=1

conda activate procedure

BASE_DIR="../saves"
TASK="asynchow_natural"
SIZE="1.5"
# Round 0 produces `star_0` from the base model.
# Then we run NUM_ROUNDS additional STaR rounds to produce `star_1`..`star_{NUM_ROUNDS}`.
NUM_ROUNDS=10

scripts_dir="./py_scripts"
cli_dir="./examples/train_full"

# start with base model
echo "Starting generating training data round 0..."
# skip sampling if the output dir is not empty
if [ -d "$BASE_DIR/$TASK/qwen${SIZE}b_star_0" ]; then
    echo "Output directory is not empty, skipping sampling"
else
    echo "Start sampling data for qwen2.5_${SIZE}b"
    python ./py_scripts/vllm_causallms.py \
    --pretrained Qwen/Qwen2.5-${SIZE}B-Instruct \
    --dtype auto \
    --temperature 1 \
    --input_path ../data/asynchow/${TASK}_train.jsonl \
    --save_path $BASE_DIR/$TASK/qwen${SIZE}b_star_0 \
    --batch_size 32 \
    --num_samples 4
fi

# get training data for round 0 (from base model generations)
python ./py_scripts/rejection_trainer.py \
--selected_answer_path $BASE_DIR/$TASK/qwen${SIZE}b_star_0/Qwen2.5-${SIZE}B-Instruct/rejection_sampling.json \
--input_path ../data/asynchow/${TASK}_train.jsonl \
--model_name qwen${SIZE}b_star_0 \
--task $TASK

# skip if the output dir is not empty
if [ -d "$BASE_DIR/$TASK/qwen${SIZE}b_star_0/checkpoint" ]; then
    echo "Output directory is not empty, skipping training"
else
    echo "Start training qwen2.5_${SIZE}b"
    # get new training script
    cp $cli_dir/general_template.yaml $cli_dir/qwen2.5_${SIZE}b_star_0_${TASK}.yaml
    sed -i '' "s/PRETRAINED_MODEL/Qwen\/Qwen2.5-${SIZE}B-Instruct/g" $cli_dir/qwen2.5_${SIZE}b_star_0_${TASK}.yaml
    sed -i '' "s/DATASET_PATH/..\/data\/asynchow\/${TASK}_train.jsonl/g" $cli_dir/qwen2.5_${SIZE}b_star_0_${TASK}.yaml
    sed -i '' "s/OUTPUT_DIR/..\/saves\/${TASK}\/qwen${SIZE}b_star_0/g" $cli_dir/qwen2.5_${SIZE}b_star_0_${TASK}.yaml

    echo "Starting training baseline..."
    # Run the training command
    llamafactory-cli train $cli_dir/qwen2.5_${SIZE}b_star_0_${TASK}.yaml
    echo "Training baseline completed."
fi

# start next rounds (10 STaR iterations: 1..NUM_ROUNDS)
for i in $(seq 1 $NUM_ROUNDS); do
    prev=$((i-1))   # previous checkpoint index (0..9)
    curr=$i         # current checkpoint index   (1..10)

    # skip sampling if the output dir is not empty
    if [ -d "$BASE_DIR/$TASK/qwen${SIZE}b_star_${curr}" ]; then
        echo "Output directory is not empty, skipping sampling for round ${curr}"
    else
        echo "Starting generating training data round ${curr}..."
        python ./py_scripts/vllm_causallms.py \
        --pretrained $BASE_DIR/$TASK/qwen${SIZE}b_star_${prev}/checkpoint \
        --dtype auto \
        --temperature 1 \
        --input_path ../data/asynchow/${TASK}_train.jsonl \
        --save_path $BASE_DIR/$TASK/qwen${SIZE}b_star_${curr} \
        --batch_size 32 \
        --num_samples 4
        echo "Generating training data round ${curr} completed."
    fi

    # get training data for current round
    python ./py_scripts/rejection_trainer.py \
    --selected_answer_path $BASE_DIR/$TASK/qwen${SIZE}b_star_${curr}/checkpoint/rejection_sampling.json \
    --input_path ../data/asynchow/${TASK}_train.jsonl \
    --model_name qwen${SIZE}b_star_${curr} \
    --task $TASK

    # skip if the output dir is not empty
    if [ -d "$BASE_DIR/$TASK/qwen${SIZE}b_star_${curr}/checkpoint" ]; then
        echo "Output directory is not empty, skipping training for round ${curr}"
    else
        echo "Starting training round ${curr}..."
        # get new training script
        cp $cli_dir/general_template.yaml $cli_dir/qwen2.5_${SIZE}b_star_${curr}_${TASK}.yaml
        sed -i '' "s|PRETRAINED_MODEL|$BASE_DIR/$TASK/qwen${SIZE}b_star_${prev}/checkpoint|g" $cli_dir/qwen2.5_${SIZE}b_star_${curr}_${TASK}.yaml
        sed -i '' "s|DATASET_PATH|../data/asynchow/${TASK}_train.jsonl|g" $cli_dir/qwen2.5_${SIZE}b_star_${curr}_${TASK}.yaml
        sed -i '' "s|OUTPUT_DIR|../saves/${TASK}/qwen${SIZE}b_star_${curr}|g" $cli_dir/qwen2.5_${SIZE}b_star_${curr}_${TASK}.yaml

        # Run the training command
        llamafactory-cli train $cli_dir/qwen2.5_${SIZE}b_star_${curr}_${TASK}.yaml
        echo "Training round ${curr} completed."
    fi
done