export VLLM_WORKER_MULTIPROC_METHOD=spawn
export FORCE_TORCHRUN=1
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate procedure

res_dir="../saves/asynchow_res"
ck_dir="../saves/sft_baseline"
cli_dir="./examples/train_full"
scripts_dir="./py_scripts"
BASE_OUTPUT_DIR="../saves/asynchow_res"

for dataset in asynchow asynchow_graph asynchow_python; do
    if [ "$dataset" = "asynchow" ]; then
        train_set="sft_natural"
        data_file="asynchow_natural_train.jsonl"
    elif [ "$dataset" = "asynchow_python" ]; then
        train_set="sft_python"
        data_file="asynchow_python_train.jsonl"
    else
        train_set="sft_graph"
        data_file="asynchow_graph_train.jsonl"
    fi
    for size in 1.5 3 7; do
        # skip training if the output dir is not empty
        if [ -d "$ck_dir/$train_set/qwen${size}b" ] && [ -n "$(ls -A $ck_dir/$train_set/qwen${size}b)" ]; then
            echo "Output directory is not empty, skipping training"
        else
            echo "Start training qwen2.5_${size}b"
            sed -e "s|MODEL_NAME|Qwen/Qwen2.5-${size}B-Instruct|" \
                -e "s|OUTPUT_DIR|$ck_dir/$train_set/qwen${size}b|" \
                -e "s|RUN_NAME|qwen${size}b_${train_set}|" \
                -e "s|DATASET|$dataset|" \
                -e "s|DATA_FILE|$data_file|" \
                $cli_dir/general_template.yaml \
                > $cli_dir/qwen2.5_${size}b_${train_set}.yaml
            # Run the training command
            llamafactory-cli train $cli_dir/qwen2.5_${size}b_${train_set}.yaml
        fi
        echo "Training qwen2.5_${size}b completed"

        python $scripts_dir/asynchow_eval.py \
            --pretrained $ck_dir/$train_set/qwen${size}b \
            --dtype auto \
            --temperature 0.0 \
            --n 1 \
            --output_dir $BASE_OUTPUT_DIR \
            --record_likelihoods False \
            --chat_template True \
            --model_name "Qwen2.5-${size}B-Instruct" \
            --train_set $train_set
    done
done

# start with base model
echo "Starting generating training data round 0..."
# skip sampling if the output dir is not empty
if [ -d "$res_dir/qwen2.5-1.5b-instruct" ]; then
    echo "Output directory is not empty, skipping sampling"
else
    echo "Start sampling data for qwen2.5_1.5b"
    python ./py_scripts/vllm_causallms.py \
    --pretrained Qwen/Qwen2.5-1.5B-Instruct \
    --dtype auto \
    --max_tokens 2048 \
    --temperature 0.7 \
    --top_p 0.9 \
    --input_path ../data/asynchow/asynchow_natural_train.jsonl \
    --save_path $res_dir/qwen2.5-1.5b-instruct \
    --batch_size 32 \
    --num_samples 32
fi

# get training data
python ./py_scripts/rejection_trainer.py \
--selected_answer_path $res_dir/qwen2.5-1.5b-instruct/Qwen2.5-1.5B-Instruct/rejection_sampling.json \
--input_path ../data/asynchow/asynchow_natural_train.jsonl \
--model_name qwen1.5b_sft_0 \
--task asynchow_natural

# skip if the output dir is not empty
if [ -d "$res_dir/qwen2.5-1.5b-instruct/checkpoint" ]; then
    echo "Output directory is not empty, skipping training"
else
    echo "Start training qwen2.5_1.5b"
    # get new training script
    cp $cli_dir/general_template.yaml $cli_dir/qwen2.5_1.5b_sft_0_asynchow_natural.yaml
    sed -i '' "s/PRETRAINED_MODEL/Qwen\/Qwen2.5-1.5B-Instruct/g" $cli_dir/qwen2.5_1.5b_sft_0_asynchow_natural.yaml
    sed -i '' "s/DATASET_PATH/..\/data\/asynchow\/asynchow_natural_train.jsonl/g" $cli_dir/qwen2.5_1.5b_sft_0_asynchow_natural.yaml
    sed -i '' "s/OUTPUT_DIR/..\/saves\/asynchow_res\/qwen2.5-1.5b-instruct/g" $cli_dir/qwen2.5_1.5b_sft_0_asynchow_natural.yaml

    echo "Starting training baseline..."
    # Run the training command
    llamafactory-cli train $cli_dir/qwen2.5_1.5b_sft_0_asynchow_natural.yaml
    echo "Training baseline completed."
fi

# start eval
echo "Starting eval round 0..."
python ./py_scripts/asynchow_eval.py \
--pretrained $res_dir/qwen2.5-1.5b-instruct/checkpoint \
--dtype auto \
--max_tokens 2048 \
--temperature 0.7 \
--top_p 0.9 \
--record_likelihoods False \
--chat_template True \
--model_name qwen1.5b_sft_0 \
--train_set asynchow_natural \
--splits test