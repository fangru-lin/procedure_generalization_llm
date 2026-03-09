export VLLM_WORKER_MULTIPROC_METHOD=spawn
export FORCE_TORCHRUN=1
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate procedure



BASE_OUTPUT_DIR="../saves/distill_results"
res_dir="../saves/distill"
ck_dir="../saves/distill_baseline"
cli_dir="./examples/train_full"
scripts_dir="./py_scripts"

for input_path in ../data/asynchow/asynchow_natural_train.jsonl; do
    task=$(basename $input_path .jsonl)
    teacher_dir="DeepSeek-R1-Distill-Qwen-32B"
    student_dir="qwen2.5-1.5b-instruct"

    # generate samples
    python ./py_scripts/vllm_causallms.py \
        --pretrained deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
        --task $task \
        --save_path ../saves/distill/${task} \
        --input_path $input_path \
        --n 4 \
        --tensor_parallel_size 1 \
        --iter_increase_n_samples 2 \
        --max_iter 5 \
        --initialization \
        --rationalization

    # get training data
    python ./py_scripts/rejection_trainer.py \
        --selected_answer_path ../saves/distill/${task}/$teacher_dir/rejection_sampling.json \
        --model_name qwen1.5b_distill_0 \
        --input_path $input_path \
        --task $task

    # skip if the output dir is not empty
    if [ -d "../saves/distill/${task}/$student_dir" ] && [ -n "$(ls -A ../saves/distill/${task}/$student_dir)" ]; then
        echo "Output directory is not empty, skipping training"
    else
        echo "Start training qwen2.5_1.5b"
        # get new training script
        sed -e "s|MODEL_NAME|Qwen/Qwen2.5-1.5B-Instruct|" \
            -e "s|OUTPUT_DIR|../saves/distill/${task}/$student_dir|" \
            -e "s|RUN_NAME|qwen1.5b_distill_0_${task}|" \
            -e "s|DATASET|qwen1.5b_distill_0_${task}|" \
            $cli_dir/general_template.yaml \
            > $cli_dir/qwen2.5_1.5b_distill_0_${task}.yaml

        echo "Starting training baseline..."
        # Run the training command
        llamafactory-cli train $cli_dir/qwen2.5_1.5b_distill_0_${task}.yaml
        echo "Training baseline completed."
    fi

    # start eval
    echo "Starting eval round 0..."
    python ./py_scripts/asynchow_eval.py \
        --pretrained ../saves/distill/${task}/$student_dir \
        --dtype auto \
        --temperature 0.0 \
        --n 1 \
        --output_dir ../saves/distill/${task}_results \
        --record_likelihoods False \
        --chat_template True \
        --model_name "Qwen2.5-1.5B-Instruct" \
        --train_set ${task}_distill_0
done


for dataset in asynchow_distil_deepseek_qwen; do
    train_set="TRAIN_SET"
    for size in 1.5 3 7; do
        # skip training if the output dir is not empty
        if [ -d "$ck_dir/$train_set/qwen${size}b" ] && [ -n "$(ls -A $ck_dir/$train_set/qwen${size}b)" ]; then
            echo "Output directory is not empty, skipping training"
        else
            echo "Start training qwen2.5_${size}b"
            sed -e "s|MODEL_NAME| Qwen/Qwen2.5-${size}B-Instruct|" \
                -e "s|OUTPUT_DIR|$ck_dir/$train_set/qwen${size}b|" \
                -e "s|RUN_NAME|qwen${size}b_${train_set}|" \
                -e "s|DATASET|$dataset|" \
                $cli_dir/general_template.yaml \
                > $cli_dir/qwen2.5_${size}b_${train_set}.yaml
            # Run the training command
            llamafactory-cli train $cli_dir/qwen2.5_${size}b_${train_set}.yaml
        fi
        echo "Training qwen2.5_${size}b completed"

        python py_scripts/asynchow_eval.py \
            --pretrained $ck_dir/$train_set/qwen${size}b \
            --dtype auto \
            --temperature 0.0 \
            --n 1 \
            --output_dir $BASE_OUTPUT_DIR \
            --record_likelihoods False \
            --chat_template True \
            --model_name Qwen2.5-${size}B-Instruct \
            --train_set $train_set \
            --splits test
    done
done