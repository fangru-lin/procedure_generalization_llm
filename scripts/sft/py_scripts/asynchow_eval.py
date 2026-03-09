from scripts.sft.py_scripts.asynchow_utils import *
from datasets import load_dataset
from scripts.sft.py_scripts.vllm_causallms import *
import json
import argparse


def gen_and_eval(model,
                 prompts,
                 ground_truths,
                 task,
                 out_dir,
                 model_name,
                 sampling_params,
                 chat_template=True,
                 record_likelihoods=False,
                 return_samples=True,
                 rewrite=False):
    if not Path(os.path.join(out_dir, task, model_name)).exists():
        os.makedirs(os.path.join(out_dir, task, model_name))
    else:
        if not rewrite and Path(os.path.join(out_dir, task, model_name, 'acc.json')).exists():
            return
    generated_outputs = model.generate_with_loglikelihood(prompts,
                                                        sampling_params=sampling_params,
                                                        chat_template=chat_template,
                                                        record_likelihood=record_likelihoods)
    with open(os.path.join(out_dir, task, model_name, 'generated_outputs.json'), 'w') as f:
        json.dump(generated_outputs, f)
    if return_samples:
        eval_res, acc = eval_generation(generated_outputs, ground_truths, task=task, return_samples=return_samples)
        with open(os.path.join(out_dir, task, model_name, 'eval_res.json'), 'w') as f:
            json.dump(eval_res, f)
        with open(os.path.join(out_dir, task, model_name, 'acc.json'), 'w') as f:
            json.dump(acc, f)
    else:
        acc = eval_generation(generated_outputs, ground_truths, task=task, return_samples=return_samples)
        with open(os.path.join(out_dir, task, model_name, 'acc.json'), 'w') as f:
            json.dump(acc, f)
    print(f"Finished evaluating {model_name} on {task}")


def get_store_dir_task(dataset_name):
    if "asynchow/asynchow_natural" in dataset_name:
        store_dir="test_natural"
        task="asynchow"
    elif "asynchow/asynchow_aave" in dataset_name:
        store_dir="test_aave"
        task="asynchow"
    elif "asynchow_protograph_test" in dataset_name:
        store_dir="test_proto_graph"
        task="proto_graph"
    elif "asynchow/asynchow_python" in dataset_name:
        store_dir="test_python_graph"
        task="proto_graph"
    elif "graph_ablation" in dataset_name:
        store_dir="graph_ablation"
        task="proto_graph"
    elif "python_ablation" in dataset_name:
        store_dir="python_ablation"
        task="proto_graph"
    elif "math" in dataset_name:
        store_dir="math"
        task = "math"
    elif "physics" in dataset_name:
        store_dir="physics"
        task = "physics"
    else:
        store_dir="test_graph"
        task="asynchow"
    
    return store_dir, task

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Pretrained model identifier.")
    parser.add_argument("--dtype", type=str, default="auto", help="Data type to use (e.g., float16, bfloat16, float32, auto).")
    parser.add_argument("--revision", type=str, default=None, help="Model revision if needed.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling.")
    parser.add_argument("--n", type=int, default=1, help="Number of best completions to return.")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code for custom models.")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer identifier (defaults to pretrained model's tokenizer).")
    parser.add_argument("--tokenizer_mode", type=str, default="auto", help="Tokenizer mode.")
    parser.add_argument("--tokenizer_revision", type=str, default=None, help="Tokenizer revision if needed.")
    parser.add_argument("--add_bos_token", action="store_true", help="Whether to add beginning-of-sequence token.")
    parser.add_argument("--prefix_token_id", type=int, default=None, help="Custom prefix token id.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--quantization", type=str, default=None, help="Quantization scheme, if any.")
    parser.add_argument("--max_gen_toks", type=int, default=8000, help="Maximum tokens to generate.")
    parser.add_argument("--swap_space", type=int, default=4, help="Swap space value for the model.")
    parser.add_argument("--batch_size", type=str, default="auto", help="Batch size (or 'auto').")
    parser.add_argument("--max_model_len", type=int, default=16384, help="Maximum model sequence length.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for generation.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="Fraction of GPU memory to utilize.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model (e.g., cuda).")
    parser.add_argument("--data_parallel_size", type=int, default=1, help="Data parallelism size.")
    parser.add_argument("--lora_local_path", type=str, default=None, help="Path to LoRA adapters if using them.")
    parser.add_argument("--save_path", type=str, default="../saves/rejection_sampling/sample_32", help="Path to store log probs for generation")
    parser.add_argument("--chat_template", type=bool, default=True, help="Apply chat template to the input")
    parser.add_argument("--enforce_eager", type=bool, default=True, help="Enforce eager execution")
    parser.add_argument("--max_num_seqs", type=int, default=128, help="Maximum number of sequences to generate")
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'test'], help="List of splits to evaluate on")
    parser.add_argument('--record_likelihoods', type=bool, default=False, help="Record likelihoods for each generation")
    parser.add_argument('--rewrite', type=bool, default=False, help="Rewrite existing results")
    parser.add_argument('--model_name', type=str, required=True, help="Model name")
    parser.add_argument('--datasets', type=str, nargs='+', default=['../data/asynchow/asynchow_natural', '../data/asynchow/asynchow_graph', '../data/asynchow/asynchow_aave', '../data/asynchow/asynchow_python'], help="List of datasets to evaluate on")
    parser.add_argument('--train_set', type=str, required=True, help="Train setting name")
    parser.add_argument('--domains', type=str, nargs='+', default=['math', 'physics'], help="List of domains to evaluate on")
    args = parser.parse_args()

    model = VLLM(
        pretrained=args.pretrained,
        dtype=args.dtype,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
        tokenizer=args.tokenizer,
        tokenizer_mode=args.tokenizer_mode,
        tokenizer_revision=args.tokenizer_revision,
        add_bos_token=args.add_bos_token,
        prefix_token_id=args.prefix_token_id,
        tensor_parallel_size=args.tensor_parallel_size,
        quantization=args.quantization,
        max_gen_toks=args.max_gen_toks,
        swap_space=args.swap_space,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        device=args.device,
        data_parallel_size=args.data_parallel_size,
        lora_local_path=args.lora_local_path,
        enforce_eager=args.enforce_eager,
        max_num_seqs=args.max_num_seqs
    )
    sampling_params = {'n': args.n, 'temperature': args.temperature}

    for split in args.splits:
        for dataset_name in args.datasets:
            try:
                curr_dataset = load_dataset("json", data_files=dataset_name+'_'+split+'.jsonl')
            except Exception as e:
                print(f"Failed to load dataset {dataset_name} for split {split}: {e}")
                continue
            ground_truths = curr_dataset['answer']
            prompts = curr_dataset['question']
            store_dir, task = get_store_dir_task(dataset_name)
            out_dir = os.path.join(args.output_dir, args.model_name, split, args.train_set, store_dir)
            # skip if output_dir exists and is not empty
            if Path(out_dir).exists() and len(os.listdir(out_dir)) > 0 and not args.rewrite:
                print(f"Output directory {out_dir} already exists and is not empty. Skipping...")
                continue
            gen_and_eval(model,
                 prompts,
                 ground_truths,
                 task,
                 out_dir,
                 args.model_name,
                 sampling_params,
                 return_samples=True,
                 rewrite=args.rewrite)


if __name__ == "__main__":
    main()
