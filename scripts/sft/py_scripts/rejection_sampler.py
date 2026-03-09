import argparse
import re
import json
import logging
import random
from pathlib import Path
import numpy as np
from scripts.sft.py_scripts.vllm_causallms import *
import torch

num_gpus = torch.cuda.device_count()
print(f"Number of available GPUs: {num_gpus}")

eval_logger = logging.getLogger(__name__)


def extract_answer(output: str) -> Optional[int]:
    """Extract an integer answer from the generated output."""
    match = re.search(r'answer is (\d+)', output, re.IGNORECASE)
    if match:
        return int(match.group(1))
    # match yes or no
    if "answer is yes" in output:
        return "yes"
    else:
        assert "answer is no" in output, ("Exceptional case")
        return "no"


def eval_instance(output: str, ground_truth, task):
    """Extract an integer answer from the generated output."""
    if task == 'prototypical':
        ground_truth = extract_answer(ground_truth)
        binary = type(ground_truth)==str
        if not binary:
            match = re.search(r'is (\d+)', output, re.IGNORECASE)
            if match:
                return int(match.group(1))==ground_truth
        else:
            lower_output = set(output.lower().split())
            # match yes or no
            if 'no.' not in lower_output and 'no,' not in lower_output and 'no' not in lower_output:
                return 'yes' == ground_truth
            else:
                return 'no' == ground_truth
        return False
    elif task == 'explagraph':
        if '<answer>support</answer>' in output and '<answer>counter</answer>' not in output:
            if '<answer>support</answer>' == ground_truth:
                return True
        elif '<answer>counter</answer>' in output and '<answer>support</answer>' not in output:
            if '<answer>counter</answer>' == ground_truth:
                return True
        else:
            return False

    else:
        raise Exception(f"Task {task} not implemented")


def eval_generation(outputs, inputs, return_samples=False, round_acc=True):
    labels = [extract_answer(dp['output']) for dp in inputs]
    correct = [[eval_instance(outputs[i][j], labels[i]) for j in range(len(outputs[i]))] for i in range(len(outputs))]
    acc = np.mean([i for c in correct for i in c])
    if round_acc:
        acc = round(acc, 3)
    print(f'Current model accuracy is :{acc}')
    if return_samples:
        return correct, acc
    return acc


def rejection_sampler(outputs, eval_res, n_per_prompt=2, fill_with_false=False):
    """
    Selects up to n_per_prompt generations per prompt based on boolean evaluation results.
    
    For each prompt:
      - First, select all generations with a True evaluation.
      - If the number of True outputs is less than n_per_prompt, fill up the selection
        with the remaining (False) outputs, preserving their original order.
      - If there are more True outputs than n_per_prompt, randomly sample n_per_prompt of them.
    
    Args:
        outputs (dict): Dictionary containing 'generated_outputs', a list of lists (one per prompt).
        eval_res (list of lists): Boolean evaluation matrix corresponding to each generation.
        n_per_prompt (int): Maximum number of selected generations per prompt.
    
    Returns:
        List of lists: A list (per prompt) containing the selected generations.
    """
    selected_samples = []
    unfilled_idxes = []
    
    # Loop over each prompt's generated outputs and its evaluation results.
    for prompt_idx, gen_outputs in enumerate(outputs):
        # Extract the boolean evaluation results for this prompt.
        eval_row = eval_res[prompt_idx]
        
        # First, select the outputs evaluated as True.
        accepted = [gen for gen, flag in zip(gen_outputs, eval_row) if flag]
        
        # If there are fewer than n_per_prompt True outputs, fill up with the False ones.
        if len(accepted) < n_per_prompt and fill_with_false:
            false_outputs = [gen for gen, flag in zip(gen_outputs, eval_row) if not flag]
            num_needed = n_per_prompt - len(accepted)
            accepted.extend(false_outputs[:num_needed])
        
        # If there are more than n_per_prompt accepted outputs, randomly sample n_per_prompt.
        elif len(accepted) > n_per_prompt:
            random.seed(1234)
            accepted = random.sample(accepted, n_per_prompt)
        
        selected_samples.append(accepted)
    
    return selected_samples, unfilled_idxes


def main():
    parser = argparse.ArgumentParser(description="VLLM Model Inference")
    parser.add_argument("--pretrained", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Pretrained model identifier.")
    parser.add_argument("--input_path", type=str, default="../data/expla_graph_train_sft.json",
    help="Input file path")
    parser.add_argument("--dtype", type=str, default="auto", help="Data type to use (e.g., float16, bfloat16, float32, auto).")
    parser.add_argument("--revision", type=str, default=None, help="Model revision if needed.")
    parser.add_argument("--temperature", type=float, default=1, help="Temperature for sampling.")
    parser.add_argument("--n", type=int, default=8, help="Number of best completions to return.")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code for custom models.")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer identifier (defaults to pretrained model's tokenizer).")
    parser.add_argument("--tokenizer_mode", type=str, default="auto", help="Tokenizer mode.")
    parser.add_argument("--tokenizer_revision", type=str, default=None, help="Tokenizer revision if needed.")
    parser.add_argument("--add_bos_token", action="store_true", help="Whether to add beginning-of-sequence token.")
    parser.add_argument("--prefix_token_id", type=int, default=None, help="Custom prefix token id.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--quantization", type=str, default=None, help="Quantization scheme, if any.")
    parser.add_argument("--max_gen_toks", type=int, default=16384, help="Maximum tokens to generate.")
    parser.add_argument("--swap_space", type=int, default=4, help="Swap space value for the model.")
    parser.add_argument("--batch_size", type=str, default="auto", help="Batch size (or 'auto').")
    parser.add_argument("--max_model_len", type=int, default=16384, help="Maximum model sequence length.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for generation.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.3, help="Fraction of GPU memory to utilize.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model (e.g., cuda).")
    parser.add_argument("--data_parallel_size", type=int, default=1, help="Data parallelism size.")
    parser.add_argument("--lora_local_path", type=str, default=None, help="Path to LoRA adapters if using them.")
    parser.add_argument("--save_path", type=str, default="../saves/rejection_sampling", help="Path to store log probs for generation")
    parser.add_argument("--chat_template", type=bool, default=True, help="Apply chat template to the input")
    parser.add_argument("--enforce_eager", type=bool, default=True, help="Enforce eager execution")
    parser.add_argument("--max_num_seqs", type=int, default=256, help="Maximum number of sequences to generate")
    parser.add_argument("--record_likelihoods", type=bool, default=True, help="Record likelihoods for each generation")
    parser.add_argument("--task", type=str, default='explagraph', help="Task to perform")
    parser.add_argument("--rejection_sampling", type=bool, default=True, help="Whether to perform rejection sampling")

    args = parser.parse_args()
    print(args)
    with open(args.input_path, 'r') as f:
        input_file = json.load(f)
    prompts = [sample['instruction'] for sample in input_file]

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
    sampling_params = {'n': args.n,
    'temperature': args.temperature}
    # prompts = ['The capital of France is?', 'The capital of Belgian is?']

    if args.record_likelihoods:
        generated_outputs, likelihoods = model.generate_with_loglikelihood(prompts, sampling_params=sampling_params, chat_template=args.chat_template, record_likelihood=args.record_likelihoods)
        outputs, re_ord = likelihoods
        # make dir if it does not exist
        Path(args.save_path).mkdir(parents=True, exist_ok=True)
        # save the results
        print(len(re_ord))
        log_probs = []
        for i in range(len(generated_outputs)): 
            log_probs.append([o[0] for o in re_ord[i*len(generated_outputs[0]):(i+1)*len(generated_outputs[0])]])
        
        with open(args.save_path+'/test_graph_proto_qwen_2.5_7b_instruct_all_logs.json', 'w') as f:
            json.dump({'re_ord':re_ord, 'prompts':prompts, 'generated_outputs':generated_outputs, 'log_probs':log_probs, "logs":str(outputs)}, f)
        with open(args.save_path+'/graph_proto_qwen_2.5_7b_instruct_output_likelihood.json', 'w') as f:
            json.dump({'prompts':prompts, 'generated_outputs':generated_outputs, 'log_probs':log_probs}, f)
    else:
        assert args.rejection_sampling
        generated_outputs = model.generate(prompts, sampling_params=sampling_params, chat_template=args.chat_template)
        eval_res, acc = eval_generation(generated_outputs, prompts, task=args.task)
        selected_samples, unfilled_idxes = rejection_sampler(generated_outputs, eval_res)
        # save results
        with open(args.save_path+"/rejection_sampling.json", "w") as f:
            json.dump({"selected_samples": selected_samples, "unfilled_idxes": unfilled_idxes}, f)


if __name__ == "__main__":
    main()