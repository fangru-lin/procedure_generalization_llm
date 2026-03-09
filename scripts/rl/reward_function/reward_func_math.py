import re
import torch
import logging
# Configure logging
logging.basicConfig(level=logging.DEBUG,
    filemode='w',  # optional, use 'a' to append instead of overwriting
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_results(result, ground_truth) -> int:
    indices = [pos for pos, char in enumerate(result) if char == "$"]
    if len(indices) <= 1:
        answer = result
    else:
        answer = result[indices[0] + 1 : indices[-1]]

    if is_equiv(answer, ground_truth):
        return 1
    else:
        return 0


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string

def find_answer_comprehensive(answer):
    '''
    Find the answer from the response.
        Parameters:
            answer (str): response
    '''
    try:
        ans = re.findall(r'<answer>(.*?)</answer>', answer, re.DOTALL)[-1].strip().lower()
        return ans
    except:
        return ''


def check_correctness_other(output, ground_truth):
    ans = find_answer_comprehensive(output)
    try:
        res = process_results(ans, ground_truth)
        assert int(res) in [0, 1]
        return int(res)
    except:
        return 0


def reward_func(queries, prompts, labels, kl_div=None, kl_coef=0.1, action_mask=None, reward_clip_range=None, use_kl_loss=False):
    """Compute rewards for asynchow responses."""
    try:
        logger.info(f"Starting reward computation for {len(queries)} queries")
        logger.debug(f"Sample query: {queries[0][:100]}...")
        logger.debug(f"Sample prompt: {prompts[0][:100]}...")
        logger.debug(f"Sample label: {labels[0]}")
        
        # Pre-compute prompt length
        prompt_len = len(prompts[0]) if prompts else 0
        logger.debug(f"Prompt length: {prompt_len}")
        
        # Vectorized reward computation
        rewards = []
        for i, (query, label) in enumerate(zip(queries, labels)):
            try:
                response = query[prompt_len:].strip()
                reward = check_correctness_other(response, label)
                rewards.append(reward)
                if i < 3:  # Log first few examples for debugging
                    logger.debug(f"Query {i}: response={response[:500]}..., label={label}, reward={reward}")
            except:
                logger.error(f"Error processing query {i}")
                rewards.append(0)
        
        # Convert all rewards to tensor at once
        rewards = torch.tensor(rewards, dtype=torch.float32)
        logger.info(f"Computed rewards: mean={rewards.mean():.3f}, non-zero={torch.count_nonzero(rewards)}/{len(rewards)}")
        
        # Apply reward clipping if specified
        if reward_clip_range is not None:
            rewards = rewards.clamp(min=reward_clip_range[0], max=reward_clip_range[1])
            logger.debug(f"Clipped rewards: {rewards}")

        # Handle KL penalty
        if action_mask is not None and kl_div is not None:
            if not use_kl_loss and kl_coef > 0:
                kl_reward = -kl_coef * kl_div
                eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)
                last_reward = torch.zeros_like(kl_div).scatter_(dim=1, index=eos_indices, src=rewards.unsqueeze(1).to(kl_div.dtype))
                rewards = last_reward + kl_reward
                logger.debug(f"Applied KL penalty with mask: final rewards shape {rewards.shape}")
        elif kl_div is not None and not use_kl_loss and kl_coef > 0:
            rewards = rewards - kl_coef * kl_div
            logger.debug(f"Applied KL penalty without mask: final rewards shape {rewards.shape}")
        
        return rewards
    except Exception as e:
        logger.error(f"Error in reward_func: {e}")
        # Return zero rewards as fallback
        return torch.zeros(len(queries) if queries else 1)
