from datetime import timedelta
import datetime as dt
import re
import torch


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


def str_to_timedelta_list(time_str: str) -> list:
    '''
    Convert a string representation of a list of timedelta objects back to a list of timedelta objects.
        Parameters:
            time_str (str): time string in the format '[datetime.timedelta(seconds=4800), datetime.timedelta(seconds=4800)]'
        Returns:
            time_delta_list (list): list of timedelta objects
    '''
    # Evaluate the string within a controlled environment
    time_delta_list = eval(time_str, {"timedelta": timedelta, "datetime": dt})
    return time_delta_list


def check_correctness_comprehensive(parsed_response: str,
                                    gt: str,
                                    task: str):
    '''
    Check the correctness of the response.
        Parameters:
            parsed_response (str): parsed response
            gt (str): ground truth
        Returns:
            correctness (bool): whether the response is correct or not
    '''
    parsed_response = find_answer_comprehensive(parsed_response)
    if isinstance(gt, torch.Tensor):
        gt = gt.item()
    gt = str(gt)
    if not parsed_response:
        return False
    if 'timedelta' not in gt:
        gold_ans = gt.replace('<answer>', '').replace('</answer>', '').strip()
        try:
            if float(gold_ans) == float(parsed_response):
                return True
            return False
        except:
            return False
    else:
        try:
            time_gt = str_to_timedelta_list(gt)
            return measure_perf(parsed_response, time_gt)[1]
        except:
            return False


# def check_correctness_comprehensive(parsed_response: str,
#                                     gt: str):
#     '''
#     Check the correctness of the response.
#         Parameters:
#             parsed_response (str): parsed response
#             gt (str): ground truth
#         Returns:
#             correctness (bool): whether the response is correct or not
#     '''
#     if not parsed_response:
#         return False
#     try:
#         time_gt = str_to_timedelta_list(gt)
#         return measure_perf(parsed_response, time_gt)[1]
#     except:
#         return False

def text_to_number_updated(sentence: str):
    # Updated mapping of number words to their numerical equivalents
    num_words = {
        "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, 'a': 1,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
        "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
        "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
        "ninety": 90, "hundred": 100, "thousand": 1000, "million": 1000000
    }

    # Helper function to convert a textual number expression to a number
    def text_number_to_num(text_num: str):
        parts = text_num.split()
        if 'and' in parts:
            parts.remove('and')

        total = 0
        current = 0

        for part in parts:
            if part in num_words:
                scale = num_words[part]
                if scale > 100:
                    current *= scale
                    total += current
                    current = 0
                elif scale == 100:
                    current *= scale
                else:
                    if current == 0:
                        current = scale
                    else:
                        current += scale
            else:
                # In case of numbers like 'forty-five'
                nums = part.split('-')
                for num in nums:
                    current += num_words.get(num, 0)

        return total + current

    # Regular expression pattern for matching text number expressions
    num_pattern = re.compile(r'\b(?:[a-zA-Z]+(?:-)?)+\b')

    # Find all matches
    matches = re.findall(num_pattern, sentence)

    # Process each match
    captured_patterns = {}
    for match in matches:
        number = text_number_to_num(match)
        if number > 0:
            captured_patterns[match] = number
            sentence = sentence.replace(match, str(number), 1)

    return sentence, captured_patterns


def measure_perf(response: str,
                 gold_timedelta: timedelta):
    text_num_set = set(["an", "one", "two", "three", "four", "five", 'a',
        "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
        "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
        "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty",
        "ninety", "hundred", "thousand", "million", "half"])
    if not response:
        return [timedelta(), timedelta()], False
    potential_answers = list()
    try:
        # if a model follows instruction
        # answer should be in double quotes, and either the first or the last one is the answer
        potential_answers = [response.lower()]
        for i, answer in enumerate(potential_answers):
            if re.findall(r'\b\w+ and a half', answer):
                pattern = re.findall(r'\b\w+ and a half', answer)[0]
                prec_word = re.findall(r'\b\w+', pattern)[0]
                if prec_word not in text_num_set:
                    answer = answer.replace(pattern, f'and half {prec_word}')

                answer = answer.replace('half a ', '0.5')
                answer = text_to_number_updated(answer)[0].replace(' and half', '.5')
                potential_answers[i] = answer
    except Exception as e:
        # if a model does not follow instruction
        # try to get response after 'is'
        try:
            answer = response.split('is ')[-1].lower().split('\n')[0]
        except:
            answer = response.lower()
        if re.findall(r'\b\w+ and a half', answer):
            pattern = re.findall(r'\b\w+ and a half', answer)[0]
            prec_word = re.findall(r'\b\w+', pattern)[0]
            if prec_word not in text_num_set:
                answer = answer.replace(pattern, f'and half {prec_word}')

            answer = answer.replace('half a ', '0.5')
            answer = text_to_number_updated(answer)[0].replace(' and half', '.5')
            potential_answers = [answer]

    if not potential_answers:
        return [timedelta(), timedelta()], False
    for answer in potential_answers:
        if '=' in answer:
            answer = answer.split('=')[-1]
        if ' or ' in answer:
            answer = answer.split(' or ')[-1]
        if '(' in answer:
            answer = answer.split('(')[0]
        timedelta_ans = [timedelta(), timedelta()]
        if ' to ' in answer:
            return [timedelta(), timedelta()], False
        try:
            time_spans = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:min|minute|minutes|hr|hour|hours|sec|second|seconds|week|weeks|day|days|month|months|year|years|s|h|m|d|w)', answer)
            for time_span in time_spans:
                time = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?', time_span)[-1].replace(',','')
                unit = re.findall(r'\b[a-z]+', time_span)[-1].strip()
                if unit in ['year', 'years', 'y']:
                    delta = [timedelta(days=float(time)*365), timedelta(days=float(time)*366)]
                elif unit in ['month', 'months', 'm']:
                    # define a loose range for month
                    # match other units in same format
                    delta = [timedelta(days=float(time)*28), timedelta(days=float(time)*31)]
                elif unit in ['week', 'weeks', 'w']:
                    delta = [timedelta(weeks=float(time)), timedelta(weeks=float(time))]
                elif unit in ['day', 'days', 'd']:
                    delta = [timedelta(days=float(time)), timedelta(days=float(time))]
                elif unit in ['hour', 'hours', 'h']:
                    delta = [timedelta(hours=float(time)), timedelta(hours=float(time))]
                elif unit in ['minute', 'min', 'minutes', 'mins']:
                    delta = [timedelta(minutes=float(time)), timedelta(minutes=float(time))]
                elif unit in ['second', 'sec', 'seconds', 'secs']:
                    delta = [timedelta(seconds=float(time)), timedelta(seconds=float(time))]
                else:
                    raise ValueError(f'unit not found: {time_span}')
                timedelta_ans[0] += delta[0]
                timedelta_ans[1] += delta[1]

            if gold_timedelta[0] <= timedelta_ans[0] <= gold_timedelta[1]:
                return timedelta_ans, True
        except Exception:
            continue
        if gold_timedelta[0] <= timedelta_ans[1] <= gold_timedelta[1]:
            return timedelta_ans, True

    return [timedelta(), timedelta()], False