# -*- coding: utf-8 -*-

'''
@Author : Jiangjie Chen
@Time   : 2023/1/2 21:00
@Contact: jjchen19@fudan.edu.cn
'''

import os
import openai
import math
import sys
import time
from tqdm import tqdm
from typing import Iterable, List, TypeVar

T = TypeVar('T')
KEY_INDEX = 0
KEY_POOL = [os.environ["OPENAI_API_KEY"]] # your key pool
openai.api_key = KEY_POOL[0]


def batchify(data: Iterable[T], batch_size: int) -> Iterable[List[T]]:
    # function copied from allenai/real-toxicity-prompts
    assert batch_size > 0
    batch = []
    for item in data:
        # Yield next batch
        if len(batch) == batch_size:
            yield batch
            batch = []
        batch.append(item)

    # Yield last un-filled batch
    if len(batch) != 0:
        yield batch


def openai_unit_price(model_name):
    if 'gpt-3.5-turbo' in model_name:
        unit = 0.002
    elif 'davinci' in model_name:
        unit = 0.02
    elif 'curie' in model_name:
        unit = 0.002
    elif 'babbage' in model_name:
        unit = 0.0005
    elif 'ada' in model_name:
        unit = 0.0004
    else:
        unit = -1
    return unit


def calc_cost_w_tokens(total_tokens: int, model_name: str):
    unit = openai_unit_price(model_name)
    return round(unit * total_tokens / 1000, 2)


def calc_cost_w_prompt(prompt: str, model_name: str):
    # 750 words == 1000 tokens
    unit = openai_unit_price(model_name)
    return round(len(prompt.split()) / 750 * unit, 2)


def get_perplexity(logprobs):
    assert len(logprobs) > 0, logprobs
    return math.exp(-sum(logprobs)/len(logprobs))


def keep_logprobs_before_eos(tokens, logprobs):
    keep_tokens = []
    keep_logprobs = []
    start_flag = False
    for tok, lp in zip(tokens, logprobs):
        if start_flag:
            if tok == "<|endoftext|>":
                break
            else:
                keep_tokens.append(tok)
                keep_logprobs.append(lp)
        else:
            if tok != '\n':
                start_flag = True
                if tok != "<|endoftext>":
                    keep_tokens.append(tok)
                    keep_logprobs.append(lp)

    return keep_tokens, keep_logprobs


def catch_openai_api_error(prompt_input: list):
    global KEY_INDEX
    error = sys.exc_info()[0]
    if error == openai.error.InvalidRequestError:
        # something is wrong: e.g. prompt too long
        print(f"InvalidRequestError\nPrompt:\n\n{prompt_input}\n\n")
        assert False
    elif error == openai.error.RateLimitError:
        KEY_INDEX = (KEY_INDEX + 1) % len(KEY_POOL)
        openai.api_key = KEY_POOL[KEY_INDEX]
        print("RateLimitError, now change the key.")
    else:
        print("API error:", error)


def prompt_gpt3(prompt_input: list, model_name='text-davinci-003', max_tokens=128,
                clean=False, batch_size=16, verbose=False, **kwargs):
    # return: output_list, money_cost

    def request_api(prompts: list):
        # prompts: list or str
        while True:
            try:
                return openai.Completion.create(
                    model=model_name,
                    prompt=prompts,
                    max_tokens=max_tokens,
                    # temperature=kwargs.get('temperature', 0.9),
                    # top_p=kwargs.get('top_p', 1),
                    # frequency_penalty=kwargs.get('frequency_penalty', 0),
                    # presence_penalty=kwargs.get('presence_penalty', 0),
                    **kwargs
                )
            except:
                catch_openai_api_error(prompt_input)
                time.sleep(1)

    total_tokens = 0
    results = []
    for batch in tqdm(batchify(prompt_input, batch_size), total=len(prompt_input) // batch_size):
        batch_response = request_api(batch)
        total_tokens += batch_response['usage']['total_tokens']
        if not clean:
            results += batch_response['choices']
        else:
            results += [choice['text'] for choice in batch_response['choices']]

    return results, calc_cost_w_tokens(total_tokens, model_name)


def prompt_chatgpt(system_input, user_input, history=[], model_name='gpt-3.5-turbo'):
    '''
    :param system_input: "You are a helpful assistant/translator."
    :param user_input: you texts here
    :param history: ends with assistant output.
                    e.g. [{"role": "system", "content": xxx},
                          {"role": "user": "content": xxx},
                          {"role": "assistant", "content": "xxx"}]
    return: assistant_output, (updated) history, money cost
    '''
    if len(history) == 0:
        history = [{"role": "system", "content": system_input}]
    history.append({"role": "user", "content": user_input})

    for _ in range(5):
        try:
            completion = openai.ChatCompletion.create(
                model=model_name,
                messages=history
            )
            break
        except:
            catch_openai_api_error()
            time.sleep(1)

    assistant_output = completion['choices'][0]['message']['content']
    history.append({"role": "assistant", "content": assistant_output})
    total_tokens = completion['usage']['total_tokens']

    return assistant_output, history, calc_cost_w_tokens(total_tokens, model_name)


if __name__ == '__main__':
    prompt = 'hello world'
    response = prompt_gpt3([prompt]*4, batch_size=4, clean=True)
    print(response)
    response = prompt_chatgpt('You are a super villian.', 'What is your plan when get caught?')
    print(response)