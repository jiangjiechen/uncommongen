import sys
from accelerate import Accelerator
import cjjpy as cjj

sys.path.append(cjj.AbsParentDir(__file__, '..'))
from gpt3_helper import prompt_gpt3, calc_cost_w_prompt
# from flant5_helper import prompt_flant5
from utils import load_jsonl, chunks_list_first, answer2bool
from llm_utils import (
    save_llm_results, 
    prepare_prompt,
    TASK_KEY
)

accelerator = Accelerator()


def judge_ambi_w_prob(prompt, prompt_func, **kwargs):
    # TODO: flan-t5 prob, but flan-t5 will only output yes/no.
    results, money = prompt_func([prompt + ' yes', prompt + 'no'], echo=True, clean=False, logprobs=1, **kwargs)
    results = list(results)
    prob_yes = results[0]['logprobs']['token_logprobs'][-1]
    prob_no = results[1]['logprobs']['token_logprobs'][-1]
    return ' yes' if prob_yes > prob_no else ' no'


def llm_boolqa(model_name, input_file, output_file=None, key_q='text-davinci-002_ex-8', k_pos_ex=2, k_neg_ex=2, batch_size=8, cot='none'):
    data = load_jsonl(input_file)
    prompts = []
    for js in data:
        prompt = prepare_prompt('qa', js, k_pos_ex, k_neg_ex, key_q, cot)
        prompts.append(prompt)
    print(prompts[0])
    
    num_cot_tokens = 40 if cot is not None else 0   # TODO: approximate number of cot tokens
    if 'flan-t5' in model_name:
        prompt_func = prompt_flant5
    else:
        prompt_func = prompt_gpt3
    res, money = prompt_func(prompts, model_name=model_name, clean=True, temperature=0., 
                             max_tokens=5 + num_cot_tokens, verbose=True, batch_size=batch_size, n=1)
    
    y_pred = []
    for i, (ny, indiv_prompt) in enumerate(zip(chunks_list_first(res, n=1), prompts)):
        # handle n_returned sequences, where n = 1 in QA.
        y = ny[0]
        if answer2bool(y, 'Answer' if cot != 'none' else None) < 0 and 'flan-t5' not in model_name: # if y is undecidable, use probability
            y = judge_ambi_w_prob(indiv_prompt, prompt_func, model_name=model_name, temperature=0., max_tokens=0, batch_size=batch_size, n=1)
            money += calc_cost_w_prompt(indiv_prompt + y + indiv_prompt + y, model_name)
        y_pred.append(y)
        if i == 0: print(indiv_prompt + y)

    if args.instruct_id == 0:
        model_key = f"{model_name}_ex-{k_pos_ex}p{k_neg_ex}n"
    else:
        model_key = f"{model_name}_ex-{k_pos_ex}p{k_neg_ex}n_i{args.instruct_id}" # TODO: hard-coded
    
    if args.temperature == 0:
        model_key = f"{model_name}_ex-{k_pos_ex}p{k_neg_ex}n"
    else:
        model_key = f"{model_name}_ex-{k_pos_ex}p{k_neg_ex}n_t{args.temperature}" # TODO: hard-coded
    
    input_type = 'keywords' if key_q == 'keywords' else 'question'
    task_key = TASK_KEY['qa'][input_type]
    if cot != 'none':
        task_key += f'_{cot}'
    
    if accelerator.is_main_process:
        # save into file, override previous model key.
        save_llm_results(input_file, y_pred, task_key, model_key, output_file)

        if 'bloom' not in model_name or 'flan-t5' not in model_name:
            cjj.lark(f"This run has cost you {round(money, 2)}$: {task_key}/{model_key}.")
    
    return f"{task_key}/{model_key}"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', '-i', type=str, required=True)
    parser.add_argument('--model_name', '-m', type=str, required=True, 
                        choices=['flan-t5-large', 'flan-t5-xl', 'flan-t5-xxl',
                                 'davinci', 'curie', 'babbage', 'ada', 
                                 'text-davinci-001', 'text-curie-001', 'text-babbage-001', 'text-ada-001',
                                 'text-davinci-002', 'text-davinci-003', 'code-davinci-002'])
    parser.add_argument('--posk', type=int, help='Number of positive examples in the demonstration.')
    parser.add_argument('--negk', type=int, help='Number of negative examples in the demonstration.')
    parser.add_argument('--output_file', '-o', type=str, required=True, 
                        help='Note: just for convenience, input and output file are the same.')
    parser.add_argument('--input_key', '-q', default='text-davinci-002_ex-8', type=str, 
                        help='Key for input questions (in QA) in the jsonline file. InstructGPT-002 generated questions by default.')
    parser.add_argument('--batch_size', '-b', type=int, default=16)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--cot', type=str, choices=['fact', 'logic', 'none'], default='none')
    parser.add_argument('--instruct_id', type=int, default=0, 
                        help='For testing different instructions. Use the first instruction by default.')
    args = parser.parse_args()
    
    y_pred = llm_boolqa(args.model_name, args.input_file, args.output_file, 
                        args.input_key, k_pos_ex=args.posk, k_neg_ex=args.negk, 
                        batch_size=args.batch_size, cot=args.cot)