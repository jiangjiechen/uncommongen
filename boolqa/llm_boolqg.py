import os
import sys
import random
import ujson as json
import numpy as np
import cjjpy as cjj

sys.path.append('..')
from gpt3_helper import prompt_gpt3, calc_cost_w_prompt
from utils import load_jsonl, rel2text, chunks_list_first
from llm_utils import examples_to_text
np.random.seed(42)
random.seed(42)

boolqg_instructions = [
    "Write a question that asks about the validity of a commonsense knowledge triple (subject, relation, object):",
]

boolqg_examples_generally = [
    "Triple: (bird, capable of, fly)\nQuestion: can most birds fly?",
    "Triple: (playing tennis, causes, feel relaxed)\nQuestion: does playing tennis generally make you feel relaxed?",
    "Triple: (room, located at, buildings)\nQuestion: are rooms usually located at buildings?",
    "Triple: (fish, capable of, sleep)\nQuestion: can fish sleep?",
    "Triple: (sheepskin, used for, writing)\nQuestion: can sheepskin be used for writing?",
    "Triple: (spicy, is a, pain)\nQuestion: is spicy a kind of pain?",
    "Triple: (water, has property, spicy)\nQuestion: is water usually spicy?",
    "Triple: (shields, made of, grass)\nQuestion: are shields generally made of grass?",
    "Triple: (penguins, is a, mammal)\nQuestion: are penguins a kind of mammal?",
    "Triple: (work, causes desire, rest)\nQuestion: does working usually make you want to rest?",
    "Triple: (sleeves, part of, shoes)\nQuestion: are sleeves a part of shoes?",
    "Triple: (books, part of, kitchen)\nQuestion: are books usually part of a kitchen?",
]

boolqg_examples = [
    "Triple: (bird, capable of, fly)\nQuestion: can birds fly?",
    "Triple: (playing tennis, causes, feel relaxed)\nQuestion: does playing tennis make you feel relaxed?",
    "Triple: (room, located at, buildings)\nQuestion: are rooms located at buildings?",
    "Triple: (fish, capable of, sleep)\nQuestion: can fish sleep?",
    "Triple: (sheepskin, used for, writing)\nQuestion: can sheepskin be used for writing?",
    "Triple: (spicy, is a, pain)\nQuestion: is spicy a kind of pain?",
    "Triple: (water, has property, spicy)\nQuestion: is water spicy?",
    "Triple: (shields, made of, grass)\nQuestion: are shields made of grass?",
    "Triple: (penguins, is a, mammal)\nQuestion: are penguins a kind of mammal?",
    "Triple: (work, causes desire, rest)\nQuestion: does working make you want to rest?",
    "Triple: (sleeves, part of, shoes)\nQuestion: are sleeves a part of shoes?",
    "Triple: (books, part of, kitchen)\nQuestion: are books part of a kitchen?",
]


def prep_triple(subj, pred, obj):
    pred = rel2text(pred)
    return f"({subj}, {pred}, {obj})"


def llm_boolqg(model_name, input_file, output_file=None, k_ex=8, batch_size=8, generally=False):
    data = load_jsonl(input_file)

    prompts = []
    for line in data:
        # TODO: triple
        triple = prep_triple(line['subject'], line['predicate'], line['object'])
        instruct = boolqg_instructions[0]
        qg_ex = boolqg_examples_generally if generally else boolqg_examples
        examples = np.random.choice(qg_ex, k_ex, replace=False).tolist()
        random.shuffle(examples)
        example_text = examples_to_text(examples, '###')
        demonstration = f"{instruct}\n\n{example_text}\nTriple:"

        prompts.append(f"{demonstration} {triple}\nQuestion:")

    y_pred = []

    res, money = prompt_gpt3(prompts, model_name=model_name, clean=True, temperature=0., max_tokens=32, batch_size=batch_size, verbose=True)
    for ny, indiv_prompt in zip(chunks_list_first(res), prompts):
        # handle n_returned sequences
        y_pred.append(ny)
        print(indiv_prompt + ny[0])
    
    general = '_general' if generally else ''
    model_key = f"{model_name}_ex-{k_ex}{general}"
    # log predicted answers by LLM $-$!!!
    if output_file is not None:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            for x, a in zip(data, y_pred):
                if isinstance(a, list): a = a[0]
                if x.get('boolq') is None:
                    x['boolq'] = {model_key: a}
                else:
                    x['boolq'][model_key] = a
                # x['prompts'] = p
                f.write(json.dumps(x) + '\n')
    
    cjj.lark(f"This run has cost you {round(money, 2)}$: {model_key}.")
    
    return y_pred


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', '-i', type=str, required=True)
    parser.add_argument('--model_name', '-m', type=str, required=True)
    parser.add_argument('--output_file', '-o', type=str, required=True)
    parser.add_argument('--k', '-k', type=int, help='number of examples', default=8, required=True)
    parser.add_argument('--batch_size', '-b', type=int, default=8)
    parser.add_argument('--generally', action='store_true')
    args = parser.parse_args()
    
    y_pred = llm_boolqg(args.model_name, args.input_file, args.output_file, k_ex=args.k, batch_size=args.batch_size, generally=args.generally)