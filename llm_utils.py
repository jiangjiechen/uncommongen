import os
import re
import random
import ujson as json
from utils import load_jsonl, rel2text
import cjjpy as cjj

random.seed(42)

NEG_EX_FILE = f"{cjj.AbsParentDir(__file__, '.')}/ICL_examples/neg_example_pool.jsonl"
POS_EX_FILE = f"{cjj.AbsParentDir(__file__, '.')}/ICL_examples/pos_example_pool.jsonl"

_CG_INSTRUCTIONS = {
    'question': {
        'none': [
            "Answer the question by writing a short sentence that contains correct common sense knowledge.",
        ],
        'fact': [
            "Answer the question by writing a sentence that contains correct common sense knowledge. Find a related fact to help write the sentence:",
        ],
        'logic': [
            "Answer the question by writing a sentence that contains correct common sense knowledge. Let's think step by step before writing the sentence:",
        ]
    },
    'keywords': {
        'none': [
            "Write a short and factual sentence according to commonsense based on the keywords:",
            "Use the keywords to create a short and factual sentence that accurately reflects commonsense knowledge.",
            "Create a short, factual sentence based on the keywords and what is generally accepted as true.",
            "Construct a factual and concise statement based on the provided keywords and commonsense knowledge."
        ],
        'fact': [
            "Write a short and factual sentence according to commonsense based on the keywords. Find a related fact to help write the sentence:",
        ],
        'logic': [
            "Write a short and factual sentence according to commonsense based on the keywords. Let's think step by step before writing the sentence:",
        ]
    }
}
_QA_INSTRUCTIONS = {
    'question': {
        'none': [
            "Answer the commonsense questions with yes or no:",
            "Choose \"yes\" or \"no\" to indicate whether you agree or disagree with the commonsense questions.",
            "Respond to the questions using \"yes\" or \"no\".",
            "Indicate whether the commonsense questions are correct or incorrect by writing \"yes\" or \"no\".",
        ],
        'fact': [
            "Find a related fact to about the question, then answer it with yes or no:",
        ],
        'logic': [
            "Let's think step by step to about the question, then answer it with yes or no:",
        ]
    },
    'keywords': {
        'none': [
            "Can these keywords form a truthful commonsense fact? Answer with yes or no:",
        ],
        'fact': [
            "Can these keywords form a truthful commonsense fact? Find a helpful and related fact to answer this question. Answer with yes or no:",
        ],
        'logic': [
            "Can these keywords form a truthful commonsense fact? Let's think step by step to answer this question. Answer with yes or no:",
        ]
    }
}

INSTRUCTIONS = {
    'qa': _QA_INSTRUCTIONS,
    'cg': _CG_INSTRUCTIONS
}
TASK_KEY = {
    'qa': {
        'question': 'qa_pred',
        'keywords': 'qa_pred_from_kw',
    },
    'cg': {
        'question': 'cg_pred_from_q',
        'keywords': 'cg_pred',
    }
}
INPUT_TYPE_PARAMS = {
    'qa': {
        'question': {
            'input_key': 'question',
            'input_str': 'Question',
        },
        'keywords': {
            'input_key': 'keywords',
            'input_str': 'Keywords',
        }
    },
    'cg': {
        'question': {
            'input_key': 'question',
            'input_str': 'Question',
        },
        'keywords': {
            'input_key': 'keywords',
            'input_str': 'Keywords',
        }
    }
}
OUTPUT_EXAMPLE_KEY = {  # used when loading examples
    'qa': 'answer',
    'cg': 'sentence'
}
OUTPUT_TYPE_PARAMS = {
    'qa': "Answer",
    'cg': "Sentence"
}

def load_examples(input_key, input_str, output_key, output_str, cot_key=None, cot_str=None):
    neg_ex = load_jsonl(NEG_EX_FILE)
    pos_ex = load_jsonl(POS_EX_FILE)

    def assemble_demonstration(x):
        '''
        param x: {'keywords': '...', 'sentence': '...'}
        return: 'Keywords: ... \nSentence: ...'
        '''
        input_ex = x[input_key].strip() if input_key != 'keywords' else ', '.join(
            x[input_key]).strip()
        ds = f"{input_str}: {input_ex}\n"
        if cot_key is not None and cot_str is not None:
            if x.get(cot_key) is not None:
                ds += f"{cot_str}: {x[cot_key]}\n"
            else:
                print(f'Warning: no {cot_key}:', x)
        ds += f"{output_str}: {x[output_key].strip()}"
        return ds

    neg_ex = [assemble_demonstration(x) for x in neg_ex]
    pos_ex = [assemble_demonstration(x) for x in pos_ex]

    return pos_ex, neg_ex


def prepare_prompt(task, js, k_pos_ex=1, k_neg_ex=1, key_q='text-davinci-002_ex-8', cot='none'):
    assert task in ['qa', 'cg'], task
    assert cot in ['fact', 'logic', 'none'], cot

    input_type = 'keywords' if key_q == 'keywords' else 'question'
    if cot == 'none':
        cot_key, cot_str = None, None
    elif cot == 'fact':
        cot_key, cot_str = 'cot_fact', 'Related fact'
    elif cot == 'logic':
        cot_key, cot_str = 'cot_logic', "Let's think step by step"
    else:
        raise NotImplementedError

    examples_pos, examples_neg = load_examples(input_key=INPUT_TYPE_PARAMS[task][input_type]['input_key'],
                                               input_str=INPUT_TYPE_PARAMS[task][input_type]['input_str'],
                                               output_key=OUTPUT_EXAMPLE_KEY[task], output_str=OUTPUT_TYPE_PARAMS[task] if cot == 'none' else 'Therefore',
                                               cot_key=cot_key, cot_str=cot_str)

    if key_q != 'keywords':
        q = js['boolq'][key_q].strip().lower()
    else:
        q = triple2keywords(js['subject'], js['predicate'], js['object']).lower()

    # build demonstrations with random examples
    instruct = INSTRUCTIONS[task][input_type][cot][0]  # TODO: hardcoded
    examples = sample_examples_pos_neg(examples_neg, k_neg_ex, examples_pos, k_pos_ex)
    example_text = examples_to_text(examples, '###')
    
    _cue = OUTPUT_TYPE_PARAMS[task] if cot_str is None else cot_str
    demonstration = f"{instruct}\n\n{example_text}\n{INPUT_TYPE_PARAMS[task][input_type]['input_str']}: {q}\n{_cue}:"

    return demonstration


def triple2keywords(s, p, o):
    p_text = rel2text(p)
    return f"{s}, {p_text}, {o}"


def examples_to_text(examples: list, delim="###"):
    if examples == []: 
        return ''
    else:
        return f"\n{delim}\n".join(examples) + f"\n{delim}"


def sample_examples_pos_neg(ex1: list, k1: int, ex2: list, k2: int):
    ex = random.sample(ex1, k1)
    ex += random.sample(ex2, k2)
    random.shuffle(ex)
    return ex


def save_llm_results(input_file_or_data, y_pred, task_key, model_key, output_file):
    # support (pseudo) multiprocessing: avoid overwrite, reload in-disk data
    data = load_jsonl(input_file_or_data)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as fo:
        for x, a in zip(data, y_pred):
            if x.get(task_key) is None:
                x[task_key] = {model_key: a}
            else:
                x[task_key][model_key] = a
            fo.write(json.dumps(x) + '\n')
