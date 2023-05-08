import sys
import re
import argparse 
from tqdm import tqdm
import torch
from collections import defaultdict, Counter
from transformers import pipeline
sys.path.append('..')

import cjjpy as cjj
from utils import *


negation_cue = {'nothing', "no", "not", "never", "none", "hardly", "rarely", "scarcely", "seldom", 'barely',
                "nor", "neither", "nothing", "nowhere", "without", "lack", "cant", "dont", "doesnt",
                "doesn't", "don't", "isn't", "wasn't", "aren't", "weren't", "haven't", "hasn't", 
                "shouldn't", "won't", "wouldn't", "can't", "couldn't", "cannot", "unable"}


def majority_vote(pred_list):
    return [Counter(x).most_common()[0][0] for x in pred_list]


def fetch_target_sent(x, prefix):
    # prefix = None: ignore prefix, use the first sentence
    # prefix = xxxx: find the first of that pattern
    if prefix is not None:
        for subx in x.strip().replace('###', '\n').split('\n'):
            # find the sentence after prefix
            sent = re.findall(f'{prefix}(.*)', subx)
            if sent != []:
                return sent[0].strip()
    return x.strip().split('\n')[0].strip()


def detect_negation(pipe, texts: list, prefix=None, only_rule=False):
    # True for neg, False for pos
    def _has_neg(x):
        for j in x:
            if j['entity'] == 'Y':
                return True
        return False
    
    labels = []
    for t in texts:
        rule_hit_flag = False
        t1 = fetch_target_sent(t, prefix)
        
        if only_rule:
            label = False
            for w in t1.split():
                if w.lower() in negation_cue:
                    label = True
                    break
            labels.append(label)
        else:
            # filtering with rules first
            for w in t1.split():
                if w.lower() in negation_cue:
                    labels.append(True)
                    rule_hit_flag = True
                    break
            
            if not rule_hit_flag:
                labels.append(_has_neg(pipe(t1)))
    
    return labels


def eval_negation(model_name, filename, pipe):
    data = load_jsonl(filename)
    y_pred = []
    y_true = []
    y_pred_by_rel = defaultdict(list)
    y_true_by_rel = defaultdict(list)

    if '/' in model_name:
        task_key = model_name.split('/')[0]
        model_key = '/'.join(model_name.split('/')[1:])
    else:
        task_key = 'cg_pred'
        model_key = model_name
    
    for js in tqdm(data, desc='Negation Detection'):
        if js.get('label') is None:
            _label = int('pos' in js['source_kb'])
        else:
            _label = js['label']
        y_true.append(_label)
        y_true_by_rel[js['predicate']].append(_label)
        
        prefix = 'Therefore:' if 'logic' in task_key or 'fact' in task_key else None
        _pred = detect_negation(pipe, js[task_key][model_key], prefix=prefix)
        _pred = [int(not x) for x in _pred]
        
        y_pred.append(_pred)
        y_pred_by_rel[js['predicate']].append(_pred)
    
    y_score = majority_vote(y_pred)
    scores = calc_biclf_metrics(y_score, y_true)
    
    return scores, y_score


def load_negation_detector():
    pipe = pipeline('token-classification', model='spoiled/roberta-large-condaqa-neg-tag-token-classifier', device=0 if torch.cuda.is_available() else -1)
    return pipe


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', '-i')
    parser.add_argument('--model_name', '-m')
    parser.add_argument('--log', action='store_true')
    args = parser.parse_args()

    pipe = load_negation_detector()
    scores, neg_pred = eval_negation(args.model_name, args.input_file, pipe)
    print(scores)
    cjj.lark(f"CG - {args.model_name} in {args.input_file}: {str(scores)}")
    
    if args.log:
        with open('metrics.log', 'a') as f:
            f.write(f"{args.model_name}\t{scores}\tNone\n")