import argparse
import sys
import random
from collections import Counter
sys.path.append('..')
import cjjpy as cjj
from utils import answer2bool, load_jsonl, calc_biclf_metrics

random.seed(42)


def eval_qa(file_or_data, model_name):
    data = load_jsonl(file_or_data)
    y_pred = []
    y_true = []
    i = 0
    for x in data:
        if '/' in model_name:
            task_key = model_name.split('/')[0]
            model_key = '/'.join(model_name.split('/')[1:])
        else:
            task_key = 'qa_pred'
            model_key = model_name
        if 'fact' in task_key or 'logic' in task_key:
            eval_cot = True
        else:
            eval_cot = False
        pred = x[task_key][model_key]
        if isinstance(pred, str): # TODO: majority vote QA
            pred = [pred]
        
        vote_pred = []
        for p in pred:
            bool_pred = answer2bool(p, prefix='Answer' if eval_cot else None)
            vote_pred.append(bool_pred)
        
        voted_pred = Counter(vote_pred).most_common()[0][0]
        y_pred.append(voted_pred)
        if voted_pred < 0: 
            # bool_pred = random.choice([0, 1])
            print(f"[{i}] Undecided: {x}")
            i += 1

        if x.get('label') is None:
            _label = int('pos' in x['source_kb'])
        else:
            _label = x['label']
        y_true.append(_label)
    
    y_pred_f, y_true_f = [], []
    for y1, y2 in zip(y_pred, y_true):
        if y1 >= 0:
            y_pred_f.append(y1)
            y_true_f.append(y2)
    scores = calc_biclf_metrics(y_pred_f, y_true_f)
    return scores, y_pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', '-i', type=str)
    parser.add_argument('--qa_model', '-m', type=str, help='nested or not: `qa_pred_cot_fact/text-davinci-002_ex-8p8n` or `text-davinci-002_ex-8p8n`')
    parser.add_argument('--log', action='store_true')
    args = parser.parse_args()

    scores, y_pred = eval_qa(args.input_file, args.qa_model)
    print(scores)
    cjj.lark(f"QA - {args.qa_model} in {args.input_file}: {str(scores)}")

    if args.log:
        with open('metrics.log', 'a') as f:
            f.write(f"{args.qa_model}\t{scores}\tNone\n")