import sys
import argparse

sys.path.append('..')
try:
    from ..utils import *
    from .eval_boolqa import eval_qa
    from .eval_constrained_generation import eval_negation, load_negation_detector
except:
    from utils import *
    from eval_boolqa import eval_qa
    from eval_constrained_generation import eval_negation, load_negation_detector


def agreement(y_qa, y_cg, y_gt):
    cnt = 0
    agree = 0
    cnt_pos = 0
    agree_pos = 0
    cnt_neg = 0
    agree_neg = 0
    for y1, y2, yt in zip(y_qa, y_cg, y_gt):
        if y1 < 0:
            continue
        if yt == 1:
            cnt_pos += 1
        else:
            cnt_neg += 1
        if y1 == y2:
            agree += 1
            if yt == 1: # pos
                agree_pos += 1
            else:
                agree_neg += 1
        cnt += 1
    return agree / cnt, agree_pos / cnt_pos, agree_neg / cnt_neg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', '-i', type=str)
    parser.add_argument('--cg_model_name', '-g', type=str)
    parser.add_argument('--qa_model_name', '-q', type=str)
    parser.add_argument('--log', action='store_true')
    args = parser.parse_args()

    qa_score, y_qa = eval_qa(args.input_file, args.qa_model_name)
    print(qa_score)
    pipe = load_negation_detector()
    neg_scores, y_cg = eval_negation(args.cg_model_name, args.input_file, pipe)
    data = load_jsonl(args.input_file)
    y_gt = [int('pos' in x['source_kb']) for x in data]
    print(neg_scores)
    agree, agree_pos, agree_neg = agreement(y_qa, y_cg, y_gt)
    print(agree, agree_pos, agree_neg)
    cjj.lark(f"Agreement between [{args.cg_model_name}] and [{args.qa_model_name}]:\nALL: {agree}, POS: {agree_pos}, NEG: {agree_neg}\nCG score:\n{neg_scores}\nQA score:\n{qa_score}")
    
    if args.log:
        with open('metrics.log', 'a') as f:
            f.write(f"{args.cg_model_name}\t{neg_scores}\tALL: {agree}, POS: {agree_pos}, NEG: {agree_neg}\n")