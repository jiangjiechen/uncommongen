import os
import ujson as json
from tqdm import tqdm
import cjjpy as cjj
from multiprocessing import Pool


stopwords = cjj.LoadWords(f"{os.environ['PJ_HOME']}/preprocessing/stopwords.txt")

def load_sentences():
    prefix = f"{os.environ['PJ_HOME']}/data/corpus"
    sents = []    
    with open(f'{prefix}/omcs_sentences.txt') as f:
        sents += f.read().splitlines()
    with open(f'{prefix}/wiki1m_for_simcse.txt') as f:
        sents += f.read().splitlines()
    sents = [x.lower() for x in sents]
    return sents 


def cooccur_cnt(js):
    hit = 0
    w1 = js['subject'].lower().split()
    w2 = js['object'].lower().split()
    pairs = [(x, y) for x in w1 for y in w2 if x not in stopwords and y not in stopwords]
    for sent in sents:
        for p1, p2 in pairs:
            if p1 in sent and p2 in sent:
                hit += 1
    if pairs == []: hit = 0
    
    js['cooccur_cnt'] = hit / len(pairs)
    js['cooccur_total'] = hit
    return js


def callback(x):
    bar.update(1)
    fw.write(json.dumps(x) + '\n')


if __name__ == "__main__":
    sents = load_sentences()

    with open(f'{os.environ["PJ_HOME"]}/data/probe_datasets/true-neg-llm_test.clean.jsonl') as f:
        data = f.readlines()
        print(len(data))

    fw = open(f'{os.environ["PJ_HOME"]}/data/probe_datasets/true-neg-llm_test.clean.cooccur.jsonl', 'w')
    p = Pool(16)
    bar = tqdm(total=len(data))
    for line in data:
        js = json.loads(line)
        p.apply_async(cooccur_cnt, (js,), callback=callback)
    p.close()
    p.join()
    fw.close()
    
