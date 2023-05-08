import os
import ujson as json


pronouns = ['me', 'i', 'I', 'her', 'she', 'him', 'he', 'us', 'we', 'them', 'they', 'it', 'its', 'my', 'your', 'his', 'her', 'their', 'our', 'mine', 'yours', 'hers', 'theirs', 'ours', 'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'yourselves', 'themselves', 'this', 'that', 'these', 'those', 'who', 'whom', 'whose', 'which', 'what', 'where', 'when', 'why', 'how', 'there', 'here', 'anybody', 'anyone', 'anything', 'each', 'either', 'everybody', 'everyone', 'everything', 'neither', 'nobody', 'none', 'nothing', 'one', 'other', 'others', 'somebody', 'someone', 'something', 'both', 'few', 'many', 'several', 'all', 'any', 'any', 'more', 'most', 'none', 'some', 'such']

negation_cue = ['nothing', "no", "not", "never", "none", "hardly", "rarely", "scarcely", "seldom", 'barely',
                "nor", "neither", "nothing", "nowhere", "without", "lack", "cant", "dont", "doesnt",
                "doesn't", "don't", "isn't", "wasn't", "aren't", "weren't", "haven't", "hasn't", 
                "shouldn't", "won't", "wouldn't", "can't", "couldn't", "cannot", "unable"]


def load_ids(file):
    id_lut = {}
    with open(file) as f:
        for line in f.readlines():
            i, text = line.strip().split('\t')
            id_lut[i] = text
    return id_lut


def has_pronoun(s):
    for p in pronouns:
        if p.lower() in s.lower().split():
            return True
    return False


def has_negation(s):
    for p in negation_cue:
        if p in s.lower().split():
            return True
    return False


def load_triples(file, ent_lut, rel_lut, kb):
    triples = []
    with open(file) as f:
        for line in f.readlines():
            s, p, o = line.strip().split('\t')
            s_s = ent_lut[s]
            p_s = rel_lut[p]
            o_s = ent_lut[o]
            if has_pronoun(s_s) or has_pronoun(o_s) or has_negation(s_s) or has_pronoun(o_s): 
                continue
            if 'or' in o_s.split():
                continue
            triples.append({'subject': s_s, 'predicate': p_s, 'object': o_s, 'source_kb': kb})
    print(f"* {file} has {len(triples)} valid triples")
    return triples


def assemble(negater_dir, output_file):
    rel_lut = load_ids(f'{negater_dir}/relation_ids.txt')
    ent_lut = load_ids(f'{negater_dir}/entity_ids.txt')
    neg_triples = load_triples(f'{negater_dir}/test_negatives.txt', ent_lut, rel_lut, 'conceptnet-neg-test')
    neg_triples += load_triples(f'{negater_dir}/valid_negatives.txt', ent_lut, rel_lut, 'conceptnet-neg-valid')
    pos_triples = load_triples(f'{negater_dir}/test.txt', ent_lut, rel_lut, 'conceptnet-pos-test')
    pos_triples += load_triples(f'{negater_dir}/valid.txt', ent_lut, rel_lut, 'conceptnet-pos-valid')
    
    min_size = min(len(pos_triples), len(neg_triples))
    triples = pos_triples[:min_size] + neg_triples[:min_size]
    
    print(f"* total {len(triples)} triples")
    with open(output_file, 'w') as f:
        for t in triples:
            f.write(json.dumps(t) + '\n')


if __name__ == '__main__':
    assemble(f'{os.environ["PJ_HOME"]}/data/NegatER/data/conceptnet/true-neg', f'{os.environ["PJ_HOME"]}/data/probe_datasets/true-neg.jsonl')