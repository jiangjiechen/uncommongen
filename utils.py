import os
import re
import ujson as json
import cjjpy as cjj


REL_TO_BOOLQ_TEMPLATE = {
    "IsA": "is [w1] a type of [w2]?",
    'CapableOf': "can [w1] [w2]?",
    'UsedFor': "is [w1] used for [w2]?",
    "MadeOf": "is [w1] made of [w2]?",
    'HasProperty': "does [w1] has the property of [w2]?",
    'HasSubevent': "does [w1] have a subevent of [w2]?",
    "AtLocation": "is [w1] likely to be found in [w2]?",
    "PartOf": "is [w1] part of [w2]?",
    "HasA": "does [w1] have [w2]?",
    # "ReceivesAction": "can [w1] be [w2]?",
    "Causes": "does [w1] cause [w2]?",
    # "HasPrerequisite": "in order for [w1] to happen, does [w2] need to happen?",
    # "NotCapableOf": "is [w1] capable of [w2]?",
    "RelatedTo": "is [w1] like [w2]?",
    "Desires": "does [w1] want [w2]?",
    "MotivatedByGoal": "is [w1] movitated by the goal of [w2]?",
    # "NotHasProperty":  "does [w1] have the property of [w2]?",
    "CreatedBy": "is [w1] created by [w2]?",
    "CausesDesire": "does [w1] make people want [w2]?",
    # "NotIsA": "is [w1] a type of [w2]?",
    # "HasFirstSubevent": "is [w2] the first subevent of [w1]?",
    # "DefinedAs": "is [w1] defined as [w2]?"
}

USUALLY_REL_TO_BOOLQ_TEMPLATE = {
    "IsA": "is [w1] a type of [w2]?",
    'CapableOf': "can [w1] generally [w2]?",
    'UsedFor': "is [w1] generally used for [w2]?",
    "MadeOf": "is [w1] generally made of [w2]?",
    'HasProperty': "does [w1] generally have the property of [w2]?",
    'HasSubevent': "does [w1] generally have a subevent of [w2]?",
    "AtLocation": "is [w1] likely to be found in [w2]?",
    "PartOf": "is [w1] generally part of [w2]?",
    "HasA": "does [w1] generally have [w2]?",
    # "ReceivesAction": "can [w1] generally be [w2]?",
    "Causes": "does [w1] generally cause [w2]?",
    # "HasPrerequisite": "in order for [w1] to happen, does [w2] generally need to happen?",
    # "NotCapableOf": "is [w1] generally capable of [w2]?",
    "RelatedTo": "is [w1] like [w2]?",
    "Desires": "does [w1] generally want [w2]?",
    "MotivatedByGoal": "is [w1] generally movitated by the goal of [w2]?",
    # "NotHasProperty":  "does [w1] generally have the property of [w2]?",
    "CreatedBy": "is [w1] generally created by [w2]?",
    "CausesDesire": "does [w1] generally make people want [w2]?",
    # "NotIsA": "is [w1] a type of [w2]?",
    # "HasFirstSubevent": "is [w2] generally the first subevent of [w1]?",
    # "DefinedAs": "is [w1] generally defined as [w2]?"
}

REL_TO_NEG_TEMPLATE = {
    "IsA": "[w1] is not a type of [w2]",
    'CapableOf': "[w1] can not [w2]",
    'UsedFor': "[w1] is not used for [w2]",
    "MadeOf": "[w1] is not made of [w2]",
    'HasProperty': "[w1] is not [w2]",
    'HasSubevent': "Something you do when you [w1] is [w2]",
    "AtLocation": "You are not likely to find [w1] in [w2]",
    "PartOf": "[w1] is not part of [w2]",
    "HasA": "[w1] does not have [w2]",
    "ReceivesAction": "[w1] can not be [w2]",
    "Causes": "[w1] does not cause [w2]",
    "HasPrerequisite": "In order for [w1] to happen, [w2] needs not to happen",
    "NotCapableOf": "[w1] is capable of [w2]",
    "RelatedTo": "[w1] is not like [w2]",
    "Desires": "[w1] does not want [w2]",
    "MotivatedByGoal": "You would [w1] not because you want to [w2]",
    "NotHasProperty":  "[w1] has the property of [w2]",
    "CreatedBy": "[w1] is not created by [w2]",
    "CausesDesire": "[w1] does not make people want [w2]",
    "NotIsA": "[w1] is a type of [w2]",
    "HasFirstSubevent": "the first thing you do when you [w1] is not [w2]",
    "DefinedAs": "[w1] is not defined as [w2]"
}

REL_TO_TEMPLATE = {
    "RelatedTo": "[w1] is like [w2]",
    "ExternalUrl": "[w1] is described at the following URL [w2]",
    "FormOf": "[w1] is a form of the word [w2]",
    "IsA": "[w1] is a type of [w2]",
    "NotIsA": "[w1] is not [w2]",
    "PartOf": "[w1] is part of [w2]",
    "UsedFor": "[w1] is used for [w2]",
    "CapableOf": "[w1] can [w2]",
    "AtLocation": "You are likely to find [w1] in [w2]",
    "Causes": "Sometimes [w1] causes [w2]",
    "HasA": "[w1] has [w2]",
    "HasSubevent": "Something you do when you [w1] is [w2]",
    "HasFirstSubevent": "the first thing you do when you [w1] is [w2]",
    "HasLastSubevent": "the last thing you do when you [w1] is [w2]",
    "HasPrerequisite": "In order for [w1] to happen, [w2] needs to happen",
    "HasProperty": "[w1] is [w2]",
    "HasContext": "[w1] is a word used in the context of [w2]",
    "MotivatedByGoal": "You would [w1] because you want to [w2]",
    "ObstructedBy": "[w1] can be prevented by [w2]",
    "Desires": "[w1] wants [w2]",
    "CreatedBy": "[w1] is created by [w2]",
    "Synonym": "[w1] and [w2] have similar meanings",
    "Antonym": "[w1] is the opposite of [w2]",
    "DistinctFrom": "it cannot be both [w1] and [w2]",
    "DerivedFrom": "the word [w1] is derived from the word [w2]",
    "DefinedAs": "[w1] is defined as [w2]",
    "Entails": "if [w1] is happening, [w2] is also happening",
    "MannerOf": "[w1] is a specific way of doing [w2]",
    "LocatedNear": "[w1] is located near [w2]",
    "dbpedia": "[w1] is conceptually related to [w2]",
    "SimilarTo": "[w1] is similar to [w2]",
    "EtymologicallyRelatedTo": "the word [w1] and the word [w2] have the same origin",
    "EtymologicallyDerivedFrom": "the word [w1] comes from the word [w2]",
    "CausesDesire": "[w1] makes people want [w2]",
    "MadeOf": "[w1] is made of [w2]",
    "ReceivesAction": "[w1] can be [w2]",
    "InstanceOf": "[w1] is an example of [w2]",
    "NotDesires": "[w1] does not want [w2]",
    "NotUsedFor": "[w1] is not used for [w2]",
    "NotCapableOf": "[w1] is not capable of [w2]",
    "NotHasProperty": "[w1] does not have the property of [w2]",
    "NotMadeOf": "[w1] is not made of [w2]"
}

def avg(x):
    return sum(x) / len(x)


def load_conceptnet_weight(cw_filename=os.path.join(os.environ.get('PJ_HOME', '..'),
                                                    'data/conceptnet/conceptnet_weight.txt'),
                           top_percentage=1.):
    cw_dict = {}
    with open(cw_filename) as f:
        for x in f.readlines():
            c, w = x.strip().split('\t')
            cw_dict[c] = w
    cw_tuple = cjj.SortDict(cw_dict)
    weight_threshold = cw_tuple[int(top_percentage * len(cw_dict))]
    return cw_dict, weight_threshold[-1]


def load_jsonl(jsl_or_path):
    if isinstance(jsl_or_path, str):
        with open(jsl_or_path) as f:
            data = [json.loads(line) for line in f]
    else:
        data = jsl_or_path
    return data


def save_jsonl(jsl, output_file):
    with open(output_file, 'w') as f:
        for js in jsl:
            f.write(json.dumps(js, ensure_ascii=False) + '\n')
    return output_file


def calc_biclf_metrics(y_pred, y_true):
    from sklearn import metrics
    acc = metrics.accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {'accuracy': acc, 'tn': tn/(tn+fp), 'fp': fp/(tn+fp), 'fn': fn/(fn+tp), 'tp': tp/(fn+tp)}


def rel2text(rel):
    if rel == 'ReceivesAction':
        rel_text = 'can be'
    else:
        p = re.compile(r'([a-z]|\d)([A-Z])')
        rel_text = re.sub(p, r'\1 \2', rel).lower()
    return rel_text


def chunks_list_first(lst, n=1):
    """Yield successive n-sized chunks from lst."""
    lst = list(lst)
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def answer2bool(text, prefix='Answer'):
    if prefix is not None:
        # find if Yes, yes or No, no exsits in the text following the prefix with re
        x = re.findall(f'{prefix}:\s*(Yes|yes|No|no)', text)
        x = x[0] if len(x) > 0 else text
    else:
        x = text
    x = x.strip().lower().replace("<pad>", "").replace("###</s>", "").strip()
    if x.startswith('yes'):
        return 1
    elif x.startswith('no'):
        return 0
    else:
        return -1
