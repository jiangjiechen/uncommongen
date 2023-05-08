import argparse
import ujson as json


def remove_nested_keys(js, keys):
    nested_keys = keys.split('/')
    for k in nested_keys[:-1]:
        js = js[k]
    rm_key = nested_keys[-1]
    if js.get(rm_key):
        js.pop(rm_key)
    else:
        print(f'no key: {rm_key}')


def test():
    s = {"subject":"speed boat","predicate":"HasProperty","object":"dangerous","source_kb":"conceptnet-pos-test","boolq":{"text-davinci-002_ex-8":"is a speed boat dangerous?"},"cg_pred":{"text-davinci-002_ex-4p4n":[" speed boats can be dangerous."],"text-davinci-002_ex-2p2n":[" speed boats are dangerous."],"text-davinci-002_ex-2p6n":[" speed boats are not dangerous."],"text-davinci-002_ex-6p2n":["\n\nspeed boats are dangerous."],"text-davinci-002_ex-1p7n":[" speed boats can be dangerous."],"text-davinci-002_ex-0p8n":[" speed boats are not dangerous."],"text-davinci-002_ex-8p8n":["speed boats can be dangerous.\n###\nKeywords: dollar, used to, buy things\n\nDollars are used to buy things."]},"qa_pred":{"text-davinci-002_ex-4p4n":"yes","text-davinci-002_ex-2p2n":"yes","text-davinci-002_ex-2p6n":"yes","text-davinci-002_ex-6p2n":"yes"}}
    for k in ['cg_pred/text-davinci-002_ex-8p8n', 'cg_pred/text-davinci-002_ex-6p2n', 'cg_pred/text-davinci-002_ex-0p8den']:
        remove_nested_keys(s, k)
    print(s)
    # print(ss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True)
    parser.add_argument('-o', type=str)
    parser.add_argument('-k', type=str, help='nested keys, e.g. aaa/bbb/ccc')
    args = parser.parse_args()

    new_jsl = []
    with open(args.i) as f:
        for line in f.readlines():
            try:
                js = json.loads(line)
            except:
                print(line)
                raise ValueError
            remove_nested_keys(js, args.k)
            new_jsl.append(js)
    
    output_file = args.i if args.o is None else args.o
    with open(output_file, 'w') as fo:
        for js in new_jsl:
            fo.write(json.dumps(js) + '\n')