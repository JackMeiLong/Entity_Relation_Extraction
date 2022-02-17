import json
import os

def prepare_data(data_path, mode):
    print("---prepare {} data---".format(mode))
    with open(os.path.join(data_path, "{}_data.json".format(mode)), 'r', encoding='utf-8') as load_f:
        info = []
        import random
        for line in load_f.readlines():
            dic = json.loads(line)
            for j in dic['spo_list']:
                single_data = {}
                single_data['relation'] = j["predicate"]
                single_data['obj'] = j["object"]
                single_data['obj_type'] = j["object_type"]
                single_data['sub'] = j["subject"]
                single_data['sub_type'] = j["subject_type"]
                single_data['text'] = dic['text']
                single_data['token_list'] = list(map(lambda x: x['word'], dic['postag']))
                single_data['pos_list'] = list(map(lambda x: x['pos'], dic['postag']))
                info.append(single_data)
        sub_train = info

    with open(os.path.join(data_path, "{}.json".format(mode)), "w", encoding='utf-8') as dump_f:
        for i in sub_train:
            a = json.dumps(i, ensure_ascii=False)
            dump_f.write(a)
            dump_f.write("\n")

if __name__ == '__main__':

    data_path = './data_source'
    prepare_data(data_path, 'train')
    prepare_data(data_path, 'dev')