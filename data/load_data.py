import json
import os
import numpy as np
import yaml
from transformers import BertTokenizer, RobertaTokenizer

def map_id_rel():
    id2rel = {0: 'UNK', 1: '主演', 2: '歌手', 3: '简称', 4: '总部地点', 5: '导演', 6: '出生地', 7: '目',
            8: '出生日期', 9: '占地面积', 10: '上映时间', 11: '出版社', 12: '作者', 13: '号', 14: '父亲',
            15: '毕业院校', 16: '成立日期', 17: '改编自', 18: '主持人', 19: '所属专辑', 20: '连载网站',
            21: '作词', 22: '作曲', 23: '创始人', 24: '丈夫', 25: '妻子', 26: '朝代', 27: '民族', 28: '国籍',
            29: '身高', 30: '出品公司', 31: '母亲', 32: '编剧', 33: '首都', 34: '面积', 35: '祖籍', 36: '嘉宾',
            37: '字', 38: '海拔', 39: '注册资本', 40: '制片人', 41: '董事长', 42: '所在城市', 43: '气候',
            44: '人口数量', 45: '邮政编码', 46: '主角', 47: '官方语言', 48: '修业年限'}

    rel2id = dict({(v, k) for k, v in id2rel.items()})
    return id2rel, rel2id

def load_data(data_path, mode, config, bert_type):

    rel2id, id2rel = map_id_rel()
    tokenizer = BertTokenizer.from_pretrained(config['bert_path'][bert_type])
    dataset = []

    with open(os.path.join(data_path, '{}.json'.format(mode)), 'r', encoding='utf-8') as load_f:
        temp = load_f.readlines()
        temp = temp[:200]
        for line in temp:
            dic = json.loads(line)
            data = []
            context = dic['obj']+dic['sub']+dic['text']
            bert_input = tokenizer.encode_plus(context, add_special_tokens=True, padding='max_length', max_length=config['max_seq_length'],
                                               return_attention_mask=True, truncation=True)
            data.append(bert_input['input_ids'])
            data.append(bert_input['attention_mask'])
            data.append(bert_input['token_type_ids'])
            data.append(rel2id.get(dic['relation'], 0))
            dataset.append(data)

    if ~os.path.exists(os.path.join(data_path, '{}_data.npy').format(mode)):
        np.save(os.path.join(data_path, '{}_data.npy').format(mode), dataset)
    return dataset

if __name__ == '__main__':
    data_path = './data_source'

    conf_file = open('./data_config.yml', 'r', encoding='utf-8')
    conf = conf_file.read()
    conf_file.close()

    config = yaml.load(conf)
    bert_type = 'bert_base_chinese'
    load_data(data_path, 'train', config, bert_type)
    load_data(data_path, 'dev', config, bert_type)

    '''
    data type:
    concat
    obj_type/sub_type
    obj_pos/sub_pos
    '''