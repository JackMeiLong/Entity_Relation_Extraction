import json
import os
import numpy as np
import yaml
from transformers import BertTokenizer, RobertaTokenizer
from collections import Counter

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
# obj/sub 使用同一套look-up table
def map_id_type():
    id2type = {
        0:'UNK', 1:'作品', 2:'学科专业', 3:'音乐专辑', 4:'影视作品', 5:'地点', 6:'歌曲', 7:'网站', 8:'出版社', 9:'目',
        10:'图书作品', 11:'气候', 12:'历史人物', 13:'date', 14:'城市', 15:'number', 16:'text', 17:'景点', 18:'国家', 19:'网络小说',
        20:'生物', 21:'企业', 22:'语言', 23:'学校', 24:'人物', 25:'书籍', 26:'电视综艺', 27:'机构', 28:'行政区'
    }

    type2id = dict({(v, k) for k, v in id2type.items()})
    return id2type, type2id

def load_data(data_path, mode, config, bert_type, data_type):
    id2rel, rel2id = map_id_rel()
    id2type, type2id = map_id_type()

    tokenizer = BertTokenizer.from_pretrained(config['bert_path'][bert_type])
    dataset = []

    if data_type == 'entity_mask':
        tokenizer.add_tokens(['obj'])
        tokenizer.add_tokens(['sub'])

    if data_type == 'entity_type_mask':
        tokenizer.add_tokens(['obj_{}'.format(x).lower() for x in config['data_type'][data_type]['obj_type']])
        tokenizer.add_tokens(['sub_{}'.format(x).lower() for x in config['data_type'][data_type]['sub_type']])

    with open(os.path.join(data_path, '{}.json'.format(mode)), 'r', encoding='utf-8') as load_f:
        temp = load_f.readlines()
        #temp = temp[:100]
        for line in temp:
            #print(line)
            dic = json.loads(line)
            data = []

            if dic['obj'] not in dic['text'] or dic['sub'] not in dic['text']:
                continue

            if data_type == 'concat':
                context = dic['obj'] + dic['sub'] + dic['text']
            if data_type == 'entity_mask':
                context = dic['text']
                # 暂不考虑obj/sub 出现多次的情况
                context = context.replace(dic['obj'], 'obj')
                context = context.replace(dic['sub'], 'sub')
            if data_type == 'entity_type_mask':
                context = dic['text']
                # 暂不考虑obj/sub 出现多次的情况
                context = context.replace(dic['obj'], 'obj_{}'.format(dic['obj_type']).lower())
                context = context.replace(dic['sub'], 'sub_{}'.format(dic['sub_type']).lower())

            bert_input = tokenizer.encode_plus(context, add_special_tokens=True, padding='max_length',
                                               max_length=config['max_seq_length'],
                                               return_attention_mask=True, truncation=True)
            data.append(bert_input['input_ids'])
            data.append(bert_input['attention_mask'])
            data.append(bert_input['token_type_ids'])
            data.append(rel2id.get(dic['relation'], 0))
            #print(rel2id.get(dic['relation'],0))

            if data_type == 'entity_mask':
                # obj index
                if tokenizer.convert_tokens_to_ids('obj') in bert_input['input_ids']:
                    data.append(bert_input['input_ids'].index(tokenizer.convert_tokens_to_ids('obj')))
                else:
                    continue
                # sub index
                if tokenizer.convert_tokens_to_ids('sub') in bert_input['input_ids']:
                    data.append(bert_input['input_ids'].index(tokenizer.convert_tokens_to_ids('sub')))
                else:
                    continue

                # obj/sub entity_type 
                data.append(type2id.get(dic['obj_type'].lower(), 0))
                data.append(type2id.get(dic['sub_type'].lower(), 0))

            if data_type == 'entity_type_mask':
                # obj_type index
                #print(context)
                if tokenizer.convert_tokens_to_ids('obj_{}'.format(dic['obj_type']).lower()) in bert_input['input_ids']:
                    data.append(bert_input['input_ids'].index(tokenizer.convert_tokens_to_ids('obj_{}'.format(dic['obj_type']).lower())))
                else:
                    continue
                # sub_type index
                if tokenizer.convert_tokens_to_ids('sub_{}'.format(dic['sub_type']).lower()) in bert_input['input_ids']:
                    data.append(bert_input['input_ids'].index(tokenizer.convert_tokens_to_ids('sub_{}'.format(dic['sub_type']).lower())))
                else:
                    continue
            

            dataset.append(data)
           
    
    #print(dataset)
    sta = Counter(np.asarray(dataset)[:,-1])
    sta = dict({(id2rel.get(k),v) for k, v in sta.items()})
    f = open('./data_source/statistic_{0}_v2.txt'.format(data_type),'a+',encoding='utf-8')
    f.write('{} data:\n'.format(mode))
    f.write('{} data {}:\n'.format(mode, len(dataset)))
    f.write(str(sta))
    f.write('\n')
    f.close()

    if ~os.path.exists(os.path.join(data_path, '{}_data_v2.npy').format(mode)):
        np.save(os.path.join(data_path, '{}_data_v2.npy').format(mode), dataset)
    return dataset


if __name__ == '__main__':
    data_path = './data_source'

    conf_file = open('./data_config_v2.yml', 'r', encoding='utf-8')
    conf = conf_file.read()
    conf_file.close()

    config = yaml.load(conf)
    bert_type = 'bert_base_chinese'
    data_type = 'entity_mask'
    load_data(data_path, 'train', config, bert_type, data_type)
    load_data(data_path, 'dev', config, bert_type, data_type)
