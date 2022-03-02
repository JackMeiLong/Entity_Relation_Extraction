import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from model import CustomModel
import numpy as np
import yaml
import os
from transformers import BertTokenizer, RobertaTokenizer

os.environ['NVIDIA_VISIBLE_DEVICES'] = '0, 1'


def create_dataset(data_path, batch_size, data_type):
    dataset = np.load(data_path, allow_pickle=True)
    input_ids = np.asarray(dataset[:, 0].tolist())
    attention_mask = np.asarray(dataset[:, 1].tolist())
    token_type_ids = np.asarray(dataset[:, 2].tolist())
    output_y = np.asarray(dataset[:, 3].tolist()).astype('int')

    if data_type != 'concat':
        obj_type_idx = np.asarray(dataset[:, 4].tolist())
        sub_type_idx = np.asarray(dataset[:, 5].tolist())
        custom_dataset = TensorDataset(torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(token_type_ids), torch.tensor(obj_type_idx), 
                                            torch.tensor(sub_type_idx), torch.tensor(output_y))
    else:
        custom_dataset = TensorDataset(torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(token_type_ids), torch.tensor(output_y))

    batch_dataset = DataLoader(custom_dataset, shuffle=True, batch_size=batch_size, drop_last=True)

    return batch_dataset


def train_model(model, train_config, data_type, train_dataset, dev_dataset):
    if torch.cuda.is_available():
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=[0])
    else:
        model = model.to('cpu')

    params = [k for k, v in model.named_parameters() if v.requires_grad == True]
    print(params)

    optim = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'])

    for epoch in range(train_config['max_epochs']):
        model.train()
        loss_sum = 0
        metric_sum = 0

        for step, batch in enumerate(train_dataset, 1):
            input_ids = batch[0].cuda() if torch.cuda.is_available() else batch[0]
            attention_mask = batch[1].cuda() if torch.cuda.is_available() else batch[1]
            token_type_ids = batch[2].cuda() if torch.cuda.is_available() else batch[2]
            obj_type_idx = None
            sub_type_idx = None
            output_y = batch[-1].cuda() if torch.cuda.is_available() else batch[-1]

            if data_type != 'concat':
                obj_type_idx = batch[3].cuda() if torch.cuda.is_available() else batch[3]
                sub_type_idx = batch[4].cuda() if torch.cuda.is_available() else batch[4]

            _, loss, metric = model(input_ids, attention_mask, token_type_ids, obj_type_idx, sub_type_idx, output_y)

            optim.zero_grad()
            loss.mean().backward()
            optim.step()

            loss_sum += loss.mean().item()
            metric_sum += metric.mean().item()

            if step % 1000 == 0:
                print('step : {}, loss : {}, metric : {}'.format(step, loss_sum / step, metric_sum / step))

        model.eval()
        dev_loss_sum = 0
        dev_metric_sum = 0

        for dev_step, batch in enumerate(dev_dataset, 1):
            input_ids = batch[0].cuda() if torch.cuda.is_available() else batch[0]
            attention_mask = batch[1].cuda() if torch.cuda.is_available() else batch[1]
            token_type_ids = batch[2].cuda() if torch.cuda.is_available() else batch[2]
            obj_type_idx = None
            sub_type_idx = None
            output_y = batch[-1].cuda() if torch.cuda.is_available() else batch[-1]

            if data_type != 'concat':
                obj_type_idx = batch[3].cuda() if torch.cuda.is_available() else batch[3]
                sub_type_idx = batch[4].cuda() if torch.cuda.is_available() else batch[4]

            with torch.no_grad():
                _, loss, metric = model(input_ids, attention_mask, token_type_ids, obj_type_idx, sub_type_idx, output_y)

            dev_loss_sum += loss.mean().item()
            dev_metric_sum += metric.mean().item()

        print('epoch : {}, loss : {}, metric : {}, dev_loss : {}, dev_metric : {}'.format(epoch, loss_sum / step,
                                                                                          metric_sum / step,
                                                                                          dev_loss_sum / dev_step,
                                                                                          dev_metric_sum / dev_step))

    torch.save(model.state_dict(), './model_save/net_params.pkl')


if __name__ == '__main__':
    data_type = 'concat'

    conf_file = open('./configs.yml', 'r', encoding='utf-8')
    conf = conf_file.read()
    conf_file.close()
    configs = yaml.load(conf)

    model_config = configs['model_config']
    train_config = configs['train_config']

    bert_type = 'bert_base_chinese'
    data_type = 'entity_type_mask'

    train_dataset = create_dataset('../data/data_source/train_data.npy', train_config['batch_size'], data_type)
    dev_dataset = create_dataset('../data/data_source/dev_data.npy', train_config['batch_size'], data_type)

    model = CustomModel(model_config['bert_path'][bert_type], train_config['fine_tune'], model_config, 49)

    if data_type != 'concat':
        conf_file = open('../data/data_config.yml', 'r', encoding='utf-8')
        conf = conf_file.read()
        conf_file.close()
        
        data_config = yaml.load(conf)
        
        tokenizer = BertTokenizer.from_pretrained(data_config['bert_path'][bert_type])
        
        if data_type == 'entity_mask':
            tokenizer.add_tokens(['obj'])
            tokenizer.add_tokens(['sub'])
        
        if data_type == 'entity_type_mask':
            tokenizer.add_tokens(['obj_{}'.format(x).lower() for x in data_config['data_type'][data_type]['obj_type']])
            tokenizer.add_tokens(['sub_{}'.format(x).lower() for x in data_config['data_type'][data_type]['sub_type']])
        
        model.bert_model.resize_token_embeddings(len(tokenizer))

    train_model(model, train_config, data_type, train_dataset, dev_dataset)
