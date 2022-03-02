import torch
import torch.nn as nn
from transformers import BertModel, RobertaModel
from transformers import AutoConfig
from sklearn.metrics import accuracy_score


class CustomModel(nn.Module):
    def __init__(self, bert_dir, hidden_szie, fine_tune, model_config, type_size, num_labels):
        super(CustomModel, self).__init__()

        self.bert_model = BertModel.from_pretrained(bert_dir)
        self.bert_config = AutoConfig.from_pretrained(bert_dir)

        for param in self.bert_model.parameters():
            if fine_tune:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        self.type_embedding_layer = nn.Embedding(type_size, self.bert_config.hidden_size)
        #self.type_embedding_layer.requires_grad_(requires_grad=True)

        self.linear_layer_0 = nn.Sequential(
            nn.Linear(2*self.bert_config.hidden_size, hidden_szie),
            nn.Tanh(),
            nn.Dropout(model_config['dropout_prob'])
        )

        self.linear_layer_1 = nn.Sequential(
            nn.Linear(2 * self.bert_config.hidden_size, hidden_szie),
            nn.Tanh(),
            nn.Dropout(model_config['dropout_prob'])
        )

        self.final_layer = nn.Sequential(
            nn.Linear(2 * hidden_szie, num_labels),
            nn.Tanh(),
            nn.Dropout(model_config['dropout_prob'])
        )

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids, obj_type_mask_idx, sub_type_mask_idx, obj_type, sub_type, output_y):
        # [batch_size, max_seq_length]
        # [batch_size, ]
        # [batch_size, ]
        # [batch_size, ]
        bert_output = self.bert_model(input_ids, attention_mask, token_type_ids)
        # [batch_size, max_seq_length, 768]
        bert_output = bert_output[0]

        if obj_type_mask_idx is not None and sub_type_mask_idx is not None:
            obj_type_mask_idx = obj_type_mask_idx.view(-1)
            # [batch_size, max_seq_length]
            obj_type_mask = torch.zeros(obj_type_mask_idx.size(0), input_ids.size(-1)).to(obj_type_mask_idx.device)
            obj_type_mask[range(obj_type_mask_idx.size(0)), obj_type_mask_idx] = 1
            # [batch_size, 1, max_seq_length]
            obj_type_mask = torch.unsqueeze(obj_type_mask, -1).permute(0, 2, 1)
            # [batch_size, 768]
            obj_bert_output = torch.einsum('ijk,ikl->ijl', obj_type_mask, bert_output).squeeze(1)

            sub_type_mask_idx = sub_type_mask_idx.view(-1)
            # [batch_size, max_seq_length]
            sub_type_mask = torch.zeros(sub_type_mask_idx.size(0), input_ids.size(-1)).to(sub_type_mask_idx.device)
            sub_type_mask[range(sub_type_mask_idx.size(0)), sub_type_mask_idx] = 1
            # [batch_size, 1, max_seq_length]
            sub_type_mask = torch.unsqueeze(sub_type_mask, -1).permute(0, 2, 1)
            # [batch_size, 768]
            sub_bert_output = torch.einsum('ijk,ikl->ijl', sub_type_mask, bert_output).squeeze(1)
            # [batch_size, 2*768]
            final_bert_output = torch.cat([obj_bert_output, sub_bert_output], dim=-1)
            # [batch_size, hidden_size]
            final_bert_output = self.linear_layer_0(final_bert_output)
            
        # [batch_size, 768]
        obj_type_emb = self.type_embedding_layer(obj_type)
        #print(obj_type_emb.size())
        sub_type_emb = self.type_embedding_layer(sub_type)
        #print(sub_type_emb.size())
        # [batch_size, 2*768]
        type_emb = torch.cat([obj_type_emb, sub_type_emb], dim=-1)
        # [batch_size, hidden_size]
        type_output = self.linear_layer_1(type_emb)
        # [batch_size, 2*hidden_size]
        final_output = torch.cat([final_bert_output, type_output], dim=-1)
        
        logits_y = self.final_layer(final_output)

        results = (logits_y,)

        if output_y is not None:
            # log softmax nllloss
            loss = self.loss_func(logits_y, output_y)
            metric = self.get_metric(output_y, logits_y.argmax(-1))

            results += (loss, metric)

        return results

    def get_metric(self, y_true, y_pred):
        return torch.tensor(accuracy_score(y_true.cpu().numpy(), y_pred.cpu().numpy()))
