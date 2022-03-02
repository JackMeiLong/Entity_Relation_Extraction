import torch
import torch.nn as nn
from transformers import BertModel, RobertaModel
from transformers import AutoConfig
from sklearn.metrics import accuracy_score


class CustomModel(nn.Module):
    def __init__(self, bert_dir, fine_tune, model_config, num_labels):
        super(CustomModel, self).__init__()

        self.bert_model = BertModel.from_pretrained(bert_dir)
        self.bert_config = AutoConfig.from_pretrained(bert_dir)

        for param in self.bert_model.parameters():
            if fine_tune:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.linear_layer = nn.Sequential(
            nn.Linear(self.bert_config.hidden_size, num_labels),
            nn.Tanh(),
            nn.Dropout(model_config['dropout_prob'])
        )

        self.linear_layer_2 = nn.Sequential(
            nn.Linear(2 * self.bert_config.hidden_size, num_labels),
            nn.Tanh(),
            nn.Dropout(model_config['dropout_prob'])
        )

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids, obj_type_idx, sub_type_idx, output_y):
        # [batch_size, max_seq_length]
        # [batch_size, ]
        # [batch_size, ]
        bert_output = self.bert_model(input_ids, attention_mask, token_type_ids)
        # [batch_size, max_seq_length, 768]
        bert_output = bert_output[0]

        if obj_type_idx is not None and sub_type_idx is not None:
            obj_type_idx = obj_type_idx.view(-1)
            # [batch_size, max_seq_length]
            obj_type = torch.zeros(obj_type_idx.size(0), input_ids.size(-1)).to(obj_type_idx.device)
            obj_type[range(obj_type_idx.size(0)), obj_type_idx] = 1
            # [batch_size, 1, max_seq_length]
            obj_type = torch.unsqueeze(obj_type, -1).permute(0, 2, 1)
            # [batch_size, 768]
            obj_bert_output = torch.einsum('ijk,ikl->ijl', obj_type, bert_output).squeeze(1)

            sub_type_idx = sub_type_idx.view(-1)
            # [batch_size, max_seq_length]
            sub_type = torch.zeros(sub_type_idx.size(0), input_ids.size(-1)).to(sub_type_idx.device)
            sub_type[range(sub_type_idx.size(0)), sub_type_idx] = 1
            # [batch_size, 1, max_seq_length]
            sub_type = torch.unsqueeze(sub_type, -1).permute(0, 2, 1)
            # [batch_size, 768]
            sub_bert_output = torch.einsum('ijk,ikl->ijl', sub_type, bert_output).squeeze(1)
            # [batch_size, 2*768]
            final_bert_output = torch.cat([obj_bert_output, sub_bert_output], dim=-1)
            logits_y = self.linear_layer_2(final_bert_output)
        else:
            # [CLS]
            final_bert_output = bert_output[:, 0, :]
            logits_y = self.linear_layer(final_bert_output)

        results = (logits_y,)

        if output_y is not None:
            # log softmax nllloss
            loss = self.loss_func(logits_y, output_y)
            metric = self.get_metric(output_y, logits_y.argmax(-1))

            results += (loss, metric)

        return results

    def get_metric(self, y_true, y_pred):
        return torch.tensor(accuracy_score(y_true.cpu().numpy(), y_pred.cpu().numpy()))
