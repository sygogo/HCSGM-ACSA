import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, AutoModel, BartConfig
from transformers.models.bart.modeling_bart import BartEncoder, PretrainedBartModel


class BertEncoder(nn.Module):
    def __init__(self, args):
        super(BertEncoder, self).__init__()
        self.args = args
        self.bert = AutoModel.from_pretrained(args.bert_directory, output_hidden_states=True)
        if args.fix_bert_embeddings:
            self.bert.embeddings.word_embeddings.weight.requires_grad = False
            self.bert.embeddings.position_embeddings.weight.requires_grad = False
            self.bert.embeddings.token_type_embeddings.weight.requires_grad = False
        self.args.config = self.bert.config

    def forward(self, input_ids, attention_mask):
        obj = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = obj.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        return last_hidden_state, cls_output
