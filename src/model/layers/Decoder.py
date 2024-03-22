from random import random

import torch
from torch import nn

from src.model.layers.Attentions import Attention
from src.utils.misc import polarity2id


class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.aspect_embedding = nn.Embedding(embedding_dim=self.args.config.hidden_size, num_embeddings=len(self.args.category2id))
        nn.init.xavier_uniform_(self.aspect_embedding.weight)
        self.attention = Attention(self.args.config.hidden_size, self.args.config.hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.V = nn.Linear(self.args.config.hidden_size, self.args.config.hidden_size)
        nn.init.xavier_uniform_(self.V.weight)
        self.init_model(self.args.config.hidden_size, self.args.config.hidden_size, self.args.config.hidden_size * 2)

    def init_model(self, aspect_input_size, polarity_input_size, gru_input_size):
        """
        """
        self.decoder = nn.GRU(batch_first=True, hidden_size=self.args.config.hidden_size, input_size=gru_input_size, num_layers=self.args.decoder_num_layer)
        self.decoder2aspect = nn.Linear(aspect_input_size, len(self.args.category2id))
        self.decoder2polarity = nn.Linear(polarity_input_size, len(polarity2id))

        nn.init.xavier_uniform_(self.decoder2aspect.weight)
        nn.init.xavier_uniform_(self.decoder2polarity.weight)

    def decoding(self, output_aspect, output_polarity, encoder_hidden_states, attention_mask, target_aspects, aspect_logits, polarity_logits, step):
        aspect_logit = self.decoder2aspect(self.dropout(output_aspect))
        polarity_logit = self.decoder2polarity(self.dropout(output_polarity))
        # replace logit = -1e9 ,if predicted
        if len(aspect_logits) > 0:
            aspect_logit = self.remove_repeat_labels(aspect_logits, aspect_logit)
        aspect_logits.append(aspect_logit)
        polarity_logits.append(polarity_logit)
        input = self.get_next_input(target_aspects, step, aspect_logit)
        return input

    def get_next_input(self, target_aspects, step, aspect_logit):
        teacher_force = (random() < 0.5) and (self.args.mode == 'train')
        input = target_aspects[:, step].unsqueeze(1) if teacher_force else aspect_logit.argmax(-1)
        input = self.aspect_embedding(input.to(target_aspects.device))
        return input

    def get_trg_len(self, target_aspects):
        trg_len = target_aspects.size(1) if self.args.mode == 'train' else self.args.aspect_number
        return trg_len

    def remove_repeat_labels(self, aspect_logits, aspect_logit):
        max_logit_index = torch.max(torch.cat(aspect_logits, dim=1), dim=-1)[1]
        # 将出现过的aspect替换成最小值
        min_logit = torch.min(torch.cat(aspect_logits, dim=1), dim=-1)[0]
        # 如果出现的是 eos 则不忽略，还是可以继续出现eos
        max_logit_index = torch.where(max_logit_index == self.args.category2id['<EOS>'], torch.zeros_like(max_logit_index), max_logit_index)
        aspect_logit = aspect_logit.squeeze(1).scatter(-1, max_logit_index, min_logit)
        aspect_logit = aspect_logit.unsqueeze(1)
        return aspect_logit

    def get_attention_context(self, st, encoder_hidden_states, attention_mask):
        a = self.attention(st, encoder_hidden_states, attention_mask)
        a = a.unsqueeze(1)
        context = torch.bmm(a, encoder_hidden_states)
        return context, a

    def init_decoder(self, encoder_hidden_states, encoder_pooler_states):
        bz = len(encoder_hidden_states)
        st = self.V(encoder_pooler_states.unsqueeze(0))
        bos_input = (torch.ones(size=(bz, 1), dtype=torch.int64) * int(self.args.category2id['<BOS>'])).to(encoder_pooler_states.device)
        yt = self.aspect_embedding(bos_input)
        aspect_logits, polarity_logits = [], []
        st_layers = st.repeat(self.args.decoder_num_layer, 1, 1)
        return yt, st, st_layers, aspect_logits, polarity_logits

    def forward(self, encoder_hidden_states, attention_mask, encoder_pooler_states, target_aspects):
        yt, st, st_layers, aspect_logits, polarity_logits = self.init_decoder(encoder_hidden_states, encoder_pooler_states)
        trg_len = self.get_trg_len(target_aspects)
        for step in range(trg_len):
            attention_context, a = self.get_attention_context(st, encoder_hidden_states, attention_mask)
            rnn_input = torch.cat((yt, attention_context), dim=2)
            st, st_layers = self.decoder(rnn_input, st_layers)
            output = st
            st = st.permute(1, 0, -1)
            yt = self.decoding(output, output, encoder_hidden_states, attention_mask, target_aspects, aspect_logits, polarity_logits, step)
        return torch.cat(aspect_logits, dim=1), torch.cat(polarity_logits, dim=1)
