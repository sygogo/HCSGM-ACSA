from random import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.layers.Attentions import Attention
from src.model.layers.CoverageDecoder import CoverageDecoder
from src.utils.misc import polarity2id


class HerCoverageDecoder(CoverageDecoder):
    def __init__(self, args):
        super().__init__(args)
        self.polarity_attention = Attention(self.args.config.hidden_size, self.args.config.hidden_size)
        self.init_model(self.args.config.hidden_size, self.args.config.hidden_size, self.args.config.hidden_size * 2)
        self.theta = nn.Parameter(torch.rand(size=(1,)))

    def init_model(self, aspect_input_size, polarity_input_size, gru_input_size):
        super().init_model(aspect_input_size, polarity_input_size, gru_input_size)
        self.sentiment_tokens_fc = nn.Linear(self.args.config.hidden_size, self.args.config.hidden_size)
        nn.init.kaiming_uniform_(self.sentiment_tokens_fc.weight)
        self.sentiment_tokens_fc2 = nn.Linear(self.args.config.hidden_size, len(polarity2id))
        nn.init.kaiming_uniform_(self.sentiment_tokens_fc2.weight)

    def get_polarity_context(self, aspect_emb, encoder_hidden_states, attention_mask):
        a = self.polarity_attention(aspect_emb, encoder_hidden_states, attention_mask)
        a = a.unsqueeze(1)
        context = torch.bmm(a, encoder_hidden_states)
        return context, a

    def decoding(self, output_aspect, output_polarity, encoder_hidden_states, attention_mask, target_aspects, aspect_logits, polarity_logits, step):
        aspect_logit = self.decoder2aspect(self.dropout(output_aspect))
        polarity_logit = torch.tanh(self.decoder2polarity(self.dropout(output_polarity)))
        # replace logit = -1e9 ,if predicted
        if len(aspect_logits) > 0:
            aspect_logit = self.remove_repeat_labels(aspect_logits, aspect_logit)
        if step == 0:
            self.encoder_hidden_states_polarity = self.sentiment_tokens_fc2(torch.relu(self.sentiment_tokens_fc(self.dropout(encoder_hidden_states))))
        aspect_logits.append(aspect_logit)
        input = self.get_next_input(target_aspects, step, aspect_logit)
        polarity_context, aspect_score = self.get_polarity_context(input.squeeze(1), encoder_hidden_states, attention_mask)
        polarity_logit_agg = torch.sum(aspect_score.permute(0, -1, 1) * self.encoder_hidden_states_polarity, dim=1)
        polarity_logit = (1 - torch.clamp(self.theta,0,1)) * polarity_logit + torch.clamp(self.theta,0,1) * polarity_logit_agg.unsqueeze(1)
        polarity_logits.append(polarity_logit)
        return input
