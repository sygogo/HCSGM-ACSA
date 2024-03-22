import torch
import torch.nn as nn

from src.model.layers.Attentions import CoverageAttention, Attention
from src.model.layers.Decoder import Decoder
from src.utils.misc import polarity2id


class CoverageDecoder(Decoder):
    def __init__(self, args):
        super().__init__(args)
        self.coverage_attention = CoverageAttention(self.args.config.hidden_size, self.args.config.hidden_size)
        self.init_model(self.args.config.hidden_size , self.args.config.hidden_size , self.args.config.hidden_size * 2)

    def get_coverage_context(self, st, encoder_hidden_states, attention_mask, attention_list):
        a = self.coverage_attention(st, encoder_hidden_states, attention_mask, attention_list)
        a = a.unsqueeze(1)
        context = torch.bmm(a, encoder_hidden_states)
        return context, a

    def forward(self, encoder_hidden_states, attention_mask, encoder_pooler_states, target_aspects):
        yt, st, st_layers, aspect_logits, polarity_logits = self.init_decoder(encoder_hidden_states, encoder_pooler_states)
        attention_list = [torch.zeros_like(attention_mask).to(attention_mask.device).unsqueeze(1)]
        trg_len = self.get_trg_len(target_aspects)
        for step in range(trg_len):
            coverage_context, a = self.get_coverage_context(st, encoder_hidden_states, attention_mask, attention_list)
            rnn_input = torch.cat((yt, coverage_context), dim=2)
            st, st_layers = self.decoder(rnn_input, st_layers)
            output = st
            st = st.permute(1, 0, -1)
            yt = self.decoding(output, output, encoder_hidden_states, attention_mask, target_aspects, aspect_logits, polarity_logits, step)
            attention_list.append(a)
        return torch.cat(aspect_logits, dim=1), torch.cat(polarity_logits, dim=1)
