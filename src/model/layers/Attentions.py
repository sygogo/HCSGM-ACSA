import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.WH = nn.Linear(enc_hid_dim, dec_hid_dim)
        self.WS = nn.Linear(enc_hid_dim, dec_hid_dim)
        self.V = nn.Linear(dec_hid_dim, 1, bias=False)

        nn.init.xavier_uniform_(self.WH.weight)
        nn.init.xavier_uniform_(self.WS.weight)
        nn.init.xavier_uniform_(self.V.weight)

    def forward(self, hidden, encoder_outputs, attention_mask):
        src_len = encoder_outputs.shape[1]
        hidden = hidden.repeat(src_len, 1, 1).permute(1, 0, -1)
        energy = torch.tanh(self.WS(hidden) + self.WH(encoder_outputs))
        attention = self.V(energy).squeeze(2)
        attention = torch.where(attention_mask == 1, attention, torch.ones_like(attention) * -1e9)
        return F.softmax(attention, dim=1)


class CoverageAttention(Attention):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__(enc_hid_dim, dec_hid_dim)
        self.wc = nn.Parameter(torch.rand(1, ))

    def forward(self, hidden, encoder_outputs, attention_mask, attention_list):
        attentions = torch.cat(attention_list, dim=1)
        src_len = encoder_outputs.shape[1]
        hidden = hidden.repeat(src_len, 1, 1).permute(1, 0, -1)
        energy = torch.tanh(self.WS(hidden) + self.WH(encoder_outputs) + self.wc * torch.sum(attentions, dim=1).unsqueeze(-1))
        attention = self.V(energy).squeeze(2)
        attention = torch.where(attention_mask == 1, attention, torch.ones_like(attention) * -1e9)
        return F.softmax(attention, dim=1)
