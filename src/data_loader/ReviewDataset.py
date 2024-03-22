from torch.utils.data import Dataset
import torch
import numpy as np
from transformers import BertTokenizer


class ReviewDataset(Dataset):
    def __init__(self, data, bertTokenizer):
        super(ReviewDataset, self).__init__()
        self.data = data
        self.bertTokenizer = bertTokenizer

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __pad__(self, input_list, pad_value=0, max_seq_len=None):
        input_list_lens = [len(l) for l in input_list]
        if max_seq_len is None:
            max_seq_len = 300 if max(input_list_lens) >= 300 else max(input_list_lens)
        padded_batch = pad_value * np.ones((len(input_list), max_seq_len))
        for j in range(len(input_list)):
            current_len = 300 if input_list_lens[j] >= 300 else input_list_lens[j]
            padded_batch[j][:current_len] = input_list[j][:current_len]
        padded_batch = torch.LongTensor(padded_batch)
        input_mask = torch.ne(padded_batch, pad_value).type(torch.FloatTensor)
        return padded_batch, input_list_lens, input_mask

    def collate_fn(self, batches):
        text, text_token, text_token_id, targets = [i for i in zip(*batches)]
        target_aspect = [t["aspects"] for t in targets]
        target_polarity = [t["polarities"] for t in targets]
        padded_target_aspects, _, _ = self.__pad__(target_aspect)
        padded_target_polarities, _, _ = self.__pad__(target_polarity)
        padded_text_token_id, _, mask_text_token_id = self.__pad__(text_token_id, self.bertTokenizer.pad_token_id)
        return padded_text_token_id, mask_text_token_id, padded_target_aspects, padded_target_polarities, targets
