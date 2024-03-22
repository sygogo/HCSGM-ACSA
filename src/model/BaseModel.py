from torch import nn


class BaseModel(nn.Module):

    def get_batch(self, batch):
        input_ids, attention_mask, padded_target_aspects, padded_target_polarities, targets = batch
        return input_ids.cuda(), attention_mask.cuda(), padded_target_aspects.cuda(), padded_target_polarities.cuda(), targets
