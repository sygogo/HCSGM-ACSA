"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher(nn.Module):

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        generate the optimal assignment pi*
        :param outputs:
        :param targets:
        :return:
        """
        bsz, num_generated_aspects = outputs["pred_aspects_logit"].shape[:2]
        # We flatten to compute the cost matrices in a batch
        pred_aspect = outputs["pred_aspects_logit"].flatten(0, 1).softmax(-1)  # [bsz * num_generated_aspects, num_classes]
        gold_aspect_split = [torch.tensor([j.item() for j in i if j > 0]) for i in targets[0]]
        gold_aspect = torch.cat(gold_aspect_split)
        # after masking the pad token
        pred_polarity = outputs["pred_polarities_logit"].flatten(0, 1).softmax(-1)  # [bsz * num_generated_triples, seq_len]
        gold_polarity_split = [torch.tensor([j.item() for j in i if j > 0]) for i in targets[1]]
        gold_polarity = torch.cat(gold_polarity_split)
        cost = - (pred_aspect[:, gold_aspect] + pred_polarity[:, gold_polarity])
        cost = cost.view(bsz, num_generated_aspects, -1).cpu()
        num_gold_aspect = [len(v) for v in gold_aspect_split]
        indices = [linear_sum_assignment(c[i].T) for i, c in enumerate(cost.split(num_gold_aspect, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices], gold_aspect_split, gold_polarity_split

    def get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
