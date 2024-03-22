import torch
import torch.nn.functional as F
import torch.nn as nn
from src.model.BaseModel import BaseModel
from src.model.layers.BertEncoder import BertEncoder
from src.model.layers.CoverageDecoder import CoverageDecoder
from src.model.layers.Decoder import Decoder
from src.model.layers.HerCoverageDecoder import HerCoverageDecoder
from src.model.layers.HungarianMatcher import HungarianMatcher
from src.utils.misc import polarity2id


class Seq2Seq(BaseModel):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if self.args.encoder_type in ['bert']:
            self.encoder = BertEncoder(args)
        else:
            raise ValueError('Not this encoder')
        if self.args.decoder_type == 'decoder':
            self.decoder = Decoder(args)
        elif self.args.decoder_type == 'cov_decoder':
            self.decoder = CoverageDecoder(args)
        # elif self.args.decoder_type == 'cov_review_decoder':
        #     self.decoder = CoverageReviewDecoder(args)
        elif self.args.decoder_type == 'her_cov_decoder':
            self.decoder = HerCoverageDecoder(args)
            if self.args.set_loss:
                self.matcher = HungarianMatcher()
        else:
            raise ValueError('Not this decoder')

    def predict(self, batch):
        input_ids, attention_mask, padded_target_aspects, padded_target_polarities, targets = self.get_batch(batch)
        last_hidden_state, pooler_output = self.encoder(input_ids, attention_mask)
        self.decoder.predict(last_hidden_state, attention_mask, pooler_output, padded_target_aspects)

    def forward(self, batch):
        input_ids, attention_mask, padded_target_aspects, padded_target_polarities, targets = self.get_batch(batch)
        last_hidden_state, pooler_output = self.encoder(input_ids, attention_mask)
        aspects_logit, polarities_logit = self.decoder(last_hidden_state, attention_mask, pooler_output,
                                                       padded_target_aspects)
        outputs = {"pred_aspects_logit": aspects_logit, "pred_polarities_logit": polarities_logit}
        if self.args.mode == 'train':
            if self.args.set_loss:
                losses, loss_info = self.compute_set_loss(outputs, (padded_target_aspects, padded_target_polarities))
            else:
                losses, loss_info = self.compute_loss(outputs, (padded_target_aspects, padded_target_polarities))
            return outputs, losses, loss_info
        else:
            return outputs

    def compute_loss(self, outputs, targets):
        # loss1
        padded_target_aspects, padded_target_polarities = targets
        pred_aspect_logit = outputs['pred_aspects_logit']
        loss_aspect = F.cross_entropy(pred_aspect_logit.flatten(0, 1), padded_target_aspects.flatten(0, 1),
                                      ignore_index=self.args.category2id['<PAD>'])
        pred_polarity_logit = outputs['pred_polarities_logit']  # [bsz, num_generated_aspect, num_polarity+1]
        loss_polarity = F.cross_entropy(pred_polarity_logit.flatten(0, 1), padded_target_polarities.flatten(0, 1),
                                        ignore_index=polarity2id['<PAD>'])

        return loss_aspect + loss_polarity, {'loss_polarity': loss_polarity.item(), 'loss_aspect': loss_aspect.item()}

    def compute_set_loss(self, outputs, targets):
        indices, gold_aspect_split, gold_polarity_split = self.matcher(outputs, targets)
        idx = self.matcher.get_src_permutation_idx(indices)

        # loss1
        pred_aspect_logit = outputs['pred_aspects_logit']  # [bsz, num_generated_aspect, num_aspect+1]
        target_classes_o = torch.cat([t[i] for t, (i, _) in zip(gold_aspect_split, indices)]).to(
            pred_aspect_logit.device)
        target_classes = torch.full(pred_aspect_logit.shape[:2], 0, dtype=torch.int64, device=pred_aspect_logit.device)
        target_classes[idx] = target_classes_o
        loss_aspect = F.cross_entropy(pred_aspect_logit.flatten(0, 1), target_classes.flatten(0, 1),
                                      ignore_index=self.args.category2id['<PAD>'])

        # loss2
        pred_polarity_logit = outputs['pred_polarities_logit']  # [bsz, num_generated_aspect, num_polarity+1]
        target_polarities_o = torch.cat([t[i] for t, (i, _) in zip(gold_polarity_split, indices)]).to(
            pred_polarity_logit.device)
        target_polarities = torch.full(pred_polarity_logit.shape[:2], 0, dtype=torch.int64,
                                       device=pred_polarity_logit.device)
        target_polarities[idx] = target_polarities_o
        loss_polarity = F.cross_entropy(pred_polarity_logit.flatten(0, 1), target_polarities.flatten(0, 1),
                                        ignore_index=polarity2id['<PAD>'])
        loss = loss_aspect + loss_polarity
        return loss, {'loss_polarity': loss_polarity.item(), 'loss_aspect': loss_aspect.item()}
