import gc
import os
import pickle
import sys
from collections import OrderedDict
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data.backward_compatibility import worker_init_fn
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, set_seed
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer
import torch
from src.data_loader.ReviewDataset import ReviewDataset

from src.model.Seq2Seq import Seq2Seq
from src.utils.metrics import compute_pair_metric_micro_manual, convert_pairs, compute_pair_metric_micro, compute_aspect_metric_micro
import numpy as np
import scipy
from src.utils.misc import init_logging, polarity2id


class BaseTrainer(object):
    """
    Trainer
    """

    def __init__(self, args):
        self.args = args
        self.data_source = args.data_source
        self.data_category = args.data_category
        self.target_dir = args.target_dir
        self.model_dir = '{}/BS-{}_DS-{}_DC-{}_AE-{}_DE-{}_LR-{}'.format(args.model_dir, args.batch_size, args.data_source, args.data_category, args.encoder_type, args.decoder_type, args.decoder_lr)
        self.bertTokenizer = AutoTokenizer.from_pretrained(args.bert_directory)
        self.train_target_file = '{}/{}_{}_{}_train.pkl'.format(self.target_dir, self.data_source, self.data_category, self.args.encoder_type)
        self.test_target_file = '{}/{}_{}_{}_test.pkl'.format(self.target_dir, self.data_source, self.data_category, self.args.encoder_type)
        self.val_target_file = '{}/{}_{}_{}_val.pkl'.format(self.target_dir, self.data_source, self.data_category, self.args.encoder_type)
        self.category2id_target_file = '{}/{}_{}_category2id.pkl'.format(self.target_dir, self.data_source, self.data_category)
        self.train_dataset = pickle.load(open(self.train_target_file, 'rb'))
        self.test_dataset = pickle.load(open(self.test_target_file, 'rb'))
        self.category2id = pickle.load(open(self.category2id_target_file, 'rb'))
        self.args.category2id = self.category2id
        self.id2category = {v: k for k, v in self.category2id.items()}
        self.logging = init_logging(log_file=self.model_dir + '/trainning.log', stdout=True)

    def init_grouped_params(self, args, model):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        component = ['encoder', 'decoder']
        grouped_params = [
            {
                'params': [p for n, p in model.named_parameters() if
                           not any(nd in n for nd in no_decay) and component[0] in n],
                'weight_decay': args.weight_decay,
                'lr': args.encoder_lr
            },
            {
                'params': [p for n, p in model.named_parameters() if
                           any(nd in n for nd in no_decay) and component[0] in n],
                'weight_decay': 0.0,
                'lr': args.encoder_lr
            },
            {
                'params': [p for n, p in model.named_parameters() if
                           not any(nd in n for nd in no_decay) and component[1] in n],
                'weight_decay': args.weight_decay,
                'lr': args.decoder_lr
            },
            {
                'params': [p for n, p in model.named_parameters() if
                           any(nd in n for nd in no_decay) and component[1] in n],
                'weight_decay': 0.0,
                'lr': args.decoder_lr
            }
        ]
        return grouped_params

    def get_dataloader(self, data, batch_size, num_workers=0, shuffle=False, sampler=None):
        """
        get data loader
        :param data:
        :param batch_size:
        :param num_workers:
        :return:
        """
        dataset = ReviewDataset(data, self.bertTokenizer)
        if sampler is None:
            loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, worker_init_fn=worker_init_fn, collate_fn=dataset.collate_fn, shuffle=shuffle)
        else:
            loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, worker_init_fn=worker_init_fn, collate_fn=dataset.collate_fn, sampler=sampler)
        return loader

    def create_model(self, args):
        if args.model == 'Seq2Seq':
            model = Seq2Seq(self.args).cuda()
        else:
            raise ValueError("There is no model:{}".format(args.model))
        model_path = os.path.join(self.model_dir, '{}_seed_{}.pt'.format(args.model, self.args.seed))
        if args.set_loss:
            model_path = os.path.join(self.model_dir, '{}_seed_set_loss_{}.pt'.format(args.model, self.args.seed))
        return model, model_path

    def train(self):
        if self.args.local_rank in [0, -1]:
            self.logging.info('=======================================Start Training==========================================')
            for arg in vars(self.args):
                if arg not in ['vocab2id', 'embeddings']:
                    self.logging.info(arg + ":" + str(getattr(self.args, arg)))
            self.logging.info("use gpu:{}".format(torch.cuda.device_count()))
        train_dataset = pickle.load(open(self.train_target_file, 'rb'))
        valid_dataset = pickle.load(open(self.val_target_file, 'rb'))
        self.__train__(train_dataset, valid_dataset)

    def __train__(self, train_dataset, validation_dataset):
        """
        Train model
        :param args:
        """
        model, path = self.create_model(self.args)
        if self.args.dist:
            sampler = DistributedSampler(train_dataset)
        else:
            sampler = None
        train_loader = self.get_dataloader(train_dataset, self.args.batch_size, self.args.num_workers, sampler=sampler)
        # use gpus
        validation_loader = self.get_dataloader(validation_dataset, self.args.batch_size, shuffle=False)
        num_batch = len(train_loader)
        grouped_params = self.init_grouped_params(self.args, model)
        if self.args.dist:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.args.local_rank], output_device=self.args.local_rank, find_unused_parameters=True)
        num_train_optimization_steps = num_batch * self.args.num_epoch
        num_warmup_steps = int(0.1 * num_train_optimization_steps)
        optimizer = AdamW(params=grouped_params, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_optimization_steps)
        best_f1 = 0
        patient = 0
        for epoch in range(self.args.num_epoch):
            if self.args.dist:
                train_loader.sampler.set_epoch(epoch)
            loss_total, loss_polarity_total, loss_aspect_total = 0, 0, 0
            for batch in tqdm(train_loader, total=num_batch, ncols=50, disable=self.args.local_rank in [0, -1]):
                optimizer.zero_grad()
                outputs, losses, losses_info = model(batch)
                loss_total += losses.item()
                loss_polarity_total += losses_info['loss_polarity']
                loss_aspect_total += losses_info['loss_aspect']
                losses.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
                optimizer.step()
                scheduler.step()
                # start to evaluation
            torch.cuda.empty_cache()
            gc.collect()
            if patient > self.args.early_stopping:
                print('Training Completed!')
                sys.exit(-1)
            if self.args.local_rank in [0, -1]:
                model.eval()
                self.args.mode = 'eval'
                with torch.no_grad():
                    precision, recall, f1, acc, polarity_error = self.inner_eval(validation_loader, model)
                if f1 > best_f1:
                    best_f1 = f1
                    self.save_model(model, path)
                    patient = 0
                else:
                    if epoch >= 50:
                        patient += 1
                model.train()
                self.args.mode = 'train'
                self.logging.info('Epoch:{} || encoder lr is {},decoder lr is {}, loss is {}, validation f1 is {}, best f1 is {}'.format(epoch, optimizer.param_groups[0]['lr'], optimizer.param_groups[2]['lr'], round(loss_total / num_batch, 4), f1, best_f1))
                self.logging.info('Loss Details || {}'.format({'loss_polarity_total': round(loss_polarity_total / num_batch, 4), 'loss_aspect_total': round(loss_aspect_total / num_batch, 4)}))

    def save_model(self, model, path):
        torch.save(model.state_dict(), open(path, 'wb'))

    def load_model(self, model, path):
        state_dict = torch.load(open(path, 'rb'))
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if str(k).startswith('module'):
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        return model.cuda()

    def eval(self, verbose=False):
        test_loader = self.get_dataloader(self.test_dataset, batch_size=16)
        model, path = self.create_model(self.args)
        model = self.load_model(model, path)
        model.eval()
        p, r, f1, acc, polarity_error, p_asp, r_asp, f_asp, acc_asp = self.inner_eval(test_loader, model, verbose, required_asp=True)
        results = {'f1_all': f1, 'recall_all': r, 'precision_all': p, "acc_all": acc, "polarity_error": polarity_error}
        test_loader_single = self.get_dataloader([i for i in self.test_dataset if i[-1]['single_aspect'] == 1], batch_size=self.args.batch_size)
        if len(test_loader_single) > 0:
            test_loader_multiple = self.get_dataloader([i for i in self.test_dataset if i[-1]['single_aspect'] == 0], batch_size=self.args.batch_size)
            p_s, r_s, f1_s, acc_s, polarity_error_s, p_asp_s, r_asp_s, f_asp_s, acc_asp_s = self.inner_eval(test_loader_single, model, verbose, required_asp=True)
            p_m, r_m, f1_m, acc_m, polarity_error_m, p_asp_m, r_asp_m, f_asp_m, acc_asp_m = self.inner_eval(test_loader_multiple, model, verbose, required_asp=True)
            results.update({"f1_single": f1_s, "recall_single": r_s, "precision_single": p_s, "acc_single": acc_s, "polarity_error_s": polarity_error_s,
                            "f1_mul": f1_m, "recall_mul": r_m, "precision_mul": p_m, "acc_mul": acc_m, "polarity_error_m": polarity_error_m, "acc_asp_s": acc_asp_s, "acc_asp_m": acc_asp_m,
                            "f_asp_s": f_asp_s, "f_asp_m": f_asp_m})
        results['acc_asp'] = acc_asp
        results['f_asp'] = f_asp
        return results

    def inner_eval(self, test_loader, model, verbose=False, required_asp=False):
        pred_aspects_list, pred_polarities_list = [], []
        gold_pairs_list = []
        raw_data = test_loader.dataset.data
        polarity_error = 0
        for batch in test_loader:
            targets = batch[-1]
            outputs = model(batch)
            pred_aspects_list.extend(outputs['pred_aspects_logit'].tolist())
            pred_polarities_list.extend(outputs['pred_polarities_logit'].tolist())
            gold_pairs_list.extend(targets)
        precision, recall, f1, acc = compute_pair_metric_micro(pred_aspects_list, pred_polarities_list, gold_pairs_list, len(self.args.category2id), len(polarity2id))
        if required_asp:
            p_asp, r_asp, f_asp, acc_asp = compute_aspect_metric_micro(pred_aspects_list, gold_pairs_list, len(self.args.category2id))
            # precision, recall, f1 = compute_pair_metric_micro_manual(pred_aspects_list, pred_polarities_list, gold_pairs_list)
        instance_total = 0
        error_polarity_total = 0
        for index, (line, pred_aspects, pred_polarities) in enumerate(zip(raw_data, pred_aspects_list, pred_polarities_list)):
            text = line[0]
            obj = line[-1]
            gold_aspects, gold_polarities = obj['aspects'], obj['polarities']
            pred_aspects, pred_polarities = np.argmax(np.array(pred_aspects), axis=-1), np.argmax(np.array(pred_polarities), axis=-1)
            pred_pairs = convert_pairs([pred_aspects], [pred_polarities])
            gold_pairs = convert_pairs([gold_aspects], [gold_polarities])
            pred_pairs = [[j for j in i] for i in pred_pairs]
            gold_pairs = [[j for j in i] for i in gold_pairs]
            tp = len(set(pred_pairs[0]) & set(gold_pairs[0]))
            recall_number = len(pred_pairs[0])
            truth_number = len(gold_pairs[0])
            instance_total += truth_number
            if tp == recall_number == truth_number:
                continue
            else:
                for pred in sorted(set(pred_pairs[0])):
                    for gold in sorted(gold_pairs[0]):
                        if pred[0] == gold[0]:
                            if pred[1] != gold[1]:
                                error_polarity_total += 1
            if verbose:
                print(''.join(["="] * 50))
                print('{}\n'.format(text))
                print('pred:{}'.format([(self.id2category.get(i[0], 'NA'), i[1]) for i in pred_pairs[0]]))
                print('gold:{}'.format([(self.id2category.get(i[0], 'NA'), i[1]) for i in gold_pairs[0]]))
            polarity_error = error_polarity_total / instance_total
        if required_asp:
            return precision, recall, f1, acc, polarity_error, p_asp, r_asp, f_asp, acc_asp
        return precision, recall, f1, acc, polarity_error
