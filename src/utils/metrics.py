import scipy
import torch
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import numpy as np


def compute_aspect_metric_micro(pred_aspect, targets, num_category, average='micro'):
    gold_aspect = [i['aspects'] for i in targets]
    num_category = num_category - 3
    pred_label_list, gold_label_list = [], []
    for (pred, gold) in zip(np.argmax(np.array(pred_aspect), axis=-1), gold_aspect):
        pred_labels = [0] * num_category
        gold_labels = [0] * num_category
        for line in pred:
            pred_labels[line - 3] = 1
        for line in gold:
            gold_labels[line - 3] = 1
        pred_label_list.append(pred_labels)
        gold_label_list.append(gold_labels)
    r = recall_score(y_true=gold_label_list, y_pred=pred_label_list, average=average)
    p = precision_score(y_true=gold_label_list, y_pred=pred_label_list, average=average)
    f = f1_score(y_true=gold_label_list, y_pred=pred_label_list, average=average)
    acc = accuracy_score(y_true=np.array(gold_label_list).flatten(), y_pred=np.array(pred_label_list).flatten())
    return p, r, f, acc


def compute_pair_metric_micro(pred_aspect, pred_polarity, targets, num_category, num_polarity, average='micro'):
    num_polarity = num_polarity - 3
    num_category = num_category - 3
    pred_aspect, pred_polarity = np.argmax(np.array(pred_aspect), axis=-1), np.argmax(np.array(pred_polarity), axis=-1)
    gold_aspect, gold_polarity = [i['aspects'] for i in targets], [i['polarities'] for i in targets]
    pred_pairs = convert_pairs(pred_aspect, pred_polarity)
    gold_pairs = convert_pairs(gold_aspect, gold_polarity)
    pred_label_list = []
    gold_label_list = []
    for (pred, gold) in zip(pred_pairs, gold_pairs):
        pred_labels = [0] * (num_polarity * num_category)
        gold_labels = [0] * (num_polarity * num_category)
        for line in pred:
            aspect = line[0] - 3
            polarity = line[1] - 3
            pred_labels[aspect * num_polarity + polarity] = 1
        for line in gold:
            aspect = line[0] - 3
            polarity = line[1] - 3
            gold_labels[aspect * num_polarity + polarity] = 1
        pred_label_list.append(pred_labels)
        gold_label_list.append(gold_labels)
    r = recall_score(y_true=gold_label_list, y_pred=pred_label_list, average=average)
    p = precision_score(y_true=gold_label_list, y_pred=pred_label_list, average=average)
    f = f1_score(y_true=gold_label_list, y_pred=pred_label_list, average=average)
    acc = accuracy_score(y_true=np.array(gold_label_list).flatten(), y_pred=np.array(pred_label_list).flatten())
    return p, r, f, acc


def compute_classifier_metric(outputs, targets):
    preds = [1 if i >= 0.5 else 0 for i in outputs['pred_single_aspect_logit']]
    targets = [i['has_single_aspect'] for i in targets]
    f1 = f1_score(targets, preds)
    return f1


def compute_pair_metric(tp, recall_number, ground_number):
    recall = tp / ground_number
    precision = tp / (recall_number + 1e-10)
    f1 = 2 * recall * precision / (recall + precision + 1e-10)
    return precision, recall, f1


def compute_pair_metric_micro_manual(pred_aspect, pred_polarity, targets):
    pred_aspect, pred_polarity = np.argmax(np.array(pred_aspect), axis=-1), np.argmax(np.array(pred_polarity), axis=-1)
    gold_aspect, gold_polarity = [i['aspects'] for i in targets], [i['polarities'] for i in targets]
    pred_pairs = convert_pairs(pred_aspect, pred_polarity)
    gold_pairs = convert_pairs(gold_aspect, gold_polarity)
    tp, recall_number, ground_number, real_recall_number = 0, 0, 0, 0
    for (pred, gold) in zip(pred_pairs, gold_pairs):
        # real_recall_number += len(pred)
        pred = set(pred)
        gold = set(gold)
        recall_number += len(pred)
        ground_number += len(gold)
        tp += len(pred & gold)
    precision, recall, f1 = compute_pair_metric(tp, recall_number, ground_number)
    return precision, recall, f1


def convert_pairs(aspects, polarities):
    pair_list = []
    for (a, p) in zip(aspects, polarities):
        pairs = []
        for (i, j) in zip(a, p):
            # when meet eos , break
            if i == 2:
                break
            pairs.append((i, j))
        pair_list.append(pairs)
    return pair_list
