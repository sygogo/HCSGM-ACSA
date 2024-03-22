import os.path
import pickle

import torch
from cachetools.func import lru_cache
import nltk
from nltk.corpus import stopwords
import numpy as np
from transformers import BertTokenizer, AutoTokenizer, AutoModel

from src.utils.misc import polarity2id

stops = set(stopwords.words("english"))
lemmatizer = nltk.WordNetLemmatizer()


@lru_cache(1000000000)
def lemmatize(w: str):
    # caching the word-based lemmatizer to speed the process up
    return lemmatizer.lemmatize(w)


def remove_accents(text: str) -> str:
    accents_translation_table = str.maketrans(
        "áéíóúýàèìòùỳâêîôûŷäëïöüÿñÁÉÍÓÚÝÀÈÌÒÙỲÂÊÎÔÛŶÄËÏÖÜŸ",
        "aeiouyaeiouyaeiouyaeiouynAEIOUYAEIOUYAEIOUYAEIOUY"
    )
    return text.translate(accents_translation_table)


def get_cate2id(path, cate2id, cate2counter):
    for line in open(path, 'r'):
        objs = line.strip().split('\t')
        _, opinions = objs[0], objs[1:]
        for opinion in opinions:
            objs = opinion.split('#')
            category = '#'.join(objs[:-1])
            if category not in cate2id:
                cate2id[category] = len(cate2id)
            if category not in cate2counter:
                cate2counter[category] = 1
            else:
                cate2counter[category] += 1
    return cate2id, cate2counter


class DataGenerator(object):
    """
     To generate data from xml
    """

    def __init__(self, data_source, data_category, source_dir, target_dir, bert_directory, encoder_type):
        self.data_source = data_source
        self.data_category = data_category
        self.target_dir = target_dir
        self.source_dir = source_dir
        self.encoder_type = encoder_type
        self.bertTokenizer = AutoTokenizer.from_pretrained(bert_directory, do_lower_case=False)
        self.train_source_file = '{}/{}_{}_train.tsv'.format(self.source_dir, self.data_category, self.data_source)
        self.test_source_file = '{}/{}_{}_test.tsv'.format(self.source_dir, self.data_category, self.data_source)
        self.val_source_file = '{}/{}_{}_val.tsv'.format(self.source_dir, self.data_category, self.data_source)
        self.train_target_file = '{}/{}_{}_{}_train.pkl'.format(self.target_dir, self.data_source, self.data_category, self.encoder_type)
        self.test_target_file = '{}/{}_{}_{}_test.pkl'.format(self.target_dir, self.data_source, self.data_category, self.encoder_type)
        self.val_target_file = '{}/{}_{}_{}_val.pkl'.format(self.target_dir, self.data_source, self.data_category, self.encoder_type)
        self.category2id_target_file = '{}/{}_{}_category2id.pkl'.format(self.target_dir, self.data_source, self.data_category)
        self.category2embedding_target_file = '{}/{}_{}_category2embedding.pkl'.format(self.target_dir, self.data_source, self.data_category)

    def __processing__(self, path, cate2id):
        dataset = []
        total, only_has_single_aspect_total = 0, 0
        max_length = 0
        single_length, multiple_length = [], []
        aspect_number_count = {}
        id2cate = {v: k for k, v in cate2id.items()}
        for line in open(path, 'r'):
            # one sentence
            aspects = []
            polarities = []
            objs = line.strip().split('\t')
            sentence, opinions = objs[0], objs[1:]
            text = sentence.lower()
            # ADD c
            text_token_id = self.bertTokenizer.encode(text, add_special_tokens=True)
            length = len(text_token_id)
            if length > max_length:
                max_length = length
            # get polarity
            for opinion in opinions:
                objs = opinion.split('#')
                category = '#'.join(objs[:-1])
                # -1 neg, 0 neu, 1 pos
                if int(objs[-1]) == -1:
                    polarity = polarity2id['NEG']
                elif int(objs[-1]) == 0:
                    polarity = polarity2id['NEU']
                elif int(objs[-1]) == 1:
                    polarity = polarity2id['POS']
                if cate2id[category] not in aspects:
                    aspects.append(cate2id[category])
                    polarities.append(int(polarity))
            if len(aspects) == 0:
                continue
            total += 1
            # 单标签分类，标签设置
            if len(aspects) > 1:
                is_single_aspect = 0
                multiple_length.append(length)
            else:
                is_single_aspect = 1
                single_length.append(length)
                only_has_single_aspect_total += 1
            if len(aspects) not in aspect_number_count:
                aspect_number_count[len(aspects)] = 1
            else:
                aspect_number_count[len(aspects)] += 1
            aspects.append(cate2id['<EOS>'])
            polarities.append(polarity2id['<EOS>'])
            dataset.append((text, text.split(), text_token_id, {"aspects": aspects, "polarities": polarities, "single_aspect": is_single_aspect}))
        print('Total:{},Only Has Single Aspect Total:{},Radio:{},Category:{},Max Length:{}'.format(total, only_has_single_aspect_total, only_has_single_aspect_total / total, len(cate2id), max_length))
        print('Single aspect sentences avg length is {}, multiple aspect sentences avg length is {}'.format(np.mean(single_length), np.mean(multiple_length)))
        print(aspect_number_count)
        return dataset

    def processing(self):
        cate2id = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2}
        cate2counter = {}
        cate2id, cate2counter = get_cate2id(self.train_source_file, cate2id, cate2counter)
        cate2id, cate2counter = get_cate2id(self.test_source_file, cate2id, cate2counter)
        if os.path.exists(self.val_source_file):
            cate2id, cate2counter = get_cate2id(self.val_source_file, cate2id, cate2counter)
        train_dataset = self.__processing__(self.train_source_file, cate2id, cate2counter)
        test_dataset = self.__processing__(self.test_source_file, cate2id, cate2counter)
        if os.path.exists(self.val_source_file):
            val_dataset = self.__processing__(self.val_source_file, cate2id, cate2counter)
        else:
            np.random.shuffle(train_dataset)
            train_total = len(train_dataset)
            val_dataset = train_dataset[:int(0.1 * train_total)]
            train_dataset = train_dataset[int(0.1 * train_total):]
        print('Train:{},Test:{},Valid:{}'.format(len(train_dataset), len(test_dataset), len(val_dataset)))
        pickle.dump(train_dataset, open(self.train_target_file, 'wb'))
        pickle.dump(test_dataset, open(self.test_target_file, 'wb'))
        pickle.dump(val_dataset, open(self.val_target_file, 'wb'))
        pickle.dump(cate2id, open(self.category2id_target_file, 'wb'))
