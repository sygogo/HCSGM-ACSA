# HCSGM-ACSA

The code for paper "A Hierarchical Sequence-to-Set Model with Coverage Mechanism for Aspect Category Sentiment Analysis". Our paper is accepted by COLING 2024.

## Requirements
```shell
pytorch~=1.11.0
numpy~=1.20.1
scipy~=1.6.2
scikit-learn~=0.24.1
tqdm~=4.59.0
nltk~=3.6.1
```

## Data Processing
```shell
python preprocess.py  --data_source="2014"  --data_category="mams"  --source_dir="data/raw"  --target_dir="data/processed" 
```


## Train Model
```shell
python train.py
```


## Evaluation Model
```shell
python eval.py
```
