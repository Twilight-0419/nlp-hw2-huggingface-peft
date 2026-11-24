# hw2_huggingface/dataHelper.py

import json
import random
from pathlib import Path
from typing import List
from datasets import Dataset, DatasetDict, load_dataset,concatenate_datasets, Value

## * You can add more helper functions or modify function arguments if needed

DATA_DIR = Path(__file__).resolve().parent / 'data'


def restaurant(sep_token: str):
    '''Load the ABSA restaurant dataset.'''
    polarity = {'positive': 0, 'negative': 1, 'neutral': 2}

    base = DATA_DIR / 'supervised' / 'restaurant_sup'
    with open(base / 'train.json','r',encoding='utf-8') as f:
        train_content=json.load(f)
    train_text=[]
    train_label=[]
    for i in train_content.values():
        train_text.append(i['term']+sep_token+i['sentence'])
        train_label.append(polarity[i['polarity']])

    with open(base / 'test.json','r',encoding='utf-8') as f:
            test_content=json.load(f)
    test_text=[]
    test_label=[]
    for i in test_content.values():
        test_text.append(i['term']+sep_token+i['sentence'])
        test_label.append(polarity[i['polarity']])

    return DatasetDict({
        'train':Dataset.from_dict({
            'text':train_text,
            'labels':train_label
        }),
        'test':Dataset.from_dict({
            'text':test_text,
            'labels':test_label
        })
    })


def laptop(sep_token: str):
    '''Load the ABSA laptop dataset.'''
    polarity = {'positive': 0, 'negative': 1, 'neutral': 2}
    base = DATA_DIR / 'supervised' / 'laptop_sup'

    with open(base / 'train.json','r',encoding='utf-8') as f:
        train_content=json.load(f)
    train_text=[]
    train_label=[]
    for i in train_content.values():
        train_text.append(i['term']+sep_token+i['sentence'])
        train_label.append(polarity[i['polarity']])

    with open(base / 'test.json','r',encoding='utf-8') as f:
        test_content=json.load(f)
    test_text=[]
    test_label=[]
    for i in test_content.values():
        test_text.append(i['term']+sep_token+i['sentence'])
        test_label.append(polarity[i['polarity']])

    return DatasetDict({
        'train':Dataset.from_dict({
            'text':train_text,
            'labels':train_label
        }),
        'test':Dataset.from_dict({
            'text':test_text,
            'labels':test_label
        })
    })


def acl(sep_token: str):
    base = DATA_DIR / 'supervised' / 'acl_sup'
    label_map = {'Background': 0, 'Uses': 1, 'CompareOrContrast': 2,
                 'Extends': 3, 'Motivation': 4, 'Future': 5}
    def load(split:str):
        content=[]
        with open(base / f'{split}.jsonl','r',encoding='utf-8') as f:
            for line in f:
                content.append(json.loads(line.strip()))
        split_text=[]
        split_labels=[]
        for i in content:
            split_text.append(i['text'])
            split_labels.append(label_map[i['label']])
        return Dataset.from_dict({
            'text':split_text,
            'labels':split_labels
        })
    return DatasetDict({
        'train':load('train'),
        'test':load('test')
    })


def agnews(sep_token: str):
    '''Load the AGNews dataset (test set onlys).'''
    dataset=load_dataset(
        'csv',
        data_files=str(DATA_DIR / 'supervised' / 'agnews_sup' / 'agnews_sup.csv'),
        column_names=['labels','title','text'],
        split='train',
    )
    dataset=dataset.remove_columns('title')
    def process_agnews(example):
        example['labels']=int(example['labels'])-1
        return example
    dataset=dataset.map(process_agnews)
    dataset=dataset.cast_column('text', Value('string'))
    datasetdict=dataset.train_test_split(
        test_size=0.1,
        seed=2025,
    )
    return datasetdict


def get_fs(dataset_name: str, sep_token: str, sample_size: int):
    dataset_name = dataset_name.removesuffix('_fs')
    loaders={
        'restaurant':restaurant,
        'laptop':laptop,
        'acl':acl,
        'agnews':agnews,
    }
    datasetdict=loaders[dataset_name](sep_token)
    def sample(split:str):
        dataset=datasetdict[split]
        num_labels=len(set(dataset['labels']))
        total_nums=sample_size
        base,remainder=divmod(total_nums,num_labels)
        sample_per_class={
            label:base + (1 if label < remainder else 0)
            for label in range(num_labels)
        }
        rng=random.Random(2025)
        all_labels=dataset['labels']
        selected_labels=[]
        for label , num_samples in sample_per_class.items():
            indices=[
                i for i,y in enumerate(all_labels)
                if y==label
            ]
            one_selected_labels=rng.sample(indices,num_samples)
            rng.shuffle(one_selected_labels)
            selected_labels.extend(one_selected_labels)
        return dataset.select(selected_labels)
    return DatasetDict({
        'train':sample('train'),
        'test':sample('test')
    })

## ! DO NOT change the function name or arguments
def get_dataset(dataset_name: str | List[str], sep_token: str) -> DatasetDict:
    '''
	dataset_name: str, the name of the dataset
	sep_token: str, the sep_token used by tokenizer(e.g. '<sep>')
	'''
    supervised_loaders = {
        'restaurant_sup': restaurant,
        'laptop_sup': laptop,
        'acl_sup': acl,
        'agnews_sup': agnews,
    }
    few_shot_loaders = {
        'restaurant_fs': 'restaurant',
        'laptop_fs': 'laptop',
        'acl_fs': 'acl',
        'agnews_fs': 'agnews',
    }

    def load_one(name: str) -> DatasetDict:
        if name in supervised_loaders:
            return supervised_loaders[name](sep_token)
        if name in few_shot_loaders:
            return get_fs(few_shot_loaders[name], sep_token, 32)
        raise ValueError(f"Unsupported dataset: {name}")

    def num_labels(datasetdict: DatasetDict) -> int:
        return max(
            max(datasetdict['train']['labels']),
            max(datasetdict['test']['labels']),
        ) + 1

    if isinstance(dataset_name, str):
        return load_one(dataset_name)

    elif isinstance(dataset_name, (list, tuple)):
        if not dataset_name:
            raise ValueError("Unsupported dataset: empty list!")

        datasetdicts = [load_one(name) for name in dataset_name]

        # 聚合时重新标号，保证不同任务的标签不重叠
        train_parts, test_parts = [], []
        offset = 0
        for datasetdict in datasetdicts:
            offset_num = num_labels(datasetdict)
            train_parts.append(datasetdict['train'].map(
                lambda example, off=offset: {'labels': example['labels'] + off},
                desc='Relabeling train split',
            ))
            test_parts.append(datasetdict['test'].map(
                lambda example, off=offset: {'labels': example['labels'] + off},
                desc='Relabeling test split',
            ))
            offset += offset_num

        return DatasetDict({
            'train': concatenate_datasets(train_parts),
            'test': concatenate_datasets(test_parts),
        })

    else:
        raise ValueError("Unsupported dataset!")
