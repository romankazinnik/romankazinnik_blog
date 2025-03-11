# -*- coding: utf-8 -*-
import os
import random
import functools
import csv
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import evaluate
import json
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, classification_report, balanced_accuracy_score, accuracy_score
import os
from unity_data import create_dataset

from scipy.stats import pearsonr
from datasets import Dataset, DatasetDict
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    TrainerCallback
)
from pathlib import Path
# path = Path('us-patent-phrase-to-phrase-matching')
# "meta-llama/Meta-Llama-3-8B"
MODELS = {'mistral': "mistralai/Mistral-7B-v0.1", "lama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "microsoft": "microsoft/phi-2", "google": "google/gemma-2b"}
MODEL_KEY = "microsoft"
model_name = MODELS[MODEL_KEY]
OUTPUT_DIR = f"unity-prompt-classifier-{MODEL_KEY}"

# !ls {path}
def calculate_class_weights(df_train):

    df_train.score_category.value_counts(normalize=True)

    class_weights=(1/df_train.score_category.value_counts(normalize=True).sort_index()).tolist()
    class_weights=torch.tensor(class_weights)
    class_weights=class_weights/class_weights.sum()
    return class_weights

def split_data(train_df):
    train_size = 0.8 # 80% of data
    test_size = 0.2  # 20% of data
    df_train, df_val2 = train_test_split(train_df, train_size=train_size, test_size=test_size, random_state=42)

    # Split the temporary data into validation and test sets
    df_val, df_val_test = train_test_split(df_val2, test_size=0.5, random_state=42)

    # TODO: remove this
    # df_val_test = df_val_test.sample(n=20, random_state=42, replace=True)
    # df_val = df_val.sample(n=20, random_state=42, replace=True)
    #df_train = df_train.sample(n=60)
    return df_train, df_val, df_val_test

# text and label
def preprocess_dataset():   
    
    _, train_df = create_dataset()
    # Shuffle the DataFrame rows
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_df = train_df.rename(columns={'text': 'input'})
    train_df = train_df.rename(columns={'label': 'score_category'})

    category_map = {category: index for index, category in enumerate(train_df['score_category'].unique())}
    category_map_from_index = {index:category for category,index in category_map.items()}

    train_df['score_category']=train_df['score_category'].apply(lambda l:category_map[l])

    df_train, df_val, df_val_test = split_data(train_df)
    
    #num_labels = len(category_map)
    #label_names = list(category_map.keys())
    return df_train, df_val, df_val_test, category_map, category_map_from_index

