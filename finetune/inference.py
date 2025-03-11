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

from data import preprocess_dataset, calculate_class_weights

from train import make_predictions, get_performance_metrics, quantization_config, MODELS

MODEL_KEY = "microsoft"
model_name = MODELS[MODEL_KEY]
OUTPUT_DIR = f"unity-prompt-classifier-{MODEL_KEY}"


# Function to use the fine-tuned model for inference
def load_weights(model_path=OUTPUT_DIR, category_map = None):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        num_labels=len(category_map),
    )
    return tokenizer, model

def inference(df_train, tokenizer, model, category_map_from_index, category_map):
    make_predictions(tokenizer, model, category_map_from_index, df_train)
    get_performance_metrics(df_train, category_map)
    return 

if __name__ == "__main__":
    """
    This script is used to run inference on the fine-tuned model.
    """
    df_train, df_val, df_val_test, category_map, category_map_from_index = preprocess_dataset()
    class_weights = calculate_class_weights(df_train)
    tokenizer, model = load_weights(model_path=OUTPUT_DIR,category_map=category_map)

    for df, df_name in zip([df_val, df_val_test, df_train], ["df_val", "df_val_test", "df_train"]):
        print(f"\n *************** {df_name} = {df.shape}\n")
        inference(df, tokenizer, model, category_map_from_index, category_map)
