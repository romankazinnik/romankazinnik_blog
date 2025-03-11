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


#model_name = "google/gemma-2b"
model_name = "mistralai/Mistral-7B-v0.1" # "microsoft/phi-2"
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 


MODEL_KEY = "google"
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

df_train, df_val, df_val_test, category_map, category_map_from_index = preprocess_dataset()
class_weights = calculate_class_weights(df_train)

# Converting pandas DataFrames into Hugging Face Dataset objects:
dataset_train = Dataset.from_pandas(df_train)
dataset_val = Dataset.from_pandas(df_val)

# Combine them into a single DatasetDict
dataset = DatasetDict({
    'train': dataset_train,
    'val': dataset_val,
})

quantization_config = BitsAndBytesConfig(
    load_in_4bit = False, # True, # enable 4-bit quantization
    bnb_4bit_quant_type = 'nf4', # information theoretically optimal dtype for normally distributed weights
    bnb_4bit_use_double_quant = False, # True, # quantize quantized weights //insert xzibit meme
    bnb_4bit_compute_dtype = torch.bfloat16 # optimized fp format for ML
)


def model_train(category_map):



    lora_config = LoraConfig(
        r = 16, # the dimension of the low-rank matrices
        lora_alpha = 8, # scaling factor for LoRA activations vs pre-trained weight activations
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        lora_dropout = 0.2, # 0.05, # dropout probability of the LoRA layers
        bias = 'none', # wether to train bias weights, set to 'none' for attention layers
        task_type = 'SEQ_CLS'
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        num_labels=len(category_map)
    )

    # prepare_model_for_kbit_training() function to preprocess the quantized model for training.

    model = prepare_model_for_kbit_training(model)
    #model

    # get_peft_model prepares a model for training with a PEFT method such as LoRA by wrapping the base model and PEFT configuration with get_peft_model"""

    model = get_peft_model(model, lora_config)
    # model

    ### Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token is None:
        tokenizer.eos_token_id = tokenizer.pad_token_id
        tokenizer.eos_token = tokenizer.pad_token

    """#### Update some model configs
    * Must use .cache = False as below or it crashes from my experience
    """

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    MAX_LEN = 512
    
    def llama_preprocessing_function(examples):
        return tokenizer(examples['input'], truncation=True, max_length=MAX_LEN)

    tokenized_datasets = dataset.map(llama_preprocessing_function, batched=True) 
    tokenized_datasets = tokenized_datasets.rename_column("score_category", "label")
    tokenized_datasets.set_format("torch")

    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

    return tokenized_datasets, model, tokenizer, collate_fn

tokenized_datasets, model, tokenizer, collate_fn = model_train(category_map)

def define_custom_trainer(model, tokenizer, collate_fn, class_weights):
    """### Define custom trainer with classweights
    * We will have a custom loss function that deals with the class weights and have class weights as additional argument in constructor
    """

    class CustomTrainer(Trainer):
        def __init__(self, *args, class_weights=None, **kwargs):
            super().__init__(*args, **kwargs)
            # Ensure label_weights is a tensor
            if class_weights is not None:
                self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.args.device)
            else:
                self.class_weights = None
            self.train_losses = [-999]
            self.eval_losses = []
            self.eval_f1_scores = []
        
        def training_step(self, model, inputs): # , num_items_in_batch):
            outputs = super().training_step(model, inputs) # , num_items_in_batch)
            self.train_losses.append(outputs.item())
            return outputs

        def compute_loss(self, model, inputs, return_outputs=False):
            # Extract labels and convert them to long type for cross_entropy
            labels = inputs.pop("labels").long()

            # Forward pass
            outputs = model(**inputs)

            # Extract logits assuming they are directly outputted by the model
            logits = outputs.get('logits')

            # Compute custom loss with class weights for imbalanced data handling
            if self.class_weights is not None:
                loss = F.cross_entropy(logits, labels, weight=self.class_weights)
            else:
                loss = F.cross_entropy(logits, labels)

            return (loss, outputs) if return_outputs else loss
        
        def evaluate(self, *args, **kwargs):
            try:
                metrics = super().evaluate(*args, **kwargs)
                self.eval_losses.append(metrics["eval_loss"])
                self.eval_f1_scores.append(metrics["eval_f1"])
                
                # Print metrics at each epoch
                print(f"\nEpoch (iter)={len(self.train_losses)} and {len(self.eval_f1_scores)}:")
                print(f"  Training Loss: {self.train_losses[-1]:.4f}")
                print(f"  Evaluation Loss: {self.eval_losses[-1]:.4f}")
                print(f"  Evaluation F1 Score: {self.eval_f1_scores[-1]:.4f}")
            except Exception as e:
                print(f"metrics={metrics} in compute_metrics: {e}")
            return # metrics
        
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        try:
            # it's a classification task, take the argmax
            predictions_processed = np.argmax(predictions, axis=1)

            # Calculate Pearson correlation
            pearson, _ = pearsonr(predictions_processed, labels)
            # Compute F1 score (macro average across all classes)
            f1 = f1_score(labels, predictions_processed, average="macro", zero_division=0)
            
            # Get detailed classification report
            report = classification_report(labels, predictions_processed, target_names=list(category_map.keys()),  zero_division=0, output_dict=True)

            return {
            'pearson': pearson,
                "accuracy": (predictions_processed == labels).mean(),
                "f1": f1,
                "precision": report["macro avg"]["precision"],
                "recall": report["macro avg"]["recall"]
            }
        except Exception as e:
            print(f"Error in compute_metrics: {e}")
            return {'pearson': None}
        
        

    training_args = TrainingArguments(
        output_dir = 'sequence_classification',
        learning_rate = 1e-4,
        per_device_train_batch_size = 4,
        per_device_eval_batch_size = 4,
        gradient_accumulation_steps=1,
        num_train_epochs = 10,
        weight_decay = 0.01,
        load_best_model_at_end = True,
        push_to_hub=False,
        #label_names=["labels"],
        report_to="none",                 # Disable wandb or other reporting
        # remove_unused_columns=False,      # PEFT models need this    
        save_steps=100,
        logging_steps=5,
        evaluation_strategy="steps", # 'epoch',
        save_strategy = "steps", # 'epoch',
        eval_steps=50,
        # fp16=False,                       # Disable mixed precision training on CPU
        no_cuda=False, # Explicitly specifying we want GPU or CPU
        # Arguments
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False} # 
        
    )

    """#### Define custom trainer"""

    trainer = CustomTrainer(
        model = model,
        args = training_args,
        train_dataset = tokenized_datasets['train'],
        eval_dataset = tokenized_datasets['val'],
        tokenizer = tokenizer,
        data_collator = collate_fn,
        compute_metrics = compute_metrics,
        class_weights=class_weights,
    )
    return trainer

trainer = define_custom_trainer(model, tokenizer, collate_fn, class_weights)


train_result = trainer.train()

print(f"========\n ================== DONE train_result = trainer.train()  \n\n") 


def make_predictions(tokenizer, model, df, batch_size=1):
  """
  batch_size - adjust this based on  system's memory capacity
  """
  sentences = df.input.tolist()

  all_outputs = []

  # Process the sentences in batches
  for i in range(0, len(sentences), batch_size):
      # Get the batch of sentences
      batch_sentences = sentences[i:i + batch_size]

      # Tokenize the batch
      inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)

      # Move tensors to the device where the model is (e.g., GPU or CPU)
      inputs = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu') for k, v in inputs.items()}

      # Perform inference and store the logits
      with torch.no_grad():
          outputs = model(**inputs)
          all_outputs.append(outputs['logits'].cpu())

  final_outputs = torch.cat(all_outputs, dim=0)
  df['predictions_logits_argmax']=final_outputs.argmax(axis=1).cpu().numpy()
  df['predictions_labels']=df['predictions_logits_argmax'].apply(lambda l:category_map_from_index[l])
  return 
### Analyze performance

def get_performance_metrics(df_test):
  y_pred = df_test.predictions_logits_argmax
  y_test = df_test.score_category

  print("Confusion Matrix:")
  print(confusion_matrix(y_test, y_pred))

  print("\nClassification Report:")
  target_names = [str(label) for label,index in category_map.items() if index in set(y_pred) | set(y_test)]
  report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0, output_dict=False)
  print(report) # json.dumps(report, indent=4))

  print("Balanced Accuracy Score:", balanced_accuracy_score(y_test, y_pred))
  print("Accuracy Score:", accuracy_score(y_test, y_pred))
  return 

if False:
    print(f"\n *************** df_val = {df_val.shape}\n")
    make_predictions(tokenizer, model, df_val)
    get_performance_metrics(df_val)
    print(f"\n *************** fold out (test) df_val_test = {df_val_test.shape}\n")
    make_predictions(tokenizer, model, df_val_test)
    get_performance_metrics(df_val_test)
    print(f"\n *************** df_train = {df_train.shape}\n")
    make_predictions(tokenizer, model, df_train)
    get_performance_metrics(df_train)

print(f" ================== save weights \n\n")

metrics = train_result.metrics
max_train_samples = len(dataset_train)
metrics["train_samples"] = min(max_train_samples, len(dataset_train))
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

trainer.save_model(f"{OUTPUT_DIR}/saved_trainer")
# Save model
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
# clean GPU memory
with torch.no_grad():
    torch.cuda.empty_cache()    
print(f"Model and tokenizer saved to {OUTPUT_DIR}")

print(f" ================== make_predictions \n\n")

# Function to use the fine-tuned model for inference
def inference(df_train, model_path=OUTPUT_DIR):
    print(f"\n *************** inference:  {df_val_test.shape}\n")
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        num_labels=len(category_map),
    )
    make_predictions(tokenizer, model, df_train)
    get_performance_metrics(df_train)
    return 

print(f"\n *************** df_val = {df_val.shape}\n")
inference(df_val, model_path=OUTPUT_DIR)
print(f"\n *************** fold out (test) df_val_test = {df_val_test.shape}\n")
inference(df_val_test, model_path=OUTPUT_DIR)
print(f"\n *************** df_train = {df_train.shape}\n")
inference(df_train, model_path=OUTPUT_DIR)


