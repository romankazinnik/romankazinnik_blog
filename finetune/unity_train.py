import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant= False,
) 
from unity_data import create_dataset

my_device = "cuda"
# Define constants
MODELS = {'mistral': "mistralai/Mistral-7B-v0.1", "lama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "microsoft": "microsoft/phi-2", "google": "google/gemma-2b"}
MODEL_KEY = "lama"
MODEL_NAME = MODELS[MODEL_KEY]

OUTPUT_DIR = f"unity-prompt-classifier-{MODEL_KEY}"
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
NUM_EPOCHS = 10
MAX_SEQ_LENGTH = 512


# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Format data for training
def format_for_training(row):
    return f"Prompt: {row['prompt']}\nClassification: {row['label']}"

# Load and prepare model and tokenizer
def prepare_model_and_tokenizer():
    # Load base model   
    
    print("Loading model and tokenizer...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with 8-bit quantization to reduce memory usage
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto", 
        torch_dtype=torch.float16, 
        #torch_dtype=torch.bfloat16, 
        quantization_config=bnb_config,
        # cpu
        #low_cpu_mem_usage=True, device_map='cpu', torch_dtype=torch.float32, 
        #llm_int8_enable_fp32_cpu_offload=True,load_in_8bit=False,
        use_cache=False,
    )

    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    # gradient_checkpointing=False
    model.gradient_checkpointing_enable({"use_reentrant": False})
    
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    
    # Get PEFT model
    model = get_peft_model(model, peft_config)
    
    return model, tokenizer

# Tokenize dataset
def tokenize_dataset(dataset, tokenizer):
    def tokenize_function(examples):
        texts = [format_for_training({"prompt": p, "label": l}) for p, l in zip(examples["prompt"], examples["label"])]
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["prompt", "label"]
    )
    
    return tokenized_dataset

# Main training function
def train():
    # Create dataset
    df, df_train = create_dataset()
    
    # Convert to HuggingFace dataset
    dataset = Dataset.from_pandas(df)
    
    # Split dataset
    dataset = dataset.shuffle(seed=42)
    train_test_split = dataset.train_test_split(test_size=0.2)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]
    
    # Prepare model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer()
    
    # Tokenize datasets
    train_dataset = tokenize_dataset(train_dataset, tokenizer)
    test_dataset = tokenize_dataset(test_dataset, tokenizer)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,    # Accumulate gradients to compensate for small batch size
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        #eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        fp16=True,
        #label_names=["simple", "complex", "invalid"],
        # cpu
        save_steps=100,
        logging_steps=1,
        # fp16=False,                       # Disable mixed precision training on CPU
        report_to="none",                 # Disable wandb or other reporting
        # remove_unused_columns=False,      # PEFT models need this
        # Explicitly specifying we want GPU or CPU
        no_cuda=False,
    )
    
    # Set up data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # Set up trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    # Save model
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    # clean GPU memory
    with torch.no_grad():
        torch.cuda.empty_cache()    
    print(f"Model and tokenizer saved to {OUTPUT_DIR}")

# Function to use the fine-tuned model for inference
def inference(prompt, model_path=OUTPUT_DIR):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        #load_in_8bit=True,
        quantization_config=bnb_config,
        #
        #load_in_8bit=False,
        #llm_int8_enable_fp32_cpu_offload=True
        #
        use_cache = True
    )
    # model.config.use_cache = True
    
    # Format prompt
    formatted_prompt = f"Prompt: {prompt}\nClassification:"
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(my_device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.2,
            do_sample=True,
        )
    
    # Decode
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract classification
    classification = result.split('\n')[1].split("Classification:")[-1].split(' ')[1].strip()
    return classification, result

if __name__ == "__main__":
    # Train the model
    train()
    
    # Test inference with some examples
    test_prompts = [
        "How do I add force to a rigidbody in Unity?",
        "My game crashes when I try to load a new scene. Can you help me debug?",
        "What's the recipe for beef stroganoff?",
    ]
    
    for prompt in test_prompts:
        classification, result = inference(prompt)
        print(f"Prompt: {prompt}")
        print(f"Classification: {classification}")
        #print(f'result={result}')

