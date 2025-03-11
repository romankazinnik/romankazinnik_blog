# Fine-tuning for Text Classification

## Instructions

### Classify text with fine-tuned model

```bash
python3 train.py 

python3 inference.py 
```
### Classify text with causal language model, fine-tuning and prompt engineering:

```bash
python3 unity_train.py 

python3 unity_inference.py 
```

## Fine-tuning for Text Classification:

### Approach 1. Text Generation with Classification Label as part of text

Train the model to generate text that naturally appends the classification label at the end.

### Approach 2. Sequence Classification Head

Add a sequence classification head (linear layer) on top of the LLaMa Model transformer. This setup is similar to GPT-2 and focuses on classifying the sentiment based on the last relevant token in the sequence.
- **Training Objective**: Minimize cross-entropy loss between the predicted and the actual labels.

## Environment setup instructions

```bash

uv venv --python 3.11 uv_lora

source uv_lora/bin/activate

uv pip install -r requirements_lora.txt 

Alternatively, install the dependencies manually:

uv pip install "torch==2.2.2" tensorboard  

 uv pip install --upgrade "transformers==4.40.0" "datasets==2.18.0" "accelerate==0.29.3" "evaluate==0.4.1" "bitsandbytes==0.43.1" "huggingface_hub==0.22.2
" "trl==0.8.6" "peft==0.10.0" "numpy==1.26.4" "pandas==2.2.2"  "scikit-learn==1.6.1" 

uv pip install --force-reinstall -v "triton==3.1.0"
```
## References

https://arxiv.org/abs/2305.14314

https://github.com/adidror005/youtube-videos/blob/main/LLAMA_3_Fine_Tuning_for_Sequence_Classification_Actual_Video.ipynb

https://colab.research.google.com/github/jkyamog/ml-experiments/blob/main/fine-tuning-qlora/LLAMA_3_Fine_Tuning_for_Sequence_Classification.ipynb#scrollTo=IqufrL0vwDod

https://colab.research.google.com/github/jkyamog/ml-experiments/blob/main/fine-tuning-qlora/LLAMA_3_Fine_Tuning_for_Sequence_Classification.ipynb

## Demo screenshots

### Classification: 1st epoch

<img src="e1.png" width="800" height="200" alt="Epoch 1 metrics">

**Dataset is very small (100 samples) and produces variance .**

### Classification: 100% accuracy after 10 epochs

<img src="e10f1.png" width="400" height="600" alt="Epoch 1 metrics">

**High accuracy for 3-class confusion matrix for training, evaluation, and test (unseen during trainig) datasets.**

### Causal Language Model: prompt sentence completion

<img src="causal10.png" width="700" height="150" alt="Epoch 1 metrics">

**10 epochs.**

### Requirements
* A GPU with 12GB VRAM or more.  Nvidia/L4 orbetter would work

### Big Picture Overview of Parameter Efficient Fine Tuning Methods like LoRA and QLoRA Fine Tuning for Sequence Classification

**Fine-tuning**
- LLMs are pre-trained on vast amounts of data for broad language understanding.
- Fine-tuning is crucial for specializing in specific domains or tasks, involving adjustments with smaller, relevant datasets.

**PEFT**
- PEFT modifies only a subset of the LLM's parameters, enhancing speed and reducing memory demands, making it suitable for less powerful devices.

**LoRA: Efficiency through Adapters**
- **Low-Rank Adaptation (LoRA):** Injects small trainable adapters into the pre-trained model.
- **Equation:** For a weight matrix $W$, LoRA approximates $W = W_0 + BA$, where $W_0$ is the original weight matrix, and $BA$ represents the low-rank modification through trainable matrices $B$ and $A$.
- Adapters learn task nuances while keeping the majority of the LLM unchanged, minimizing overhead.

**QLoRA: Compression and Speed**
- **Quantized LoRA (QLoRA):** Extends LoRA by quantizing the model’s weights, further reducing size and enhancing speed.
- **Innovations in QLoRA:**
  1. **4-bit Quantization:** Uses a 4-bit data type, NormalFloat (NF4), for optimal weight quantization, drastically reducing memory usage.
  2. **Low-Rank Adapters:** Fine-tuned with 16-bit precision to effectively capture task-specific nuances.
  3. **Double Quantization:** Reduces quantization constants from 32-bit to 8-bit, saving additional memory without accuracy loss.
  4. **Paged Optimizers:** Manages memory efficiently during training, optimizing for large tasks.

**Why PEFT Matters**
- **Rapid Learning:** Speeds up model adaptation.
- **Smaller Footprint:** Eases deployment with reduced model size.
- **Edge-Friendly:** Fits better on devices with limited resources, enhancing accessibility.

**Conclusion**
- PEFT methods like LoRA and QLoRA revolutionize LLM fine-tuning by focusing on efficiency, facilitating faster adaptability, smaller models, and broader device compatibility.



