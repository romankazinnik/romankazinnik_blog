#import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModel
import numpy as np

# Move model to GPU if available
def init(is_cuda:bool=True):
    device = torch.device("cuda" if is_cuda and torch.cuda.is_available() else "cpu")
        

    model_id = "llava-hf/llava-1.5-7b-hf"
    if device.type =="cuda":
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            # gpu
            load_in_4bit=True,
            # load_in_8bit=True, #`.to` is not supported for `8-bit` bitsandbytes models. Please use the model as it is
            # use_flash_attention_2=True
        ).to(device) # 0) # cuda
        # model = model.to(device)
    else:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
        )

    processor = AutoProcessor.from_pretrained(model_id)

    # Define a chat history and use `apply_chat_template` to get correctly formatted prompt
    # Each value in "content" has to be a list of dicts with types ("text", "image") 
    # text_prompt = "Answer in one sentence: is this an office building, hotel, or something else, is it modern or old, how many levels exactly does it have, and what is the color?"
    text_prompt = "Create a three-word description of shoe on the right: what is its color, type, sex (men, women, or unisex). Examples: red, sneaker, unisex."
    
    conversation = [
        {

        "role": "user",
        "content": [
            {"type": "text", "text": text_prompt}, #"What are these?"},
            {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    return device, model, model_id, processor, conversation, prompt


# https://huggingface.co/llava-hf/llava-1.5-7b-hf



def create_and_save_embedding(model, processor, device, text, model_name="llava-hf/llava-1.5-7b-hf", 
                              output_file="embedding.npy", debug=False):
    """
    Create vector embedding for text using the specified model and save to file.
    
    Args:
        text (str): Input text (two sentences in this case)
        model_name (str): Name of the model to use for embeddings
        output_file (str): File path to save the embedding
    """

    inputs = processor(text=text, return_tensors="pt").to(device)
    
    # Generate embeddings
    with torch.no_grad():
        # Run the model
        outputs = model(**inputs, output_hidden_states=True)
        
        # Debug: print output structure
        #print("Output type:", type(outputs))
        output_attrs = [attr for attr in dir(outputs) if not attr.startswith('_')]
        #print("Available output attributes:", output_attrs)
        
        # Try to extract embeddings using different strategies
        # Strategy 1: If decoder_hidden_states is available
        def test_read(fn):
                # Converted to PyTorch tensor with shape: torch.Size([1, 32064])
                # PyTorch tensor dtype: torch.float16
                numpy_array = np.load(fn)
                
                # Convert the NumPy array to a PyTorch tensor
                pytorch_tensor = torch.from_numpy(numpy_array)
                
                print(f"Loaded NumPy array with shape: {numpy_array.shape}")
                print(f"Converted to PyTorch tensor with shape: {pytorch_tensor.shape}")
                print(f"PyTorch tensor dtype: {pytorch_tensor.dtype}")
                return
        
        if hasattr(outputs, 'decoder_hidden_states') and outputs.decoder_hidden_states is not None:
            #print("Get the last layer's hidden states")
            hidden_states = outputs.decoder_hidden_states[-1]
            embedding1 = hidden_states.mean(dim=1).cpu().numpy()
            np.save(f"{output_file}_1.npy", embedding1)
            embedding11 = hidden_states.cpu().numpy()
            np.save(f"{output_file}_11.npy", embedding11)
            embedding = embedding1
            if debug:
                print(f"Embedding saved to {output_file}_1.npy")
                test_read(f"{output_file}_1.npy")

            
        
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            #print("# Strategy 2: If hidden_states is available")
            hidden_states = outputs.hidden_states[-1]
            embedding2 = hidden_states.mean(dim=1).cpu().numpy()
            np.save(f"{output_file}_2.npy", embedding2)
            embedding22 = hidden_states.cpu().numpy()
            np.save(f"{output_file}_22.npy", embedding22)
            embedding = embedding2
            if debug:
                print(f"Embedding saved to {output_file}_22.npy")
                test_read(f"{output_file}_2.npy")

        # Strategy 3: If logits are available, use them as a proxy for embeddings
        if hasattr(outputs, 'logits') and outputs.logits is not None:
            #print("Using logits as embeddings")
            embedding3 = outputs.logits.mean(dim=1).cpu().numpy()
            np.save(f"{output_file}_3.npy", embedding3)
            embedding33 = outputs.logits.cpu().numpy()
            np.save(f"{output_file}_33.npy", embedding33)
            embedding = embedding3
            if debug:
                print(f"Embedding saved to {output_file}_3.npy")
                test_read(f"{output_file}_3.npy")
        
        # Strategy 4: Access the language model directly
        #else:
        #print("Extracting embeddings directly from the language model")
        # Use the text tokens to get embeddings from the language model
        text_tokens = inputs.get('input_ids')
        embeddings, embedding4 = None, None
        try:
            # Get embeddings from the language model's transformer
            embeddings = model.language_model.model.embed_tokens(text_tokens)
            embedding4 = embeddings.mean(dim=1).cpu().numpy()
        except Exception as e:
            print(f"Error getting embeddings: {e}")
        if embeddings is None:
            try:
                embeddings = model.language_model.embed_tokens(text_tokens)
                embedding4 = embeddings.mean(dim=1).cpu().numpy()
            except Exception as e:
                print(f"Error getting embeddings: {e}")

        np.save(f"{output_file}_4.npy", embedding4)
        embedding44 = embeddings.cpu().numpy()
        np.save(f"{output_file}_44.npy", embedding44) # 11. 4096
        if debug:
            print(f"Embedding saved to {output_file}_4.npy")
            test_read(f"{output_file}_4.npy")        
    # Save the embedding to a file
    
    #print(f"Embedding shape: {embedding.shape}")
    
    return embedding22, embedding33, embedding44

def make_text(prompt, device, processor, model, fn="cmp_b0001.png"):

    raw_image = Image.open(fn)

    if device.type == "cuda":
        inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16) # cuda
    else:
        inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(device)

    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    #print(processor.decode(output[0][2:], skip_special_tokens=True))

    spec_word = "ASSISTANT: "
    input_string = processor.decode(output[0][2:], skip_special_tokens=True)
    position = input_string.find(spec_word)
    
    # If "ABCD" is found, return everything after it
    sample_text = None
    if position != -1:
        sample_text = input_string[position + len(spec_word):]  # +4 to skip "ABCD" itself
    else:
        # If "ABCD" is not found, return the original string
        pass
        
    return sample_text

if __name__ == "__main__":
    # Example two-sentence text
    sample_text = "The quick brown fox jumps over the lazy dog. Machine learning models can generate vector embeddings from text."
    
    device, model, model_id, processor, conversation, prompt = init()    
    
    path_root = "/home/roman/PycharmProjects/jobs/dandy"
    fn=f"{path_root}/pytorch-CycleGAN-and-pix2pix/datasets/edges2shoes/test_yellow/cmp_b0001.png"

    fn1 = f"{path_root}/pytorch-CycleGAN-and-pix2pix/datasets/edges2shoes/train/9999_AB.jpg"
    fn2 = f"{path_root}/pytorch-CycleGAN-and-pix2pix/datasets/edges2shoes/train/1_AB.jpg"
    fn3 = f"{path_root}/pytorch-CycleGAN-and-pix2pix/datasets/edges2shoes/train/3852_AB.jpg" # women grey
    fn4 = f"{path_root}/pytorch-CycleGAN-and-pix2pix/datasets/edges2shoes/train/3855_AB.jpg" # black dress women 
    fn5 = f"{path_root}/pytorch-CycleGAN-and-pix2pix/datasets/edges2shoes/train/4902_AB.jpg" # blue dress women 
    fn6 = f"{path_root}/pytorch-CycleGAN-and-pix2pix/datasets/edges2shoes/train/49002_AB.jpg" # leopard dress women 
    

    sample_text_list = []
    embedding_list = []
    for fn in [fn1, fn2, fn3, fn4, fn5, fn6]:
        sample_text = make_text(prompt, device, processor, model, fn=fn)
        print(sample_text)
        sample_text_list.append(sample_text)

    for sample_text in sample_text_list:
        # Create and save the embedding
        embedding, _, _ = create_and_save_embedding(
            model=model,
            processor=processor,                        
            device=device,
            text=sample_text,
            model_name="llava-hf/llava-1.5-7b-hf",
            output_file="llava_embedding", # .npy"
            debug=True
        )
        embedding_list.append(embedding)       
        print(f"sample_text={sample_text} done. Shape: {embedding.shape}")
    print(sample_text_list)
    #Example output: ['Brown, loafer, unisex.', 'Brown, dress shoe, men.', 'Gray, loafer, unisex.', 'Black, heel, women.', 'Blue, heel, women.', 'Brown, slip on, unisex.']
