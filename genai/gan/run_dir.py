import os

# Execute the function with the specified directory
from img_to_txt import init, make_text, create_and_save_embedding

device, model, model_id, processor, conversation, prompt = init(is_cuda=False)    
    
def create_text_files_for_pngs(directory, is_png_or_jpg, new_color=None):
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return
    
    # Counter for processed files
    processed_count = 1
    
    # Walk through directory and all subdirectories
    for root, dirs, files in os.walk(directory):
        # Filter for PNG files in current directory
        if root[-1] == 'b':
            continue
        if is_png_or_jpg:
            png_files = [f for f in files if f.lower().endswith('.png')]
        else:
            png_files = [f for f in files if f.lower().endswith('.jpg')]

        # Process each PNG file
        for png_file in png_files:
            # Get the full path of the PNG file
            png_path = os.path.join(root, png_file)
            
            # Get the base filename without extension
            base_name = os.path.splitext(png_file)[0]
            
            # Create the corresponding TXT filename
            txt_file = f"{base_name}.txt"
            enb_file = f"{base_name}_llava_emb44.npy"
            txt_path = os.path.join(root,  txt_file)

            emb_path = os.path.join(root,  f"{base_name}_llava_emb")
            emb_path_44 = f"{emb_path}_44.npy"
            
            print(f"\n----------\n{txt_path}, {base_name}, {emb_path_44}")
            processed_count += 1
            
            if (os.path.isfile(txt_path) or os.path.exists(txt_file)): #  and (os.path.isfile(emb_path_44) or os.path.exists(emb_path_44)):
                # create embeddings for existing txt files
                try:
                    with open(txt_path, 'r', encoding='utf-8') as file:
                        sample_text = file.read()
                except Exception as e:
                    raise Exception(f"Error reading file: {txt_path}")      
                print(f"{txt_path}={sample_text}")    
                if new_color is not None:          
                    words = sample_text.split(',')
                    assert words
                    words[0] = new_color
                sample_text = ','.join(words)
                print(f"{txt_path}={sample_text}")    
                 
            else:
                # create txt file and embeddings 
                sample_text = make_text(prompt, device, processor, model, fn=png_path) 
                print(f"\n----------\n{sample_text}")        
                # Write to the TXT file
                with open(txt_path, 'w') as f:
                    f.write(f"{sample_text}")
            # Create and save the embedding
            emb22, emb33, emb44 = create_and_save_embedding(
                model=model,
                processor=processor,
                device=device,                        
                text=sample_text,
                model_name="llava-hf/llava-1.5-7b-hf",
                output_file=emb_path, # .npy"
            )            
            print(f"Created {processed_count}: {sample_text} ({emb22.shape}, {emb33.shape}, {emb44.shape})")
            
    
    if processed_count > 0:
        print(f"Processed {processed_count} PNG files across all subdirectories.")
    else:
        print(f"No PNG files found in '{directory}' or its subdirectories.")


if __name__ == "__main__":
    path_root = "/home/roman/PycharmProjects/jobs/dandy"
    edge_dir = f"{path_root}/pytorch-CycleGAN-and-pix2pix/datasets/edges2shoes"
    directory = f"{edge_dir}/test_yellow"    
    create_text_files_for_pngs(directory, is_png_or_jpg=False, new_color='Yellow')
    directory = f"{edge_dir}/test_red"    
    create_text_files_for_pngs(directory, is_png_or_jpg=False, new_color='Red')