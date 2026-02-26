import os
import torch
import clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F


image_dir = '../library/seed_images/sdv2.1/seed100k'

output_path = f'/root/autodl-fs/CODE/HHY/workspace/noisequery/search/outputs/drawbench/sdv2.1_clip-h14.txt'

os.makedirs(os.path.dirname(output_path), exist_ok=True)

ft_save_path = "../library/features/sdv2.1_100k/clip_vit_H14.pt"


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()



class ImageDataset(Dataset):
    def __init__(self, image_dir, preprocess):
        self.image_dir = image_dir
        self.image_files = sorted(os.listdir(image_dir), key=lambda x: int(x.split('.')[0]))  
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path)
        return self.preprocess(image)

batch_size = 128  
dataset = ImageDataset(image_dir, preprocess)
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

def extract_image_features(model, dataloader):
    all_image_features = []
    with torch.no_grad():
        for images in tqdm(dataloader, desc="Extracting image features"):
            images = images.to(device)
            features = model.encode_image(images).cpu() 
            all_image_features.append(features)
    return torch.cat(all_image_features)

image_features = extract_image_features(model, dataloader)
print(f"Extracted image features shape: {image_features.shape}")
ft_save_path = "../library/features/sdv2.1_100k/clip_vit_B32.pt"
os.makedirs(os.path.dirname(ft_save_path), exist_ok=True)
torch.save(image_features, ft_save_path) 


image_features = torch.load(ft_save_path).to(device)



def find_most_similar_image_for_prompts(prompts, image_features, output_path, find_max=True):
    with open(output_path, 'w') as output_file:  
        for prompt in tqdm(prompts, desc="Matching prompts to images"):
            with torch.no_grad():
                prompt_features = model.encode_text(clip.tokenize([prompt],truncate=True).to(device))

            similarities = F.cosine_similarity(prompt_features, image_features)

            if find_max:
                similar_image_idx = similarities.argmax().item()  
            else:
                similar_image_idx = similarities.argmin().item()  

            output_file.write(f"{similar_image_idx}\n")


prompt_path = '/workspace/data/DrawBench.txt'
with open(prompt_path, 'r') as file:
    prompts = file.readlines()
prompts = [prompt.strip() for prompt in prompts]
print(f'Number of prompts: {len(prompts)}')
find_most_similar_image_for_prompts(prompts, image_features, output_path, find_max=True)