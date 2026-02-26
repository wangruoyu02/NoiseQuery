import os
from diffusers import StableDiffusionPipeline
import torch

model_id = "stabilityai/stable-diffusion-2-1-base"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)


save_root = '。/seed_images/sd2.1/seed100k'
os.makedirs(save_root, exist_ok=True)

for seed in range(100000):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    image = pipe(prompt='', generator=generator, guidance_scale=0).images[0]
    save_path = os.path.join(save_root, f'{seed}.jpg')
    image.save(save_path)
    del generator