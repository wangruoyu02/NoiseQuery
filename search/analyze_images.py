import os
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import time

IMAGE_DIR = '../library/seed_images/sdv2.1/seed100k'
OUTPUT_CSV = os.path.join('../search/low_level_stats','sdv2.1_image_stats.csv')
NUM_WORKERS = 80


NUM_WORKERS = 80

def image_colorfulness(image):
    (B, G, R) = cv2.split(image.astype("float"))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    return stdRoot + (0.3 * meanRoot)

def extract_features(img_path):
    filename = os.path.basename(img_path)
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        height, width, _ = img.shape
        aspect_ratio = width / height
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        brightness = np.mean(img_gray)
        contrast = np.std(img_gray)

        mean_r = np.mean(img_rgb[:, :, 0])
        mean_g = np.mean(img_rgb[:, :, 1])
        mean_b = np.mean(img_rgb[:, :, 2])
        
        saturation = np.mean(img_hsv[:, :, 1])
        
        sharpness = cv2.Laplacian(img_gray, cv2.CV_64F).var()
        
        colorfulness = image_colorfulness(img)

        return {
            'filename': filename,
            'width': width,
            'height': height,
            'aspect_ratio': aspect_ratio,
            'brightness': brightness,     
            'contrast': contrast,         
            'saturation': saturation,     
            'sharpness': sharpness,       
            'colorfulness': colorfulness, 
            'mean_r': mean_r,             
            'mean_g': mean_g,             
            'mean_b': mean_b              
        }

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def main():
    all_files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    all_files.sort() 
    
    total_images = len(all_files)

    results = []
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = list(tqdm(executor.map(extract_features, all_files), total=total_images, unit="img"))
        
        for res in futures:
            if res is not None:
                results.append(res)

    end_time = time.time()

    df = pd.DataFrame(results)
    
    cols = ['filename', 'brightness', 'contrast', 'saturation', 'sharpness', 'colorfulness', 
            'mean_r', 'mean_g', 'mean_b', 'width', 'height']
    df = df[cols]
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(df.describe())

if __name__ == '__main__':
    main()

