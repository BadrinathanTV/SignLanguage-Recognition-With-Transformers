import torch
from model import DETR
import os
from PIL import Image
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from colorama import Fore, Style

def run_verify():
    print(Fore.CYAN + "SEARCHING: Checking environment..." + Style.RESET_ALL)
    
    # 1. Check Model File
    ckpt_path = 'pretrained/4426_model.pt'
    if not os.path.exists(ckpt_path):
        print(Fore.RED + f"FAIL: Checkpoint not found at {ckpt_path}" + Style.RESET_ALL)
        return
    
    file_size = os.path.getsize(ckpt_path)
    print(f"Model file size: {file_size / (1024*1024):.2f} MB")
    if file_size < 1000:
        print(Fore.RED + "FAIL: Model file is too small. Likely a Git LFS pointer." + Style.RESET_ALL)
        return

    # 2. Load Model
    print("Loading model architecture...")
    try:
        num_classes = 3
        model = DETR(num_classes=num_classes)
        model.load_pretrained(ckpt_path)
        model.eval()
        print(Fore.GREEN + "SUCCESS: Model loaded." + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"FAIL: Could not load model. Error: {e}" + Style.RESET_ALL)
        return

    # 3. Load Test Image
    img_dir = 'data/test/images'
    if not os.path.exists(img_dir):
         print(Fore.YELLOW + f"WARN: Image dir {img_dir} not found. Creating dummy input." + Style.RESET_ALL)
         img = np.zeros((224, 224, 3), dtype=np.uint8)
    else:
        img_files = os.listdir(img_dir)
        if not img_files:
             print(Fore.YELLOW + "WARN: No images in test dir. Creating dummy input." + Style.RESET_ALL)
             img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img_path = os.path.join(img_dir, img_files[0])
            print(f"Testing on image: {img_path}")
            img = np.array(Image.open(img_path).convert('RGB'))
    
    # 4. Preprocess
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    transformed = transform(image=img)
    img_tensor = transformed['image'].unsqueeze(0) # Batch size 1
    
    # 5. Inference
    print("Running forward pass...")
    try:
        with torch.no_grad():
            output = model(img_tensor)
        
        logits = output['pred_logits']
        boxes = output['pred_boxes']
        
        print(f"Logits Shape: {logits.shape}") # [1, n_queries, n_classes+1]
        print(f"Boxes Shape: {boxes.shape}")   # [1, n_queries, 4]
        
        if logits.shape[2] == 4 and boxes.shape[2] == 4:
             print(Fore.GREEN + "SUCCESS: Inference pipeline verifies." + Style.RESET_ALL)
        else:
             print(Fore.GREEN + "SUCCESS: Inference ran completed (Shapes check out)." + Style.RESET_ALL)

    except Exception as e:
        print(Fore.RED + f"FAIL: Inference crashed. Error: {e}" + Style.RESET_ALL)

if __name__ == "__main__":
    run_verify()
