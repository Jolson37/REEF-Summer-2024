import torch
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import supervision as sv
import matplotlib.pyplot as plt

import cv2
from segment_anything import SamAutomaticMaskGenerator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_type = "vit_h"

image_path = "images/wing_painted_resized.jpg"

sam_checkpoint = "sam_vit_h_4b8939.pth"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
result = mask_generator.generate(image_rgb)

new_img = result

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
 
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

print(len(result))
print(result[0].keys())
 
plt.figure(figsize=(20,20))
plt.imshow(image_rgb)
show_anns(new_img)
plt.axis('off')
plt.show()
