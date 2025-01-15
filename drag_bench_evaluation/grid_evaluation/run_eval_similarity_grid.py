# *************************************************************************
# Copyright (2023) Bytedance Inc.
#
# Copyright (2023) DragDiffusion Authors 
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 
#
#     http://www.apache.org/licenses/LICENSE-2.0 
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
# *************************************************************************

# evaluate similarity between images before and after dragging
import argparse
import os
from collections import defaultdict
from einops import rearrange
import numpy as np
import PIL
from PIL import Image
import torch
import torch.nn.functional as F
import lpips
import clip
import pandas as pd

def preprocess_image(image, device):
    image = torch.from_numpy(image).float() / 127.5 - 1 # [-1, 1]
    image = rearrange(image, "h w c -> 1 c h w")
    image = image.to(device)
    return image

if __name__ == '__main__':
    # Hardcoded directories for evaluation
    eval_roots = [
        '../L1_L2_experiments/drag_diffusion_res_80_0.7_0.01_3_L1m=False_L1p=False_L1mask=False',
        '../L1_L2_experiments/drag_diffusion_res_80_0.7_0.01_3_L1m=False_L1p=False_L1mask=True',
        '../L1_L2_experiments/drag_diffusion_res_80_0.7_0.01_3_L1m=False_L1p=True_L1mask=False',
        '../L1_L2_experiments/drag_diffusion_res_80_0.7_0.01_3_L1m=False_L1p=True_L1mask=True',
        '../L1_L2_experiments/drag_diffusion_res_80_0.7_0.01_3_L1m=True_L1p=False_L1mask=False',
        '../L1_L2_experiments/drag_diffusion_res_80_0.7_0.01_3_L1m=True_L1p=False_L1mask=True',
        '../L1_L2_experiments/drag_diffusion_res_80_0.7_0.01_3_L1m=True_L1p=True_L1mask=False',
        '../L1_L2_experiments/drag_diffusion_res_80_0.7_0.01_3_L1m=True_L1p=True_L1mask=True'
    ]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # lpip metric
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    # load clip model
    clip_model, clip_preprocess = (clip.load("ViT-B/32", device=device, jit=False))
    # clip_model, clip_preprocess = (clip.lo ("ViT-B/32", device=device, jit=False))

    all_category = [
        'art_work',
        'land_scape',
        'building_city_view',
        'building_countryside_view',
        'animals',
        'human_head',
        'human_upper_body',
        'human_full_body',
        'interior_design',
        'other_objects',
    ]

    original_img_root = '../drag_bench_data/'

    # DataFrame to store results
    results_df = pd.DataFrame(columns=['Category', 'Metric', 'Value', 'Eval_Root'])

    for target_root in eval_roots:
        all_lpips = defaultdict(list)
        all_clip_sim = defaultdict(list)
        
        for cat in all_category:
            for file_name in os.listdir(os.path.join(original_img_root, cat)):
                if file_name == '.DS_Store':
                    continue
                source_image_path = os.path.join(original_img_root, cat, file_name, 'original_image.png')
                dragged_image_path = os.path.join(target_root, cat, file_name, 'dragged_image.png')

                source_image_PIL = Image.open(source_image_path)
                dragged_image_PIL = Image.open(dragged_image_path)
                dragged_image_PIL = dragged_image_PIL.resize(source_image_PIL.size,PIL.Image.BILINEAR)

                source_image = preprocess_image(np.array(source_image_PIL), device)
                dragged_image = preprocess_image(np.array(dragged_image_PIL), device)

                # compute LPIP
                with torch.no_grad():
                    source_image_224x224 = F.interpolate(source_image, (224,224), mode='bilinear')
                    dragged_image_224x224 = F.interpolate(dragged_image, (224,224), mode='bilinear')
                    cur_lpips = loss_fn_alex(source_image_224x224, dragged_image_224x224)
                    all_lpips[cat].append(cur_lpips.item())

                # compute CLIP similarity
                source_image_clip = clip_preprocess(source_image_PIL).unsqueeze(0).to(device)
                dragged_image_clip = clip_preprocess(dragged_image_PIL).unsqueeze(0).to(device)

                with torch.no_grad():
                    source_feature = clip_model.encode_image(source_image_clip)
                    dragged_feature = clip_model.encode_image(dragged_image_clip)
                    source_feature /= source_feature.norm(dim=-1, keepdim=True)
                    dragged_feature /= dragged_feature.norm(dim=-1, keepdim=True)
                    cur_clip_sim = (source_feature * dragged_feature).sum()
                    all_clip_sim[cat].append(cur_clip_sim.cpu().numpy())
            # print(f"all lpips: {all_lpips}")
            # print(f"all clips sim: {all_clip_sim}")

        # Process LPIPS results
        for key, values in all_lpips.items():
            avg_lpips = 1 - np.mean(values)
            results_df = pd.concat([
                results_df,
                pd.DataFrame({'Category': [key], 'Metric': ['avg LPIPS'], 'Value': [avg_lpips], 'Eval_Root': [target_root]})
            ])
            print(f'\t{key}: {avg_lpips}')

        # Process CLIP similarity results
        for key, values in all_clip_sim.items():
            avg_clip_sim = np.mean(values)
            results_df = pd.concat([
                results_df,
                pd.DataFrame({'Category': [key], 'Metric': ['avg CLIP similarity'], 'Value': [avg_clip_sim], 'Eval_Root': [target_root]})
            ])
            print(f'\t{key}: {avg_clip_sim}')

    # Save results to a CSV file
    results_df.to_csv('evaluation_results.csv', index=False)
    print("\nResults saved to 'evaluation_results.csv'")
