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
import sys
import pickle
from collections import defaultdict
from einops import rearrange
import numpy as np
import PIL
from PIL import Image
from torchvision.transforms import PILToTensor
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add the directory containing dift_sd.py to sys.path
sys.path.append(os.path.abspath('..'))

from dift_sd import SDFeaturizer
import lpips
import clip
import pandas as pd
from pytorch_lightning import seed_everything

def preprocess_image(image, device):
    image = torch.from_numpy(image).float() / 127.5 - 1 # [-1, 1]
    image = rearrange(image, "h w c -> 1 c h w")
    image = image.to(device)
    return image

if __name__ == '__main__':
    # Hardcoded directories for evaluation
    # eval_roots = [
    #     '../FreeDrag_experiments/drag_diffusion_res_80_0.7_0.01_3_n_step=300',
    #     '../FreeDrag_experiments/freedrag_diffusion_res_80_0.7_0.01_3_n_step=300_d_max=5.0_l_expected=1.0'
    #     #'../L1_L2_experiments/drag_diffusion_res_80_0.7_0.01_3_L1m=False_L1p=False_L1mask=False',
    #     #'../L1_L2_experiments/drag_diffusion_res_80_0.7_0.01_3_L1m=False_L1p=False_L1mask=True',
    #     #'../L1_L2_experiments/drag_diffusion_res_80_0.7_0.01_3_L1m=False_L1p=True_L1mask=False',
    #     #'../L1_L2_experiments/drag_diffusion_res_80_0.7_0.01_3_L1m=False_L1p=True_L1mask=True',
    #     #'../L1_L2_experiments/drag_diffusion_res_80_0.7_0.01_3_L1m=True_L1p=False_L1mask=False',
    #     #'../L1_L2_experiments/drag_diffusion_res_80_0.7_0.01_3_L1m=True_L1p=False_L1mask=True',
    #     #'../L1_L2_experiments/drag_diffusion_res_80_0.7_0.01_3_L1m=True_L1p=True_L1mask=False',
    #     #'../L1_L2_experiments/drag_diffusion_res_80_0.7_0.01_3_L1m=True_L1p=True_L1mask=True'
    # ]

    eval_root = '../freedrag_experiments/'

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('test')
    # using SD-2.1
    dift = SDFeaturizer('stabilityai/stable-diffusion-2-1')
    print("Model loaded successfully.")

    # lpip metric
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    # load clip model
    # clip_model, clip_preprocess = (clip.load("ViT-B/32", device=device, jit=False))
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

    for target_root in os.listdir(eval_root):
        target_root = os.path.join(eval_root, target_root)
        all_lpips = defaultdict(list)
        # all_clip_sim = defaultdict(list)
        all_dist = defaultdict(list)
        
        for cat in all_category:
            for file_name in os.listdir(os.path.join(original_img_root, cat)):
                if file_name == '.DS_Store':
                    continue
                #extract the meta data
                with open(os.path.join(original_img_root, cat, file_name, 'meta_data.pkl'), 'rb') as f:
                    meta_data = pickle.load(f)
                prompt = meta_data['prompt']
                points = meta_data['points']

                # here, the point is in x,y coordinate
                handle_points = []
                target_points = []
                for idx, point in enumerate(points):
                    # from now on, the point is in row,col coordinate
                    cur_point = torch.tensor([point[1], point[0]])
                    if idx % 2 == 0:
                        handle_points.append(cur_point)
                    else:
                        target_points.append(cur_point)

                #open the images        
                source_image_path = os.path.join(original_img_root, cat, file_name, 'original_image.png')
                dragged_image_path = os.path.join(target_root, cat, file_name, 'dragged_image.png')

                source_image_PIL = Image.open(source_image_path)
                dragged_image_PIL = Image.open(dragged_image_path)
                dragged_image_PIL = dragged_image_PIL.resize(source_image_PIL.size,PIL.Image.BILINEAR)

                #To calculate LPIP and CLIP similarity
                source_image = preprocess_image(np.array(source_image_PIL), device)
                dragged_image = preprocess_image(np.array(dragged_image_PIL), device)

                # compute LPIP
                with torch.no_grad():
                    source_image_224x224 = F.interpolate(source_image, (224,224), mode='bilinear')
                    dragged_image_224x224 = F.interpolate(dragged_image, (224,224), mode='bilinear')
                    cur_lpips = loss_fn_alex(source_image_224x224, dragged_image_224x224)
                    all_lpips[cat].append(cur_lpips.item())

                # compute CLIP similarity
                # source_image_clip = clip_preprocess(source_image_PIL).unsqueeze(0).to(device)
                # dragged_image_clip = clip_preprocess(dragged_image_PIL).unsqueeze(0).to(device)
                #
                # with torch.no_grad():
                #     source_feature = clip_model.encode_image(source_image_clip)
                #     dragged_feature = clip_model.encode_image(dragged_image_clip)
                #     source_feature /= source_feature.norm(dim=-1, keepdim=True)
                #     dragged_feature /= dragged_feature.norm(dim=-1, keepdim=True)
                #     cur_clip_sim = (source_feature * dragged_feature).sum()
                #     all_clip_sim[cat].append(cur_clip_sim.cpu().numpy())

                # Create tensors to calculate th MD    
                source_image_tensor = (PILToTensor()(source_image_PIL) / 255.0 - 0.5) * 2
                dragged_image_tensor = (PILToTensor()(dragged_image_PIL) / 255.0 - 0.5) * 2

                _, H, W = source_image_tensor.shape

                ft_source = dift.forward(source_image_tensor,
                      prompt=prompt,
                      t=261,
                      up_ft_index=1,
                      ensemble_size=8)
                ft_source = F.interpolate(ft_source, (H, W), mode='bilinear').cpu()

                ft_dragged = dift.forward(dragged_image_tensor,
                      prompt=prompt,
                      t=261,
                      up_ft_index=1,
                      ensemble_size=8)
                ft_dragged = F.interpolate(ft_dragged, (H, W), mode='bilinear').cpu()

                cos = nn.CosineSimilarity(dim=1)
                for pt_idx in range(len(handle_points)):
                    hp = handle_points[pt_idx]
                    tp = target_points[pt_idx]

                    num_channel = ft_source.size(1)
                    src_vec = ft_source[0, :, hp[0], hp[1]].view(1, num_channel, 1, 1)
                    cos_map = cos(src_vec, ft_dragged).cpu().numpy()[0]  # H, W
                    max_rc = np.unravel_index(cos_map.argmax(), cos_map.shape) # the matched row,col

                    # calculate distance
                    dist = (tp - torch.tensor(max_rc)).float().norm()
                    all_dist[cat].append(dist.item())          


        # Process LPIPS results
        for key, values in all_lpips.items():
            avg_lpips = 1 - np.mean(values)
            results_df = pd.concat([
                results_df,
                pd.DataFrame({'Category': [key], 'Metric': ['avg LPIPS'], 'Value': [avg_lpips], 'Eval_Root': [target_root]})
            ])
            print(f'\t{key}: {avg_lpips}')

        # Process CLIP similarity results
        # for key, values in all_clip_sim.items():
        #     avg_clip_sim = np.mean(values)
        #     results_df = pd.concat([
        #         results_df,
        #         pd.DataFrame({'Category': [key], 'Metric': ['avg CLIP similarity'], 'Value': [avg_clip_sim], 'Eval_Root': [target_root]})
        #     ])
        #     print(f'\t{key}: {avg_clip_sim}')

        
        # Process MD results
        for key, values in all_dist.items():
            mean_dist = np.mean(values)
            results_df = pd.concat([
                results_df,
                pd.DataFrame({'Category': [key], 'Metric': ['avg MD'], 'Value': [mean_dist], 'Eval_Root': [target_root]})
            ])
            print(f'\t{key}: {mean_dist}')

    # Save results to a CSV file
    results_df.to_csv('metrics_freedrag_experiments_1.csv', index=False)
    print("\nResults saved to 'metrics_freedrag_experiments_1.csv'")
