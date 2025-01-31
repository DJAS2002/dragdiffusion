#!/bin/bash

### Aggregation dim = 0
# ablation on d_max = 1.5 and l_expected = 0.15 with aggregation dim 0
python3 run_freedrag_diffusion.py --reduce_dims 0 --l_expected 0.15 --d_max 1.5 --is_l1_supervision_loss True --is_l1_point_tracking True --is_l1_mask True
python3 run_freedrag_diffusion.py --reduce_dims 0 --l_expected 0.15 --d_max 1.5 --is_l1_supervision_loss True --is_l1_point_tracking True --is_l1_mask False
python3 run_freedrag_diffusion.py --reduce_dims 0 --l_expected 0.15 --d_max 1.5 --is_l1_supervision_loss True --is_l1_point_tracking False --is_l1_mask True
python3 run_freedrag_diffusion.py --reduce_dims 0 --l_expected 0.15 --d_max 1.5 --is_l1_supervision_loss True --is_l1_point_tracking False --is_l1_mask False
python3 run_freedrag_diffusion.py --reduce_dims 0 --l_expected 0.15 --d_max 1.5 --is_l1_supervision_loss False --is_l1_point_tracking True --is_l1_mask True
python3 run_freedrag_diffusion.py --reduce_dims 0 --l_expected 0.15 --d_max 1.5 --is_l1_supervision_loss False --is_l1_point_tracking True --is_l1_mask False
python3 run_freedrag_diffusion.py --reduce_dims 0 --l_expected 0.15 --d_max 1.5 --is_l1_supervision_loss False --is_l1_point_tracking False --is_l1_mask True
python3 run_freedrag_diffusion.py --reduce_dims 0 --l_expected 0.15 --d_max 1.5 --is_l1_supervision_loss False --is_l1_point_tracking False --is_l1_mask False
#
## ablation on d_max = 3 and l_expected = 0.3 with aggregation dim 0
#python3 run_freedrag_diffusion.py --reduce_dims 0 --l_expected 0.3 --d_max 3.0 --is_l1_supervision_loss true --is_l1_point_tracking true --is_l1_mask true
#python3 run_freedrag_diffusion.py --reduce_dims 0 --l_expected 0.3 --d_max 3.0 --is_l1_supervision_loss true --is_l1_point_tracking true --is_l1_mask false
#python3 run_freedrag_diffusion.py --reduce_dims 0 --l_expected 0.3 --d_max 3.0 --is_l1_supervision_loss true --is_l1_point_tracking false --is_l1_mask true
#python3 run_freedrag_diffusion.py --reduce_dims 0 --l_expected 0.3 --d_max 3.0 --is_l1_supervision_loss true --is_l1_point_tracking false --is_l1_mask false
#python3 run_freedrag_diffusion.py --reduce_dims 0 --l_expected 0.3 --d_max 3.0 --is_l1_supervision_loss false --is_l1_point_tracking true --is_l1_mask true
#python3 run_freedrag_diffusion.py --reduce_dims 0 --l_expected 0.3 --d_max 3.0 --is_l1_supervision_loss false --is_l1_point_tracking true --is_l1_mask false
#python3 run_freedrag_diffusion.py --reduce_dims 0 --l_expected 0.3 --d_max 3.0 --is_l1_supervision_loss false --is_l1_point_tracking false --is_l1_mask true
#python3 run_freedrag_diffusion.py --reduce_dims 0 --l_expected 0.3 --d_max 3.0 --is_l1_supervision_loss false --is_l1_point_tracking false --is_l1_mask false
#
## ablation on d_max = 4.5 and l_expected = 0.45 with aggregation dim 0
#python3 run_freedrag_diffusion.py --reduce_dims 0 --l_expected 0.45 --d_max 4.5 --is_l1_supervision_loss true --is_l1_point_tracking true --is_l1_mask true
#python3 run_freedrag_diffusion.py --reduce_dims 0 --l_expected 0.45 --d_max 4.5 --is_l1_supervision_loss true --is_l1_point_tracking true --is_l1_mask false
#python3 run_freedrag_diffusion.py --reduce_dims 0 --l_expected 0.45 --d_max 4.5 --is_l1_supervision_loss true --is_l1_point_tracking false --is_l1_mask true
#python3 run_freedrag_diffusion.py --reduce_dims 0 --l_expected 0.45 --d_max 4.5 --is_l1_supervision_loss true --is_l1_point_tracking false --is_l1_mask false
#python3 run_freedrag_diffusion.py --reduce_dims 0 --l_expected 0.45 --d_max 4.5 --is_l1_supervision_loss false --is_l1_point_tracking true --is_l1_mask true
#python3 run_freedrag_diffusion.py --reduce_dims 0 --l_expected 0.45 --d_max 4.5 --is_l1_supervision_loss false --is_l1_point_tracking true --is_l1_mask false
#python3 run_freedrag_diffusion.py --reduce_dims 0 --l_expected 0.45 --d_max 4.5 --is_l1_supervision_loss false --is_l1_point_tracking false --is_l1_mask true
#python3 run_freedrag_diffusion.py --reduce_dims 0 --l_expected 0.45 --d_max 4.5 --is_l1_supervision_loss false --is_l1_point_tracking false --is_l1_mask false
#
#### Aggregation dim = 1
## ablation on d_max = 1.5 and l_expected = 0.15 with aggregation dim 1
#python3 run_freedrag_diffusion.py --reduce_dims 1 --l_expected 0.15 --d_max 1.5 --is_l1_supervision_loss true --is_l1_point_tracking true --is_l1_mask true
#python3 run_freedrag_diffusion.py --reduce_dims 1 --l_expected 0.15 --d_max 1.5 --is_l1_supervision_loss true --is_l1_point_tracking true --is_l1_mask false
#python3 run_freedrag_diffusion.py --reduce_dims 1 --l_expected 0.15 --d_max 1.5 --is_l1_supervision_loss true --is_l1_point_tracking false --is_l1_mask true
#python3 run_freedrag_diffusion.py --reduce_dims 1 --l_expected 0.15 --d_max 1.5 --is_l1_supervision_loss true --is_l1_point_tracking false --is_l1_mask false
#python3 run_freedrag_diffusion.py --reduce_dims 1 --l_expected 0.15 --d_max 1.5 --is_l1_supervision_loss false --is_l1_point_tracking true --is_l1_mask true
#python3 run_freedrag_diffusion.py --reduce_dims 1 --l_expected 0.15 --d_max 1.5 --is_l1_supervision_loss false --is_l1_point_tracking true --is_l1_mask false
#python3 run_freedrag_diffusion.py --reduce_dims 1 --l_expected 0.15 --d_max 1.5 --is_l1_supervision_loss false --is_l1_point_tracking false --is_l1_mask true
#python3 run_freedrag_diffusion.py --reduce_dims 1 --l_expected 0.15 --d_max 1.5 --is_l1_supervision_loss false --is_l1_point_tracking false --is_l1_mask false
#
## ablation on d_max = 3 and l_expected = 0.3 with aggregation dim 1
#python3 run_freedrag_diffusion.py --reduce_dims 1 --l_expected 0.3 --d_max 3.0 --is_l1_supervision_loss true --is_l1_point_tracking true --is_l1_mask true
#python3 run_freedrag_diffusion.py --reduce_dims 1 --l_expected 0.3 --d_max 3.0 --is_l1_supervision_loss true --is_l1_point_tracking true --is_l1_mask false
#python3 run_freedrag_diffusion.py --reduce_dims 1 --l_expected 0.3 --d_max 3.0 --is_l1_supervision_loss true --is_l1_point_tracking false --is_l1_mask true
#python3 run_freedrag_diffusion.py --reduce_dims 1 --l_expected 0.3 --d_max 3.0 --is_l1_supervision_loss true --is_l1_point_tracking false --is_l1_mask false
#python3 run_freedrag_diffusion.py --reduce_dims 1 --l_expected 0.3 --d_max 3.0 --is_l1_supervision_loss false --is_l1_point_tracking true --is_l1_mask true
#python3 run_freedrag_diffusion.py --reduce_dims 1 --l_expected 0.3 --d_max 3.0 --is_l1_supervision_loss false --is_l1_point_tracking true --is_l1_mask false
#python3 run_freedrag_diffusion.py --reduce_dims 1 --l_expected 0.3 --d_max 3.0 --is_l1_supervision_loss false --is_l1_point_tracking false --is_l1_mask true
#python3 run_freedrag_diffusion.py --reduce_dims 1 --l_expected 0.3 --d_max 3.0 --is_l1_supervision_loss false --is_l1_point_tracking false --is_l1_mask false
#
## ablation on d_max = 4.5 and l_expected = 0.45 with aggregation dim 1
#python3 run_freedrag_diffusion.py --reduce_dims 1 --l_expected 0.45 --d_max 4.5 --is_l1_supervision_loss true --is_l1_point_tracking true --is_l1_mask true
#python3 run_freedrag_diffusion.py --reduce_dims 1 --l_expected 0.45 --d_max 4.5 --is_l1_supervision_loss true --is_l1_point_tracking true --is_l1_mask false
#python3 run_freedrag_diffusion.py --reduce_dims 1 --l_expected 0.45 --d_max 4.5 --is_l1_supervision_loss true --is_l1_point_tracking false --is_l1_mask true
#python3 run_freedrag_diffusion.py --reduce_dims 1 --l_expected 0.45 --d_max 4.5 --is_l1_supervision_loss true --is_l1_point_tracking false --is_l1_mask false
#python3 run_freedrag_diffusion.py --reduce_dims 1 --l_expected 0.45 --d_max 4.5 --is_l1_supervision_loss false --is_l1_point_tracking true --is_l1_mask true
#python3 run_freedrag_diffusion.py --reduce_dims 1 --l_expected 0.45 --d_max 4.5 --is_l1_supervision_loss false --is_l1_point_tracking true --is_l1_mask false
#python3 run_freedrag_diffusion.py --reduce_dims 1 --l_expected 0.45 --d_max 4.5 --is_l1_supervision_loss false --is_l1_point_tracking false --is_l1_mask true
#python3 run_freedrag_diffusion.py --reduce_dims 1 --l_expected 0.45 --d_max 4.5 --is_l1_supervision_loss false --is_l1_point_tracking false --is_l1_mask false
#
#### Aggregation dim = (0, 1)
## ablation on d_max = 1.5 and l_expected = 0.15 with aggregation dim (0, 1)
#python3 run_freedrag_diffusion.py --reduce_dims 0 1 --l_expected 0.15 --d_max 1.5 --is_l1_supervision_loss true --is_l1_point_tracking true --is_l1_mask true
#python3 run_freedrag_diffusion.py --reduce_dims 0 1 --l_expected 0.15 --d_max 1.5 --is_l1_supervision_loss true --is_l1_point_tracking true --is_l1_mask false
#python3 run_freedrag_diffusion.py --reduce_dims 0 1 --l_expected 0.15 --d_max 1.5 --is_l1_supervision_loss true --is_l1_point_tracking false --is_l1_mask true
#python3 run_freedrag_diffusion.py --reduce_dims 0 1 --l_expected 0.15 --d_max 1.5 --is_l1_supervision_loss true --is_l1_point_tracking false --is_l1_mask false
#python3 run_freedrag_diffusion.py --reduce_dims 0 1 --l_expected 0.15 --d_max 1.5 --is_l1_supervision_loss false --is_l1_point_tracking true --is_l1_mask true
#python3 run_freedrag_diffusion.py --reduce_dims 0 1 --l_expected 0.15 --d_max 1.5 --is_l1_supervision_loss false --is_l1_point_tracking true --is_l1_mask false
#python3 run_freedrag_diffusion.py --reduce_dims 0 1 --l_expected 0.15 --d_max 1.5 --is_l1_supervision_loss false --is_l1_point_tracking false --is_l1_mask true
#python3 run_freedrag_diffusion.py --reduce_dims 0 1 --l_expected 0.15 --d_max 1.5 --is_l1_supervision_loss false --is_l1_point_tracking false --is_l1_mask false
#
## ablation on d_max = 3 and l_expected = 0.3 with aggregation dim 0
#python3 run_freedrag_diffusion.py --reduce_dims 0 1 --l_expected 0.3 --d_max 3.0 --is_l1_supervision_loss true --is_l1_point_tracking true --is_l1_mask true
#python3 run_freedrag_diffusion.py --reduce_dims 0 1 --l_expected 0.3 --d_max 3.0 --is_l1_supervision_loss true --is_l1_point_tracking true --is_l1_mask false
#python3 run_freedrag_diffusion.py --reduce_dims 0 1 --l_expected 0.3 --d_max 3.0 --is_l1_supervision_loss true --is_l1_point_tracking false --is_l1_mask true
#python3 run_freedrag_diffusion.py --reduce_dims 0 1 --l_expected 0.3 --d_max 3.0 --is_l1_supervision_loss true --is_l1_point_tracking false --is_l1_mask false
#python3 run_freedrag_diffusion.py --reduce_dims 0 1 --l_expected 0.3 --d_max 3.0 --is_l1_supervision_loss false --is_l1_point_tracking true --is_l1_mask true
#python3 run_freedrag_diffusion.py --reduce_dims 0 1 --l_expected 0.3 --d_max 3.0 --is_l1_supervision_loss false --is_l1_point_tracking true --is_l1_mask false
#python3 run_freedrag_diffusion.py --reduce_dims 0 1 --l_expected 0.3 --d_max 3.0 --is_l1_supervision_loss false --is_l1_point_tracking false --is_l1_mask true
#python3 run_freedrag_diffusion.py --reduce_dims 0 1 --l_expected 0.3 --d_max 3.0 --is_l1_supervision_loss false --is_l1_point_tracking false --is_l1_mask false
#
## ablation on d_max = 4.5 and l_expected = 0.45 with aggregation dim 0
#python3 run_freedrag_diffusion.py --reduce_dims 0 1 --l_expected 0.45 --d_max 4.5 --is_l1_supervision_loss true --is_l1_point_tracking true --is_l1_mask true
#python3 run_freedrag_diffusion.py --reduce_dims 0 1 --l_expected 0.45 --d_max 4.5 --is_l1_supervision_loss true --is_l1_point_tracking true --is_l1_mask false
#python3 run_freedrag_diffusion.py --reduce_dims 0 1 --l_expected 0.45 --d_max 4.5 --is_l1_supervision_loss true --is_l1_point_tracking false --is_l1_mask true
#python3 run_freedrag_diffusion.py --reduce_dims 0 1 --l_expected 0.45 --d_max 4.5 --is_l1_supervision_loss true --is_l1_point_tracking false --is_l1_mask false
#python3 run_freedrag_diffusion.py --reduce_dims 0 1 --l_expected 0.45 --d_max 4.5 --is_l1_supervision_loss false --is_l1_point_tracking true --is_l1_mask true
#python3 run_freedrag_diffusion.py --reduce_dims 0 1 --l_expected 0.45 --d_max 4.5 --is_l1_supervision_loss false --is_l1_point_tracking true --is_l1_mask false
#python3 run_freedrag_diffusion.py --reduce_dims 0 1 --l_expected 0.45 --d_max 4.5 --is_l1_supervision_loss false --is_l1_point_tracking false --is_l1_mask true
#python3 run_freedrag_diffusion.py --reduce_dims 0 1 --l_expected 0.45 --d_max 4.5 --is_l1_supervision_loss false --is_l1_point_tracking false --is_l1_mask false
