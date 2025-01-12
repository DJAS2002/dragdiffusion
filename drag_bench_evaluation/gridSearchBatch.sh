#!/bin/bash
python3 run_drag_diffusion.py --is_l1_supervision_loss true --is_l1_point_tracking true --is_l1_mask true
python3 run_drag_diffusion.py --is_l1_supervision_loss true --is_l1_point_tracking true --is_l1_mask false
python3 run_drag_diffusion.py --is_l1_supervision_loss true --is_l1_point_tracking false --is_l1_mask true
python3 run_drag_diffusion.py --is_l1_supervision_loss true --is_l1_point_tracking false --is_l1_mask false
python3 run_drag_diffusion.py --is_l1_supervision_loss false --is_l1_point_tracking true --is_l1_mask true
python3 run_drag_diffusion.py --is_l1_supervision_loss false --is_l1_point_tracking true --is_l1_mask false
python3 run_drag_diffusion.py --is_l1_supervision_loss false --is_l1_point_tracking false --is_l1_mask true
python3 run_drag_diffusion.py --is_l1_supervision_loss false --is_l1_point_tracking false --is_l1_mask false