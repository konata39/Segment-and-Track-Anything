from PIL.ImageOps import colorize, scale
import gradio as gr
import importlib
import sys
import os
import pdb
from matplotlib.pyplot import step

from model_args import segtracker_args,sam_args,aot_args
from SegTracker import SegTracker
from tool.transfer_tools import draw_outline, draw_points
# sys.path.append('.')
# sys.path.append('..')

import cv2
from PIL import Image
from skimage.morphology.binary import binary_dilation
import argparse
import torch
import time, math
from seg_track_anything import aot_model2ckpt, tracking_objects_in_video, draw_mask
import gc
import numpy as np
import json
from tool.transfer_tools import mask2bbox

video_name = 'blackswan'
input_video = f'./assets/new/{video_name}.mp4'
tracking_mask_path = f'./tracking_results/{video_name}'
f = open(f'{tracking_mask_path}/{video_name}_json_frames/mask.json')

data = json.load(f)

f.close()

color_label = []
color_count = []
json_dict = {}
color_dict = []
frame_count = 0
width = data['width']
height = data['height']
total_frame = len(data['color_dict'])
pred_mask = []
for idx in range(len(data['color_dict'][0]['color_label'])):
    d = data['color_dict'][0]
    pred_mask += ([d['color_label'][idx]] * d['color_count'][idx])

pred_mask = np.array(pred_mask)
pred_mask = np.reshape(pred_mask, (-1, width))

cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
if cap.isOpened():
    ret, origin_frame = cap.read()
cap.release()

aot_args["model"] = 'r50_deaotl'
aot_args["model_path"] = aot_model2ckpt["r50_deaotl"]
aot_args["long_term_mem_gap"] = 9999
aot_args["max_len_long_term"] = 9999
# reset sam args
segtracker_args["sam_gap"] = 9999
segtracker_args["max_obj_num"] = 6
sam_args["generator_args"]["points_per_side"] = 16

Seg_Tracker = SegTracker(segtracker_args, sam_args, aot_args)
with torch.cuda.amp.autocast():
    Seg_Tracker.restart_tracker()
    torch.cuda.empty_cache()
    Seg_Tracker.add_reference(origin_frame, pred_mask, 0)
    Seg_Tracker.first_frame_mask = pred_mask
    print(Seg_Tracker)

    #return Seg_Tracker, origin_frame, [[], []], ""
    tracking_objects_in_video(Seg_Tracker, input_video, None, fps, 0, False)
