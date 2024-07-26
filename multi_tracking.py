import sys
import time
import threading
import json
import tkinter as tk
from tkinter import filedialog
from seg_track_anything import aot_model2ckpt, tracking_objects_in_video, draw_mask
from model_args import segtracker_args,sam_args,aot_args
from SegTracker import SegTracker
import numpy as np
from PIL import Image, ImageTk
import cv2
import torch
import os
from skimage import measure

def thread_tracking(Seg_Tracker, video_path, video_fps, start_frame, detect_check, reversed, end_check, end_frame):
    return tracking_objects_in_video(Seg_Tracker, video_path, None, video_fps, start_frame, end_frame, False, False, detect_check, reversed, multi_thread=True)

def SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask):
    with torch.cuda.amp.autocast():
        # Reset the first frame's mask
        frame_idx = 0
        #Seg_Tracker.restart_tracker()
        Seg_Tracker.add_reference(origin_frame, predicted_mask, frame_idx)
        Seg_Tracker.first_frame_mask = predicted_mask

    return Seg_Tracker

def seg_acc_click(Seg_Tracker, prompt, origin_frame):
    # seg acc to click
    predicted_mask, masked_frame = Seg_Tracker.seg_acc_click(
                                                      origin_frame=origin_frame,
                                                      coords=np.array(prompt["points_coord"]),
                                                      modes=np.array(prompt["points_mode"]),
                                                      multimask=prompt["multimask"],
                                                    )

    Seg_Tracker = SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask)

    return masked_frame, Seg_Tracker

def rebuild_mask(color_label, color_count, width, height):
    mask = np.zeros((height, width), dtype=int)
    current_pos = 0
    for label, count in zip(color_label, color_count):
        for _ in range(count):
            row = current_pos // width
            col = current_pos % width
            mask[row, col] = label
            current_pos += 1
    return mask

def get_click_prompt(stack):

    prompt = {
        "points_coord":stack[0],
        "points_mode":stack[1],
        "multimask":"True",
    }

    return prompt

if __name__ == '__main__':
    path_name = sys.argv[1]
    if '\\' in path_name and '/' in path_name:
        video_name = path_name.split('/')[-1].split('\\')[-1]
    else:
        video_name = path_name.split('/')[-1] if '/' in path_name else path_name.split('\\')[-1]
    now_path = os.path.dirname(os.path.abspath(__file__))
    file_path = sys.argv[2]
    print(file_path)
    json_dict = dict()
    try:
        with open(file_path) as f:
            json_dict = json.load(f)
    except:
        print("Error: video individual json not found.")
        exit()

    #SAT Parameters
    aot_args["model"] = 'r50_deaotl'
    aot_args["model_path"] = aot_model2ckpt["r50_deaotl"]
    aot_args["long_term_mem_gap"] = 9999
    aot_args["max_len_long_term"] = 9999
    # reset sam args
    segtracker_args["sam_gap"] = 9999
    segtracker_args["max_obj_num"] = 6
    sam_args["generator_args"]["points_per_side"] = 16
    Seg_Tracker = SegTracker(segtracker_args, sam_args, aot_args)


    cap = cv2.VideoCapture(path_name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    json_output_list = []

    start_index = int(sys.argv[3])
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_index)
    ret, captured_frame = cap.read()
    tracking_dict = dict()
    with open(file_path) as f:
        tracking_dict = json.load(f)
    click_prompt_path = file_path.replace('tracking_point.json', 'click_prompt.json')
    click_prompt_dict = dict()
    with open(click_prompt_path) as f:
        click_prompt_dict = json.load(f)
    for i in click_prompt_dict:
        if len(click_prompt_dict[i]) > segtracker_args["max_obj_num"]:
            segtracker_args["max_obj_num"] = len(click_prompt_dict[i])
    for idx in click_prompt_dict[str(start_index)]:
        if len(click_prompt_dict[str(start_index)][idx][1]) == 0:
            continue
        Seg_Tracker.curr_idx = int(idx)
        click_prompt = {
                "points_coord":click_prompt_dict[str(start_index)][idx][0],
                "points_mode":click_prompt_dict[str(start_index)][idx][1],
                "multimask":"True",
        }
        masked_frame = seg_acc_click(Seg_Tracker, click_prompt, captured_frame)
        prev_mask = Seg_Tracker.first_frame_mask
        Seg_Tracker.update_origin_merged_mask(prev_mask)
        #Seg_Tracker.update_origin_merged_mask(prev_mask)
    #for idx in json_dict['video_mask'][start_index]:
    #    i = json_dict['video_mask'][start_index][idx]
    #    if len(i['point']) == 0:
    #        continue
    #    Seg_Tracker.curr_idx = int(idx) - 1
    #    click_prompt = {
    #        "points_coord":i['point'],
    #        "points_mode":i['mode'],
    #        "multimask":"True",
    #    }
    #    masked_frame = seg_acc_click(Seg_Tracker, click_prompt, captured_frame)
    #    prev_mask = Seg_Tracker.first_frame_mask
    #    Seg_Tracker.update_origin_merged_mask(prev_mask)

    start_f = start_index
    end_f = int(sys.argv[4])
    output_json = thread_tracking(Seg_Tracker, path_name, video_fps, start_f, False, False, True, end_f)
    output_json_dict = dict()
    output_json_dict['range'] = [start_f,end_f]
    output_json_dict['color_dict'] = output_json['color_dict']
    json_output_list.append(output_json_dict)

    output_dict = dict()
    output_dict['width'] = width
    output_dict['height'] = height
    output_dict['object_num'] = int(segtracker_args["max_obj_num"])
    output_dict['start'] = start_f
    output_dict['end'] = end_f
    color_dict = []
    for i in range(total_frames):
        color_dict.append({'color_label':[0], 'color_count':[width*height], "label_sum": {}})
    output_dict['color_dict'] = color_dict
    for data in json_output_list:
        for idx in range(data['range'][0], data['range'][1]):
            output_dict['color_dict'][idx]['color_label'] = data['color_dict'][idx]['color_label']
            output_dict['color_dict'][idx]['color_count'] = data['color_dict'][idx]['color_count']
            label_sum_dict = dict()
            for data_idx in range(len(data['color_dict'][idx]['color_label'])):
                if data['color_dict'][idx]['color_label'][data_idx] != 0:
                    label_idx = data['color_dict'][idx]['color_label'][data_idx]
                    if str(label_idx) not in label_sum_dict:
                        label_sum_dict[str(label_idx)] = data['color_dict'][idx]['color_count'][data_idx]
                    else:
                        label_sum_dict[str(label_idx)] += data['color_dict'][idx]['color_count'][data_idx]
            output_dict['color_dict'][idx]["label_sum"] = label_sum_dict
    #print(output_dict['color_dict'][:200])
    output_path = file_path.replace('tracking_point.json', '')
    if not os.path.exists(f"{output_path}/multithread_output"):
        os.makedirs(f"{output_path}/multithread_output")
    with open(f"{output_path}/multithread_output/mask_{start_index}.json", 'w') as f:
        json.dump(output_dict, f)
        #masked_frame = seg_acc_click(Seg_Tracker, click_prompt, captured_frame)
    #total_args = len(sys.argv) - 1
    #threads = []
    #for i in range(total_args):
    #    x = int(sys.argv[i+1])
    #    threads.append(threading.Thread(target = job, args = (x,)))
    #    threads[i].start()
    #for i in range(5):
    #    threads[i].join()
