import json
import os
import cv2
import glob
import sys
from tkinter import filedialog
from tkinter import *

multi_file_path = sys.argv[1]
click_point_path = multi_file_path.replace('multithread_output', 'click_prompt.json')
tracking_point_path = click_point_path.replace('click_prompt.json', 'tracking_point.json')
output_segmant_tracking_path = click_point_path.replace('click_prompt.json', 'segmant_tracking.json')
json_file_list = glob.glob(f'{multi_file_path}/*')
json_list = []
for i in json_file_list:
    with open(i) as f:
        json_list.append(json.load(f))
print(json_list[0]['end'])
output_dict = dict()

tracking_point_dict = dict()
with open(tracking_point_path) as f:
    tracking_point_dict = json.load(f)

output_dict['width'] = json_list[0]['width']
output_dict['height'] = json_list[0]['height']
output_dict['object_num'] = 0
for i in range(len(json_list)):
    if output_dict['object_num'] < json_list[i]['object_num']:
        output_dict['object_num'] = json_list[i]['object_num']
color_dict = []
total_frames = len(tracking_point_dict['color_dict'])
for i in range(total_frames):
    color_dict.append({'color_label':[0], 'color_count':[json_list[0]['width']*json_list[0]['height']], "label_sum": {}})
output_dict['color_dict'] = color_dict

for data in json_list:
    start_f = data['start']
    end_f = data['end']
    for i in range(start_f, end_f):
        if len(tracking_point_dict['color_dict'][i]['color_label']) != 1:
            output_dict['color_dict'][i] = tracking_point_dict['color_dict'][i]
        else:
            output_dict['color_dict'][i] = data['color_dict'][i]

with open(output_segmant_tracking_path, 'w') as f:
    json.dump(output_dict, f)
