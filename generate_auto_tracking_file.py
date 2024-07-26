import json
import sys
import os
path = sys.argv[1]
if not os.path.isfile(path):
    print(f"Error: {path} not found. Exit.")
    exit()
tracking_point_list = None
try:
    with open(path) as json_data:
        tracking_point_list = json.load(json_data)
except:
    print(f"Error: cannot read {path} as json, Exit.")
    exit()
click_point_dict = dict()
prompt_path = path.replace("tracking_point_list.json", "click_prompt.json")
if os.path.isfile(prompt_path):
    with open(prompt_path) as json_data:
        click_point_dict = json.load(json_data)
else:
    print("cannot find correspond click_prompt.json, Exit.")
    exit()
current_start_idx = -1
output_pair = []
for idx, i in enumerate(tracking_point_list):
    if idx == len(tracking_point_list)-1 and current_start_idx != -1:
        output_pair.append([current_start_idx, i])
    if str(i) not in click_point_dict:
        continue
    if current_start_idx == -1:
        current_start_idx = i
    else:
        output_pair.append([current_start_idx, i])
        current_start_idx = i
bat_path = path.replace("tracking_point_list.json", "multi_track_execute.bat")
f = open(bat_path, "w")
for i in output_pair:
    f.write(f"start python multi_tracking.py {sys.argv[2]} {path.replace('tracking_point_list.json', 'tracking_point.json')} {i[0]} {i[1]}\n")
f.close()
print(output_pair)
