import tkinter as tk
from tkinter import filedialog, messagebox
import json
import numpy as np
from PIL import Image, ImageTk
import cv2
from seg_track_anything import aot_model2ckpt, tracking_objects_in_video, draw_mask
from model_args import segtracker_args,sam_args,aot_args
from SegTracker import SegTracker
from tool.transfer_tools import draw_outline, draw_points
import torch
import threading
from scipy.ndimage import binary_dilation
from scipy.spatial.distance import cdist
from pathlib import Path
import os
from itertools import groupby
import copy
from skimage import measure
from sklearn.cluster import DBSCAN
from random import randrange
from tkinter.simpledialog import askstring

np.random.seed(200)
_palette = ((np.random.random((3*255)))*255).astype(np.uint8).tolist()
_palette = [175, 0, 0, 0, 175, 0, 0, 0, 175, 75, 175, 0, 175, 0, 175, 0, 175, 175]+_palette

def thread_tracking(Seg_Tracker, video_path, video_fps, iframe, detect_check, end_check, end_frame, click_stack, current_label_dict, overwrite, os_env):
    video_name = '.'.join(os.path.basename(video_path).split('.')[:-1])
    #path tag to change posix
    if os_env == 'posix':
        tracking_result_dir = "/".join(os.getcwd().split("/")[:-1])+f'/output/{video_name}'
    elif os_env == 'nt':
        tracking_result_dir = f'{os.path.join(os.path.dirname(__file__), "tracking_results", f"{video_name}")}'

    output_masked_json_dir= f'{tracking_result_dir}'
    if end_frame < iframe:
        reversed = True
    else:
        reversed = False
    if end_check:
        result = tracking_objects_in_video(Seg_Tracker, video_path, None, video_fps, iframe, end_frame, False, False, detect_check, reversed, False, current_label_dict, os_env)
        if result == None:
            messagebox.showinfo('Finish', "No label mask and manual mask. Exit.")
            return
        messagebox.showinfo('Finish', f'Tracking success. output file is at \n{output_masked_json_dir}')
    else:
        result = tracking_objects_in_video(Seg_Tracker, video_path, None, video_fps, iframe, -1, False, False, detect_check, False, current_label_dict, os_env)
        if result == None:
            messagebox.showinfo('Finish', "No label mask and manual mask. Exit.")
            return
        messagebox.showinfo('Finish', f'Tracking success. output file is at \n{output_masked_json_dir}')
    if overwrite == False:
        return

    frame_mask = Seg_Tracker.first_frame_mask
    merged_list = []
    color_label = []
    color_count = []
    data_dict = dict()
    for i in frame_mask:
        merged_list.extend(i)
    g = groupby(merged_list)
    label_sum_dict = dict()
    for key, group in g:
        now_total = len(list(group))
        color_label.append(int(key))
        color_count.append(now_total)
        if str(key) != '0':
            if str(key) not in label_sum_dict:
                label_sum_dict[str(key)] = now_total
            else:
                label_sum_dict[str(key)] += now_total

    if os.path.isfile(f'{tracking_result_dir}/tracking_point.json'):
        manual_point_json = None
        with open(f'{tracking_result_dir}/tracking_point.json') as json_data:
            manual_point_json = json.load(json_data)
        vaild_point_list = []
        first_mask = Seg_Tracker.first_frame_mask
        for idx, i in enumerate(manual_point_json['color_dict']):
            if len(i['color_label']) != 1:
                if end_frame < iframe and idx > end_frame and idx <= iframe:
                    vaild_point_list.append(idx)
                elif end_frame > iframe and idx < end_frame and idx >= iframe:
                    vaild_point_list.append(idx)
        if iframe not in vaild_point_list:
            vaild_point_list.insert(0, iframe)
            manual_point_json["color_dict"][iframe]["color_label"] = color_label
            manual_point_json["color_dict"][iframe]["color_count"] = color_count
            manual_point_json["color_dict"][iframe]["label_sum"] = label_sum_dict
            with open(f'{tracking_result_dir}/tracking_point.json', 'w') as f:
                json.dump(manual_point_json, f)

        vaild_point_list.append(end_frame)
        #path tag to change posix
        if os_env == 'posix':
            pass
        elif os_env == 'nt':
            with open(f'{tracking_result_dir}/multi_track_execute.bat', 'w') as bat_output:
                for i in range(len(vaild_point_list)-1):
                    bat_output.write(f'start python multi_tracking.py {video_path} {tracking_result_dir}/tracking_point.json {vaild_point_list[i]} {vaild_point_list[i+1]} {reversed}\n')
            messagebox.showinfo('Finish', f'Multi track file generated. Please execute "multi_track_execute.bat" on cmd.')
    else:
        vcap = cv2.VideoCapture(video_path)
        width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        manual_point_json = dict()
        manual_point_json["width"] = int(width)
        manual_point_json["height"] = int(height)
        manual_point_json["object_num"] = Seg_Tracker.get_obj_num()
        manual_point_json['label_series'] = current_label_dict
        manual_point_json["color_dict"] = []
        for i in range(int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))):
            manual_point_json["color_dict"].append({"color_label": [0], "color_count": [width*height], "label_sum": {}})
        manual_point_json["color_dict"][iframe]["color_label"] = color_label
        manual_point_json["color_dict"][iframe]["color_count"] = color_count
        manual_point_json["color_dict"][iframe]["label_sum"] = label_sum_dict
        with open(f'{tracking_result_dir}/tracking_point.json', 'w') as f:
            json.dump(manual_point_json, f)
        messagebox.showinfo('Finish', f'tracking_point.json does not found, file is created.')
        vcap.release()

    click_dict = dict()
    if os.path.isfile(f'{tracking_result_dir}/click_prompt.json'):
        with open(f'{tracking_result_dir}/click_prompt.json') as json_data:
            click_dict = json.load(json_data)
    if str(iframe) not in click_dict:
        click_dict[str(iframe)] = dict()
    for i in range(len(click_stack)):
        click_dict[str(iframe)][str(i+1)] = click_stack[i]
    with open(f'{tracking_result_dir}/click_prompt.json', 'w') as f:
        json.dump(click_dict, f)


def chack_group_distance(mask):
    labels = measure.label(mask, connectivity=2)
    groups = measure.regionprops(labels)
    value_groups = dict()

    for i in groups:
        x, y = int(i.centroid[0]), int(i.centroid[1])
        value = mask[x][y]
        if value not in value_groups:
            value_groups[value] = []
        value_groups[value].extend(i.coords.tolist())

    is_apart = {}
    group_centers = {}
    #old
    #for value, gids in value_groups.items():
    #    if len(gids) > 1:
    #        for i in range(len(gids) - 1):
    #            for j in range(i + 1, len(gids)):
    #                group_i_points = gids[i]
    #                group_j_points = gids[j]
    #                max_distance = np.min(cdist(group_i_points, group_j_points, 'euclidean'))
    #                if value not in distances:
    #                    distances[value] = -2147483647
    #                if max_distance > distances[value]:
    #                    distances[value] = max_distance
                    #disconnected_pairs.append((gids[i], gids[j]))

    #new
    for value, gids in value_groups.items():
        clustering=DBSCAN(eps=2,min_samples=10).fit(gids)
        if len(set(clustering.labels_.tolist())) > 1:
            is_apart[value] = True
        averaged = np.average(np.array(gids), axis=0)
        group_centers[value] = averaged
    #for gid in value_groups.items():
    #    group_points = np.argwhere(mask == gid[0])
    #    group_points = tuple(tuple(sub) for sub in group_points)
    #    center_x, center_y = np.mean(group_points, axis=0)
    #    group_centers[gid[0]] = (center_x, center_y)
    return is_apart, group_centers

class DataLabelingApp:
    def __init__(self, master):
        self.master = master
        master.title("Data Labeling Tool")

        self.DEFAULT_LABEL_DICT = {"1": 'O',"2": 'X',"3": 'xx',"4": 'nn','5': 'ss','6': 'H'}
        self.current_label_dict = dict()

        self.current_frame_num = tk.IntVar(value=0)
        self.current_second_num = tk.StringVar(value="0")
        self.end_frame = tk.IntVar(value=0)

        # Frame for video and image display areas
        self.display_frame = tk.Frame(master)
        self.display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Parameters for threading
        self.Seg_Tracker_threading = None
        self.video_path_threading = None
        self.video_fps_threading = 0
        self.current_idx_threading = 0
        self.ab_check_threading = 0
        self.end_check_threading = 0
        self.entry_end_threading = 0
        self.os_env = os.name #LINUX:"posix", WIN:"nt"

        #SAT Parameters
        aot_args["model"] = 'r50_deaotl'
        aot_args["model_path"] = aot_model2ckpt["r50_deaotl"]
        aot_args["long_term_mem_gap"] = 9999
        aot_args["max_len_long_term"] = 9999
        # reset sam args
        segtracker_args["sam_gap"] = 9999
        segtracker_args["max_obj_num"] = 6
        sam_args["generator_args"]["points_per_side"] = 16
        self.Seg_Tracker = SegTracker(segtracker_args, sam_args, aot_args)
        self.click_stack = [[[],[]],[[],[]],[[],[]],[[],[]],[[],[]],[[],[]]]

        # Video frame display area
        self.video_canvas = tk.Canvas(self.display_frame, bg='black')
        self.video_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.video_canvas.bind("<Button-1>", self.on_video_frame_click)

        # Image display area for marking
        self.image_canvas = tk.Canvas(self.display_frame, bg='black')
        self.image_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.image_canvas.bind("<Button-1>", self.mark_label)

        # Frame for controls
        self.label_frame = tk.Frame(master)
        self.label_frame.pack(fill=tk.X, side=tk.TOP, padx=10)

        self.info_label = tk.Label(self.label_frame, text='current frame: \npath: ', anchor="e", justify=tk.LEFT)
        self.info_label.pack(side="right",anchor="n")

        # Frame for controls
        self.control_frame = tk.Frame(master)
        self.control_frame.pack(fill=tk.X, side=tk.TOP, padx=10, pady=5, ipady=30)

        self.upper_control_frame = tk.Frame(self.control_frame)
        self.upper_control_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, ipady=4)

        self.lower_control_frame = tk.Frame(self.control_frame)
        self.lower_control_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, ipady=4)

        self.abnormal_control_frame = tk.Frame(self.control_frame)
        self.abnormal_control_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, ipady=4)

        # Load video button
        self.load_video_button = tk.Button(self.upper_control_frame, text="Load Video", command=self.load_video)
        self.load_video_button.place(relx=0, relheight=1, relwidth=0.05)

        # Scroll bar for video navigation
        self.video_scroll = tk.Scale(self.upper_control_frame,width=15, from_=0, to=100, orient=tk.HORIZONTAL, command=self.update_video_frame)
        self.video_scroll.place(relx=0.05, rely=-0.1, relheight=1.1, relwidth=0.24)
        self.master.bind("<Left>", self.prev_frame)
        self.master.bind("<Right>", self.next_frame)

        #input for frae
        vcmd = (self.master.register(self.validate),'%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
        self.entry = tk.Entry(self.upper_control_frame, textvariable=self.current_frame_num, validate = 'key', validatecommand = vcmd)
        self.entry.place(relx=0.29, relheight=1, relwidth=0.04)


        self.jump_button = tk.Button(self.upper_control_frame, text="Jump to frame", command=self.jump_to_frame)
        self.jump_button.place(relx=0.33, relheight=1, relwidth=0.06)

        #self.end_check = tk.IntVar()
        #self.c3 = tk.Checkbutton(self.upper_control_frame, text='Ending frame',variable=self.end_check, onvalue=True, offvalue=False)
        #self.c3.place(relx=0.33, relheight=1, relwidth=0.07)

        self.second_label = tk.Label(self.upper_control_frame, text='second:')
        self.second_label.place(relx=0.41, relheight=1, relwidth=0.03)

        #input for second
        self.entry_second = tk.Entry(self.upper_control_frame, textvariable=self.current_second_num, validate = 'key', validatecommand = vcmd)
        self.entry_second.place(relx=0.44, relheight=1, relwidth=0.06)

        self.jump_second_button = tk.Button(self.upper_control_frame, text="Jump to second", command=self.jump_to_second)
        self.jump_second_button.place(relx=0.5, relheight=1, relwidth=0.08)


        # Button to capture current frame to mark
        self.capture_frame_button = tk.Button(self.upper_control_frame, text="Capture Frame", command=self.capture_frame)
        self.capture_frame_button.place(relx=0.58, relheight=1, relwidth=0.06)

        self.track_labels_button = tk.Button(self.upper_control_frame, text="Track Video", command=self.track_labels_all)
        self.track_labels_button.place(relx=0.64, relheight=1, relwidth=0.06)

        self.track_labels_frame_button = tk.Button(self.upper_control_frame, text="Track Video to Frame", command=self.track_labels_frame)
        self.track_labels_frame_button.place(relx=0.7, relheight=1, relwidth=0.1)

        self.entry_end = tk.Entry(self.upper_control_frame, textvariable=self.end_frame, validate = 'key', validatecommand = vcmd)
        self.entry_end.place(relx=0.8, relheight=1, relwidth=0.08)

        self.ab_check = tk.IntVar()
        self.c1 = tk.Checkbutton(self.upper_control_frame, text='Check abnormal when tracking',variable=self.ab_check, onvalue=True, offvalue=False)
        self.c1.place(relx=0.86, relheight=1, relwidth=0.14)

        self.cap = None
        self.current_frame = None
        self.current_idx = 0
        self.captured_frame = None
        self.current_frame_in_canvas = None
        self.label_mode = None  # Can be "positive" or "negative"
        self.current_mask = None
        self.width = None
        self.height = None
        self.masked_frame = None
        self.video_path = None
        self.video_fps = 0
        self.thread_video = None
        self.abnormal_list = []
        self.idx = 0
        self.stop_event = threading.Event()
        # Bind the resize event
        self.master.bind("<Configure>", self.on_window_resize)

        self.labels_history = []  # To store the history of label operations
        self.json_data = None  # To store json
        self.label_colors = {
            1: tuple(_palette[0:3]),
            2: tuple(_palette[3:6]),
            3: tuple(_palette[6:9]),
            4: tuple(_palette[9:12]),
            5: tuple(_palette[12:15]),
            6: tuple(_palette[15:18])
        }

        # Current label code variable
        self.current_label_code = tk.StringVar(master)
        self.current_label_code.set("1")  # Default value
        self.current_label_code.trace("w", self.label_change)

        # Load mask button
        self.load_json_button = tk.Button(self.lower_control_frame, text="Load JSON", command=self.load_json_data)
        self.load_json_button.place(relx=0, relheight=1, relwidth=0.05)

        # Load mask button
        self.load_default_json_button = tk.Button(self.lower_control_frame, text="Load default JSON", command=self.load_default_json_data)
        self.load_default_json_button.place(relx=0.05, relheight=1, relwidth=0.08)

        # Dropdown menu for label codes
        self.label_code_menu = tk.OptionMenu(self.lower_control_frame, self.current_label_code, "1", "2", "3", "4", "5", "6")
        self.label_code_menu.place(relx=0.13, relheight=1, relwidth=0.07)
        self.label_code_menu.configure(width=4)

        # Labeling buttons
        self.add_label_button = tk.Button(self.lower_control_frame, text="Add Label", command=self.add_label)
        self.add_label_button.place(relx=0.2, relheight=1, relwidth=0.05)

        self.del_label_button = tk.Button(self.lower_control_frame, text="Delete Label", command=self.del_label)
        self.del_label_button.place(relx=0.25, relheight=1, relwidth=0.08)

        # Labeling buttons
        self.positive_button = tk.Button(self.lower_control_frame, text="Positive Label", command=lambda: self.toggle_label("positive"))
        self.positive_button.place(relx=0.33, relheight=1, relwidth=0.07)

        self.negative_button = tk.Button(self.lower_control_frame, text="Negative Label", command=lambda: self.toggle_label("negative"))
        self.negative_button.place(relx=0.4, relheight=1, relwidth=0.07)

        # Labeling buttons
        self.reset_button = tk.Button(self.lower_control_frame, text="Reset", command=self.reset_label)
        self.reset_button.place(relx=0.47, relheight=1, relwidth=0.04)

        # Save manual mask
        self.save_manual_mask_button = tk.Button(self.lower_control_frame, text="Save manual mask", command=self.save_manual_mask)
        self.save_manual_mask_button.place(relx=0.51, relheight=1, relwidth=0.08)

        # Save tracking point
        self.save_tracking_point_button = tk.Button(self.lower_control_frame, text="Save tracking point", command=self.save_tracking_point)
        self.save_tracking_point_button.place(relx=0.59, relheight=1, relwidth=0.1)

        self.find_previous_abnormal_button = tk.Button(self.abnormal_control_frame, text="Find previous abnormal frame", command=self.find_previous_abnormal)
        self.find_previous_abnormal_button.place(relx=0, relheight=1, relwidth=0.15)

        self.find_next_abnormal_button = tk.Button(self.abnormal_control_frame, text="Find next abnormal frame", command=self.find_next_abnormal)
        self.find_next_abnormal_button.place(relx=0.15, relheight=1, relwidth=0.15)

        #self.find_large_range_button = tk.Button(self.abnormal_control_frame, text="Find next abnormal range frame", command=self.find_and_jump_to_large_range)
        #self.find_large_range_button.place(relx=0.3, relheight=1, relwidth=0.15)

        self.find_all_abnormal_button = tk.Button(self.abnormal_control_frame, text="Find all abnormal frame", command=self.find_all_abnormal)
        self.find_all_abnormal_button.place(relx=0.3, relheight=1, relwidth=0.15)

        self.save_abnormal_frame_button = tk.Button(self.abnormal_control_frame, text="Save abnormal frame", command=self.output_abnormal_frame)
        self.save_abnormal_frame_button.place(relx=0.45, relheight=1, relwidth=0.15)

        self.generate_video_button = tk.Button(self.abnormal_control_frame, text="Generate Video", command=self.generate_video)
        self.generate_video_button.place(relx=0.85, relheight=1, relwidth=0.15)

        #self.reversed_check = tk.IntVar()
        #self.c2 = tk.Checkbutton(self.control_frame, text='Reversed tracking',variable=self.reversed_check, onvalue=True, offvalue=False)
        #self.c2.pack(side=tk.LEFT)


    def validate(self, action, index, value_if_allowed, prior_value, text, validation_type, trigger_type, widget_name):
        value_if_allowed.replace(" ", "")
        if value_if_allowed == '':
            return True
        if value_if_allowed:
            try:
                int(value_if_allowed)
                return True
            except ValueError:
                return False
        else:
            return False

    def SegTracker_add_first_frame(self, Seg_Tracker, origin_frame, predicted_mask):
        with torch.cuda.amp.autocast():
            # Reset the first frame's mask
            frame_idx = 0
            Seg_Tracker.restart_tracker()
            Seg_Tracker.add_reference(origin_frame, predicted_mask, frame_idx)
            Seg_Tracker.first_frame_mask = predicted_mask

        return Seg_Tracker

    def seg_acc_click(self, Seg_Tracker, prompt, origin_frame):
        # seg acc to click
        #print(prompt)
        predicted_mask, masked_frame = Seg_Tracker.seg_acc_click(
                                                          origin_frame=origin_frame,
                                                          coords=np.array(prompt["points_coord"]),
                                                          modes=np.array(prompt["points_mode"]),
                                                          multimask=prompt["multimask"],
                                                        )

        Seg_Tracker = self.SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask)

        return masked_frame

    def get_click_prompt(self, click_stack, point):

        click_stack[0].append(point["coord"])
        click_stack[1].append(point["mode"])

        prompt = {
            "points_coord":click_stack[0],
            "points_mode":click_stack[1],
            "multimask":"True",
        }

        return prompt

    def load_video(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            #== TODO: load default word dict if word not in json, else load word dict in json
            self.current_label_dict = self.DEFAULT_LABEL_DICT
            menu = self.label_code_menu["menu"]
            #print(menu)
            menu.delete(0, "end")
            first_element = False
            for i in self.current_label_dict:
                label_name = self.current_label_dict[str(i)]
                menu.add_command(label=label_name, command=tk._setit(self.current_label_code, str(label_name)))
            self.current_label_code.set(self.current_label_dict["1"])
            #==
            self.video_scroll.set(0)
            self.video_path = file_path
            self.json_data = None
            self.cap = cv2.VideoCapture(file_path)
            self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_scroll.config(to=total_frames - 1)
            self.update_video_frame()
            self.Seg_Tracker = SegTracker(segtracker_args, sam_args, aot_args)
            self.click_stack = [[[],[]],[[],[]],[[],[]],[[],[]],[[],[]],[[],[]]]
            self.abnormal_list = []
            self.Seg_Tracker.restart_tracker()
            self.Seg_Tracker.sam.have_embedded = False
            for i in range(6):
                self.Seg_Tracker.reset_origin_merged_mask(None, i)
            self.Seg_Tracker.update_origin_merged_mask(None)
            self.captured_frame = self.current_frame.copy()
            self.masked_frame = self.current_frame.copy()
            self.display_frame_in_canvas(self.captured_frame, self.image_canvas)
            my_text = f"Current frame: {self.video_scroll.get()}\npath:{self.video_path}"
            self.info_label.config(text = my_text)

    def jump_to_frame(self):
        try:
            frame = int(self.entry.get())
        except:
            return
        frame_count = self.video_scroll.cget("to")
        if frame > frame_count or frame < 0:
            return
        self.video_scroll.set(frame)
        if self.video_fps != 0:
            second_val = float(self.video_scroll.get()) / self.video_fps
            self.current_second_num.set("{:.2f}".format(second_val))
        self.update_video_frame()

    def jump_to_second(self):
        try:
            second = float(self.entry_second.get())
        except:
            return
        frame_val = int(second * self.video_fps)
        frame_count = self.video_scroll.cget("to")
        if frame_val > frame_count or frame_val < 0:
            return
        self.video_scroll.set(frame_val)
        self.current_frame_num.set(int(self.video_scroll.get()))
        self.update_video_frame()

    def update_video_frame(self, event=None):
        self.current_frame_num.set(int(self.video_scroll.get()))
        if self.video_fps != 0:
            second_val = float(self.video_scroll.get()) / self.video_fps
            self.current_second_num.set("{:.2f}".format(second_val))
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.video_scroll.get()))
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.current_frame_in_canvas = frame
                if self.json_data:
                    index = int(self.video_scroll.get())
                    if index < len(self.json_data['color_dict']):
                        mask = self.rebuild_mask_from_index(index)
                        self.current_mask = mask
                        frame = self.apply_mask_to_frame(frame, mask)
                        self.current_frame_in_canvas = frame
                self.display_frame_in_canvas(frame, self.video_canvas)

    def apply_mask_to_frame(self, frame, mask):
        overlay = frame.copy()
        for label in np.unique(mask):
            if label == 0:
                continue
            color = np.array(self.label_colors[label])
            mask_layer = (mask == label)
            overlay[mask_layer] = np.clip(0.7 * color + 0.3 * frame[mask_layer], 0, 255).astype(np.uint8)
        return overlay

    def capture_frame(self):
        if self.current_frame is not None:
            self.labels_history = []
            self.click_stack = []
            for i in range(segtracker_args["max_obj_num"]):
                self.click_stack.append([[],[]])
            self.Seg_Tracker.restart_tracker()
            self.Seg_Tracker.sam.have_embedded = False
            for i in range(segtracker_args["max_obj_num"]):
                self.Seg_Tracker.reset_origin_merged_mask(None, i)
            self.Seg_Tracker.update_origin_merged_mask(None)
            self.captured_frame = self.current_frame.copy()
            self.masked_frame = self.current_frame.copy()
            self.display_frame_in_canvas(self.captured_frame, self.image_canvas)
            my_text = f"Current frame: {self.video_scroll.get()}\npath:{self.video_path}"
            self.current_idx = int(self.video_scroll.get())
            self.info_label.config(text = my_text)
            self.Seg_Tracker.first_frame_mask = None

    def rebuild_mask_from_index(self, index):
        if not hasattr(self, 'json_data'):
            print("JSON data has not been loaded.")
            return

        color_dict = self.json_data['color_dict'][index]
        color_label = color_dict['color_label']
        color_count = color_dict['color_count']
        self.width = self.json_data['width']
        self.height = self.json_data['height']

        mask = self.rebuild_mask(color_label, color_count, self.width, self.height)
        return mask

    def toggle_label(self, mode):
        if self.label_mode == mode:
            self.label_mode = None
        else:
            self.label_mode = mode
        # Update button states
        self.positive_button.config(relief="sunken" if self.label_mode == "positive" else "raised")
        self.negative_button.config(relief="sunken" if self.label_mode == "negative" else "raised")

    def mark_label(self, event):
        if self.label_mode and self.captured_frame is not None:
            beta_frame = cv2.convertScaleAbs(self.masked_frame, alpha=0.3, beta=10)
            self.display_frame_in_canvas(beta_frame, self.image_canvas)
            self.master.update()
            label_word = self.current_label_code.get()
            mydict = self.current_label_dict
            self.Seg_Tracker.curr_idx = int(list(mydict.keys())[list(mydict.values()).index(label_word)])
            x, y = event.x, event.y
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()

            scale_x = self.width / canvas_width
            scale_y = self.height / canvas_height

            if scale_x > scale_y:
                acc_x = int(event.x * scale_x)
                acc_y = int((event.y + self.height / scale_x / 2 - canvas_height / 2) * scale_x)
            else:
                acc_x = int((event.x + self.width / scale_y / 2 - canvas_width / 2) * scale_y)
                acc_y = int(event.y * scale_y)
            if acc_x > self.width or acc_x < 0:
                return
            if acc_y > self.height or acc_y < 0:
                return
            #draw mask by sam
            if self.label_mode == "positive" :
                point = {"coord": [acc_x, acc_y], "mode": 1}
            else:
                # TODO：add everything positive points
                point = {"coord": [acc_x, acc_y], "mode": 0}

            # get click prompts for sam to predict mask
            label_word = self.current_label_code.get()
            mydict = self.current_label_dict
            target_idx = int(list(mydict.keys())[list(mydict.values()).index(label_word)])
            click_prompt = self.get_click_prompt(self.click_stack[target_idx-1], point)
            #print(self.click_stack)
            #print(self.Seg_Tracker.curr_idx)
            masked_frame = self.seg_acc_click(self.Seg_Tracker, click_prompt, self.captured_frame)

            self.masked_frame = masked_frame
            self.display_frame_in_canvas(masked_frame, self.image_canvas)

            # Create the oval and store its ID along with label type and code for undo operation
            color = "blue" if self.label_mode == "positive" else "red"
            oval_id = self.image_canvas.create_oval(x - 1, y - 1, x + 1, y + 1, fill=color, outline=color)
            label_info = (oval_id, self.label_mode, target_idx)  # Include the selected label code
            self.labels_history.append(label_info)

    def reset_label(self):
        #now_click_stack = self.click_stack[int(self.current_label_code.get())]
        #now_click_stack[0] = [[]]
        #now_click_stack[1] = []
        #self.labels_history = []

        #prompt = {
        #    "points_coord":now_click_stack[0],
        #    "points_mode":now_click_stack[1],
        #    "multimask":"True",
        #}
        #masked_frame = self.seg_acc_click(self.Seg_Tracker, prompt, self.captured_frame)
        #self.display_frame_in_canvas(masked_frame, self.image_canvas)
        beta_frame = cv2.convertScaleAbs(self.masked_frame, alpha=0.3, beta=10)
        self.display_frame_in_canvas(beta_frame, self.image_canvas)
        self.master.update()
        label_word = self.current_label_code.get()
        mydict = self.current_label_dict
        now_index = int(list(mydict.keys())[list(mydict.values()).index(label_word)])
        self.click_stack[now_index-1] = [[],[]]
        self.labels_history = []
        self.masked_frame = self.captured_frame
        self.Seg_Tracker = SegTracker(segtracker_args, sam_args, aot_args)
        for i in range(segtracker_args["max_obj_num"]):
            if len(self.click_stack[i][0]) != 0:
                self.Seg_Tracker.curr_idx = i+1
                prompt = {
                 "points_coord":self.click_stack[i][0],
                 "points_mode":self.click_stack[i][1],
                 "multimask":"True",
                }
                self.masked_frame = self.seg_acc_click(self.Seg_Tracker, prompt, self.masked_frame)
        prev_mask = self.Seg_Tracker.first_frame_mask
        self.Seg_Tracker.update_origin_merged_mask(prev_mask)
        self.Seg_Tracker.curr_idx = now_index
        #self.Seg_Tracker = SegTracker(segtracker_args, sam_args, aot_args)
        self.display_frame_in_canvas(self.masked_frame, self.image_canvas)

    def save_manual_mask(self):
        if self.video_path is not None:
            #WTD: change file path name
            #path tag to change posix
            if self.os_env == 'posix':
                video_name = '.'.join(os.path.basename(self.video_path).split('.')[:-1])
                save_path = "/".join(os.getcwd().split("/")[:-1])+f'/output/{video_name}'
            elif self.os_env == 'nt':
                save_path = self.video_path.replace('assets', 'tracking_results').replace('.mp4', '').replace('raw_video/', '')
            save_mask_path = save_path
            if not os.path.exists(save_mask_path):
                os.makedirs(save_mask_path)
            json_dict = dict()
            json_dict['video_path'] = self.video_path
            json_dict['video_mask'] = dict()
            #json_file = Path(save_mask_path+'data.json')
            #if os.path.isfile(json_file):
            #    with open(save_mask_path+'data.json') as f:
            #        json_dict = json.load(f)
            frame_idx = str(self.current_idx)
            if self.click_stack == [[],[]]*segtracker_args["max_obj_num"]:
                return
            click_stack_dict = dict()
            for i in range(len(self.click_stack)):
                click_stack_dict[str(i+1)] = dict()
                click_stack_dict[str(i+1)]['point'] = self.click_stack[i][0]
                click_stack_dict[str(i+1)]['mode'] = self.click_stack[i][1]
            #json_dict['video_mask'][frame_idx] = click_stack_dict
            frame_mask = self.Seg_Tracker.first_frame_mask
            merged_list = []
            color_label = []
            color_count = []
            data_dict = dict()
            for i in frame_mask:
                merged_list.extend(i)
            g = groupby(merged_list)
            label_sum_dict = dict()
            for key, group in g:
                now_total = len(list(group))
                color_label.append(int(key))
                color_count.append(now_total)
                if str(key) != '0':
                    if str(key) not in label_sum_dict:
                        label_sum_dict[str(key)] = now_total
                    else:
                        label_sum_dict[str(key)] += now_total
            if not os.path.isfile(f'{save_path}/tracking_point.json'):
                data_dict["width"] = self.width
                data_dict["height"] = self.height
                data_dict["object_num"] = self.Seg_Tracker.get_obj_num()
                data_dict["label_series"] = self.current_label_dict
                data_dict["color_dict"] = []
                for i in range(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                    data_dict["color_dict"].append({"color_label": [0], "color_count": [self.width*self.height], "label_sum": {}})
            else:
                with open(f'{save_path}/tracking_point.json') as json_data:
                    data_dict = json.load(json_data)
            if self.Seg_Tracker.get_obj_num() > data_dict["object_num"]:
                data_dict["object_num"] = self.Seg_Tracker.get_obj_num()
                data_dict["label_series"] = self.current_label_dict
            data_dict["color_dict"][self.current_idx]["color_label"] = color_label
            data_dict["color_dict"][self.current_idx]["color_count"] = color_count
            data_dict["color_dict"][self.current_idx]["label_sum"] = label_sum_dict
            with open(f'{save_path}/tracking_point.json', 'w') as f:
                json.dump(data_dict, f)
            click_dict = dict()
            if os.path.isfile(f'{save_path}/click_prompt.json'):
                with open(f'{save_path}/click_prompt.json') as json_data:
                    click_dict = json.load(json_data)
            if str(self.current_idx) not in click_dict:
                click_dict[str(self.current_idx)] = dict()
            for i in range(len(self.click_stack)):
                click_dict[str(self.current_idx)][str(i+1)] = self.click_stack[i]
            with open(f'{save_path}/click_prompt.json', 'w') as f:
                json.dump(click_dict, f)
            #with open(save_mask_path+'data.json', 'w') as f:
            #    json.dump(json_dict, f)
            #    print("mask saved.")
    def save_tracking_point(self):
        #save different path for windows and linux
        video_name = '.'.join(os.path.basename(self.video_path).split('.')[:-1])
        if self.os_env == 'posix':
            save_path = "/".join(os.getcwd().split("/")[:-1])+f'/output/{video_name}'
        elif self.os_env == 'nt':
            save_path = os.path.dirname(self.video_path).replace('assets', 'tracking_results')+f'/{video_name}'
        tracking_point_list = []
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if os.path.isfile(f'{save_path}/tracking_point_list.json'):
            with open(f'{save_path}/tracking_point_list.json') as json_data:
                tracking_point_list = json.load(json_data)
        if int(self.current_idx) not in tracking_point_list:
            tracking_point_list.append(self.current_idx)
        tracking_point_list = sorted(tracking_point_list)
        click_point_dict = dict()
        if os.path.isfile(f'{save_path}/click_prompt.json'):
            with open(f'{save_path}/click_prompt.json') as json_data:
                click_point_dict = json.load(json_data)
        for i in click_point_dict:
            if int(i) > tracking_point_list[0] and int(i) < tracking_point_list[-1] and int(i) not in tracking_point_list:
                tracking_point_list.insert(1, int(i))
        tracking_point_list = sorted(tracking_point_list)
        with open(f'{save_path}/tracking_point_list.json', 'w') as f:
            json.dump(tracking_point_list, f)
        messagebox.showinfo('Finish', f'Tracking point json generated. \nPlease execute "generate_auto_tracking_file.py" to generate execute tracking file.')

    def display_frame_in_canvas(self, frame, canvas):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img.thumbnail((canvas.winfo_width(), canvas.winfo_height()), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        canvas.delete("all")
        canvas.create_image(canvas.winfo_width() / 2, canvas.winfo_height() / 2, image=imgtk)
        canvas.image = imgtk  # Keep a reference

    def on_window_resize(self, event):
        # Adjust the video frame according to the window size
        self.master.update_idletasks()
        new_width = int(self.master.winfo_width()*0.49)
        new_height = int(self.master.winfo_height()*0.7)
        self.video_canvas.config(width=new_width, height=new_height)
        self.image_canvas.config(width=new_width, height=new_height)
        if self.current_frame_in_canvas is not None:
            self.display_frame_in_canvas(self.current_frame_in_canvas, self.video_canvas)
        # Adjust the captured image frame according to the window size
        if self.masked_frame is not None:
            self.display_frame_in_canvas(self.masked_frame, self.image_canvas)
        #self.video_canvas.update_idletasks()
        #print("this width is:",self.video_canvas.winfo_width())

    def load_json(self, filepath):
        with open(filepath, 'r') as file:
            data = json.load(file)
        return data

    def rebuild_mask(self, color_label, color_count, width, height):
        result = np.repeat(color_label, color_count)
        mask = result.reshape(height, width)
        return mask

    def load_json_data(self):
        if self.cap is None:
            return
        filepath = filedialog.askopenfilename()
        if filepath:
            with open(filepath, 'r') as file:
                self.json_data = json.load(file)
            if 'label_series' not in self.json_data:
                while segtracker_args["max_obj_num"] < self.json_data['object_num']:
                    menu = self.label_code_menu["menu"]
                    last = menu.index("end")
                    items = []
                    for index in range(last+1):
                        items.append(int(menu.entrycget(index, "label")))
                    add_index = max(items) + 1
                    segtracker_args["max_obj_num"] = segtracker_args["max_obj_num"]+1
                    self.click_stack.append([[],[]])
                    self.label_colors[segtracker_args["max_obj_num"]] = (randrange(256), randrange(256), randrange(256))
                    menu.add_command(label=str(add_index), command=tk._setit(self.current_label_code, str(add_index)))
            else:
                self.current_label_dict = self.json_data['label_series']
                menu = self.label_code_menu["menu"]
                menu.delete(0, "end")
                first_element = False
                idx_num = 0
                for i in self.current_label_dict:
                    label_name = self.current_label_dict[str(i)]
                    menu.add_command(label=label_name, command=tk._setit(self.current_label_code, str(label_name)))
                    idx_num += 1
                    if idx_num > segtracker_args["max_obj_num"]:
                        segtracker_args["max_obj_num"] = segtracker_args["max_obj_num"]+1
                        self.click_stack.append([[],[]])
                        self.label_colors[segtracker_args["max_obj_num"]] = (randrange(256), randrange(256), randrange(256))
                self.current_label_code.set(self.current_label_dict["1"])
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.video_scroll.set(0)
            ret, frame = self.cap.read()
            mask = self.rebuild_mask_from_index(0)
            self.current_mask = mask
            frame = self.apply_mask_to_frame(frame, mask)
            self.current_frame_in_canvas = frame
            self.display_frame_in_canvas(frame, self.video_canvas)
            self.update_video_frame()
            self.captured_frame = self.current_frame.copy()
            self.masked_frame = self.current_frame.copy()
            self.display_frame_in_canvas(self.captured_frame, self.image_canvas)


    def load_default_json_data(self):
        #save different path for windows and linux
        if self.video_path is None:
            print("Video is not loading.")
            return
        video_name = '.'.join(os.path.basename(self.video_path).split('.')[:-1])
        if self.os_env == 'posix':
            tracking_result_dir = "/".join(os.getcwd().split("/")[:-1])+f'/output/{video_name}'
        elif self.os_env == 'nt':
            tracking_result_dir = f'{os.path.join(os.path.dirname(__file__), "tracking_results", f"{video_name}")}'
        filepath = f'{tracking_result_dir}/tracking_point.json'
        if filepath:
            with open(filepath, 'r') as file:
                self.json_data = json.load(file)
            if 'label_series' not in self.json_data:
                while segtracker_args["max_obj_num"] < self.json_data['object_num']:
                    menu = self.label_code_menu["menu"]
                    last = menu.index("end")
                    items = []
                    for index in range(last+1):
                        items.append(int(menu.entrycget(index, "label")))
                    add_index = max(items) + 1
                    segtracker_args["max_obj_num"] = segtracker_args["max_obj_num"]+1
                    self.click_stack.append([[],[]])
                    self.label_colors[segtracker_args["max_obj_num"]] = (randrange(256), randrange(256), randrange(256))
                    menu.add_command(label=str(add_index), command=tk._setit(self.current_label_code, str(add_index)))
            else:
                self.current_label_dict = self.json_data['label_series']
                menu = self.label_code_menu["menu"]
                menu.delete(0, "end")
                first_element = False
                idx_num = 0
                for i in self.current_label_dict:
                    label_name = self.current_label_dict[str(i)]
                    menu.add_command(label=label_name, command=tk._setit(self.current_label_code, str(label_name)))
                    idx_num += 1
                    if idx_num > segtracker_args["max_obj_num"]:
                        segtracker_args["max_obj_num"] = segtracker_args["max_obj_num"]+1
                        self.click_stack.append([[],[]])
                        self.label_colors[segtracker_args["max_obj_num"]] = (randrange(256), randrange(256), randrange(256))
                self.current_label_code.set(self.current_label_dict["1"])
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.video_scroll.set(0)
            ret, frame = self.cap.read()
            mask = self.rebuild_mask_from_index(0)
            self.current_mask = mask
            frame = self.apply_mask_to_frame(frame, mask)
            self.current_frame_in_canvas = frame
            self.display_frame_in_canvas(frame, self.video_canvas)
            self.update_video_frame()
            self.captured_frame = self.current_frame.copy()
            self.masked_frame = self.current_frame.copy()
            self.display_frame_in_canvas(self.captured_frame, self.image_canvas)

    def add_label(self):
        menu = self.label_code_menu["menu"]
        last = menu.index("end")
        items = []
        name = askstring('Label', 'Please input label name.')
        for index in range(last+1):
            items.append(menu.entrycget(index, "label"))
        add_index = len(items) + 1
        segtracker_args["max_obj_num"] = segtracker_args["max_obj_num"]+1
        self.click_stack.append([[],[]])
        self.label_colors[segtracker_args["max_obj_num"]] = (randrange(256), randrange(256), randrange(256))
        menu.add_command(label=name, command=tk._setit(self.current_label_code, name))
        self.current_label_dict[str(add_index)] = name
        #print(self.current_label_dict)

    def del_label(self):
        menu = self.label_code_menu["menu"]
        last = menu.index("end")
        menu.delete(last)    # deleted the option
        self.click_stack.pop()
        del self.label_colors[segtracker_args["max_obj_num"]]
        segtracker_args["max_obj_num"] = segtracker_args["max_obj_num"]-1
        self.current_label_code.set(self.current_label_dict["1"])  # Default value

    def on_video_frame_click(self, event):
        # 获取Canvas的当前大小
        if self.json_data is None:
            return
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()

        # 计算缩放比例
        scale_x = self.width / canvas_width
        scale_y = self.height / canvas_height

        # 调整点击坐标以匹配原始视频帧的尺寸
        if scale_x > scale_y:
            x = int(event.x * scale_x)
            y = int((event.y + self.height / scale_x / 2 - canvas_height / 2) * scale_x)
        else:
            x = int((event.x + self.width / scale_y / 2 - canvas_width / 2) * scale_y)
            y = int(event.y * scale_y)
        if x > self.width or x < 0:
            return
        if y > self.height or y < 0:
            return

        if self.current_mask is not None:
            if 0 <= y < self.current_mask.shape[0] and 0 <= x < self.current_mask.shape[1]:
                label = self.current_mask[y, x]
                if label != 0:
                    menu = self.label_code_menu["menu"]
                    self.current_label_code.set(self.current_label_dict[str(label)])

    def label_change(self, *args):
        #prev_mask = self.Seg_Tracker.first_frame_mask
        #self.Seg_Tracker.update_origin_merged_mask(prev_mask)
        #self.Seg_Tracker.restart_tracker()
        if self.video_path is None:
            return
        label_word = self.current_label_code.get()
        mydict = self.current_label_dict
        now_index = int(list(mydict.keys())[list(mydict.values()).index(label_word)])
        for i in range(segtracker_args["max_obj_num"]):
            self.Seg_Tracker.reset_origin_merged_mask(None, i)
        self.Seg_Tracker.update_origin_merged_mask(None)
        #self.masked_frame = self.captured_frame
        #print(self.click_stack)
        for i in range(segtracker_args["max_obj_num"]):
            self.Seg_Tracker.curr_idx = i+1
            if i+1 != now_index and len(self.click_stack[i][0]) != 0:
                 prompt = {
                     "points_coord":self.click_stack[i][0],
                     "points_mode":self.click_stack[i][1],
                     "multimask":"True",
                 }
                 self.masked_frame = self.seg_acc_click(self.Seg_Tracker, prompt, self.masked_frame)
                 prev_mask = self.Seg_Tracker.first_frame_mask
                 self.Seg_Tracker.update_origin_merged_mask(prev_mask)
        self.Seg_Tracker.curr_idx = now_index
        masked_frame = self.masked_frame
        if len(self.click_stack[self.Seg_Tracker.curr_idx-1][0]) != 0:
             prompt = {
                 "points_coord":self.click_stack[self.Seg_Tracker.curr_idx-1][0],
                 "points_mode":self.click_stack[self.Seg_Tracker.curr_idx-1][1],
                 "multimask":"True",
             }
             masked_frame = self.seg_acc_click(self.Seg_Tracker, prompt, self.masked_frame)
        #self.Seg_Tracker = SegTracker(segtracker_args, sam_args, aot_args)
        self.display_frame_in_canvas(masked_frame, self.image_canvas)

    def track_labels(self, by_frame=False):
        if self.video_path is not None:
            overwrite = True
            self.Seg_Tracker_threading = copy.deepcopy(self.Seg_Tracker)
            self.video_path_threading = self.video_path
            self.video_fps_threading = self.video_fps
            self.current_idx_threading = int(self.current_idx)
            self.ab_check_threading = self.ab_check.get()
            self.end_check_threading = by_frame
            self.entry_end_threading = int(self.entry_end.get())

            self.stop_event.clear()
            label_point = []
            tracking_point_json = None
            video_name = '.'.join(os.path.basename(self.video_path).split('.')[:-1])
            #save different path for windows and linux
            if self.os_env == 'posix':
                tracking_result_dir = "/".join(os.getcwd().split("/")[:-1])+f'/output/{video_name}'
            elif self.os_env == 'nt':
                tracking_result_dir = f'{os.path.join(os.path.dirname(__file__), "tracking_results", f"{video_name}")}'
            filepath = f'{tracking_result_dir}/tracking_point.json'
            if os.path.isfile(f'{tracking_result_dir}/tracking_point.json'):
                with open(f'{tracking_result_dir}/tracking_point.json') as json_data:
                    tracking_point_json = json.load(json_data)
                for idx, i in enumerate(tracking_point_json["color_dict"]):
                    if len(i['color_label']) != 1:
                        label_point.append(idx)
            #if Seg_Tracker.first_frame_mask is None, try to load tracking_point.json
            if self.Seg_Tracker_threading.first_frame_mask is None:
                Seg_Tracker = SegTracker(segtracker_args, sam_args, aot_args)

                while segtracker_args["max_obj_num"] < tracking_point_json['object_num']:
                    menu = self.label_code_menu["menu"]
                    last = menu.index("end")
                    items = []
                    for index in range(last+1):
                        items.append(menu.entrycget(index, "label"))
                    add_index = len(items) + 1
                    segtracker_args["max_obj_num"] = segtracker_args["max_obj_num"]+1
                    self.click_stack.append([[],[]])
                    self.label_colors[segtracker_args["max_obj_num"]] = (randrange(256), randrange(256), randrange(256))
                    menu.add_command(label=str(add_index), command=tk._setit(self.current_label_code, str(add_index)))

                #print(label_point)
                start_f = 0

                if self.Seg_Tracker_threading.first_frame_mask is None and label_point == []:
                    print("No label mask and manual mask. Exit.")
                    return None
                elif self.Seg_Tracker_threading.first_frame_mask is None and label_point != []:
                    in_range_index = [x for x in label_point if (x>=self.current_idx_threading and x<self.entry_end_threading)]
                    if len(in_range_index) == 0:
                        print("No label mask in range. Exit.")
                        return None
                    start_f = min(in_range_index)
                    print("no mask here, find nearest index is:", start_f)
                self.current_idx_threading = start_f
                if not os.path.isfile(f'{tracking_result_dir}/click_prompt.json'):
                    ggbb528()
                click_prompt_json = None
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
                ret, self.captured_frame = self.cap.read()
                with open(f'{tracking_result_dir}/click_prompt.json') as json_data:
                    click_prompt_json = json.load(json_data)
                for i in click_prompt_json:
                    if len(click_prompt_json[i]) > segtracker_args["max_obj_num"]:
                        segtracker_args["max_obj_num"] = len(click_prompt_json[i])
                for idx in click_prompt_json[str(start_f)]:
                    print(click_prompt_json[str(start_f)][idx])
                    if len(click_prompt_json[str(start_f)][idx][1]) == 0:
                        continue
                    self.Seg_Tracker_threading.curr_idx = int(idx)
                    click_prompt = {
                            "points_coord":click_prompt_json[str(start_f)][idx][0],
                            "points_mode":click_prompt_json[str(start_f)][idx][1],
                            "multimask":"True",
                    }
                    masked_frame = self.seg_acc_click(self.Seg_Tracker_threading, click_prompt, self.captured_frame)
                    prev_mask = self.Seg_Tracker_threading.first_frame_mask
                    self.Seg_Tracker_threading.update_origin_merged_mask(prev_mask)
                overwrite = False
            self.thread_video = threading.Thread(target=thread_tracking, args=(
                self.Seg_Tracker_threading,
                self.video_path_threading,
                self.video_fps_threading,
                self.current_idx_threading,
                self.ab_check_threading,
                self.end_check_threading,
                self.entry_end_threading,
                self.click_stack,
                self.current_label_dict,
                overwrite,
                self.os_env))
            self.thread_video.start()

            #reverse track after normal track
            if int(self.entry_end.get()) > int(self.current_idx):
                start_f = 0
                self.Seg_Tracker_reverse_threading = copy.deepcopy(self.Seg_Tracker)
                self.video_path_reverse_threading = self.video_path
                self.video_fps_reverse_threading = self.video_fps
                self.ab_check_reverse_threading = self.ab_check.get()
                self.end_check_reverse_threading = by_frame
                self.current_idx_reverse_threading = int(self.entry_end.get())
                self.entry_end_reverse_threading = int(self.current_idx)
                #print(self.current_idx_reverse_threading, self.entry_end_reverse_threading)
                if label_point == []:
                    print("No label mask and manual mask. Exit.")
                    return None
                elif label_point != []:
                    in_range_index = [x for x in label_point if (x<=self.current_idx_reverse_threading and x>self.entry_end_reverse_threading)]
                    if len(in_range_index) == 0:
                        print("No label mask in range. Exit.")
                        return None
                    start_f = max(in_range_index)
                    print("Start revered track, find nearest index is:", start_f)
                self.current_idx_reverse_threading = start_f
                if not os.path.isfile(f'{tracking_result_dir}/click_prompt.json'):
                    ggbb528()
                click_prompt_json = None
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
                ret, self.captured_frame = self.cap.read()
                with open(f'{tracking_result_dir}/click_prompt.json') as json_data:
                    click_prompt_json = json.load(json_data)
                for i in click_prompt_json:
                    if len(click_prompt_json[i]) > segtracker_args["max_obj_num"]:
                        segtracker_args["max_obj_num"] = len(click_prompt_json[i])
                for idx in click_prompt_json[str(start_f)]:
                    print(click_prompt_json[str(start_f)][idx])
                    if len(click_prompt_json[str(start_f)][idx][1]) == 0:
                        continue
                    self.Seg_Tracker_reverse_threading.curr_idx = int(idx)
                    click_prompt = {
                            "points_coord":click_prompt_json[str(start_f)][idx][0],
                            "points_mode":click_prompt_json[str(start_f)][idx][1],
                            "multimask":"True",
                    }
                    masked_frame = self.seg_acc_click(self.Seg_Tracker_reverse_threading, click_prompt, self.captured_frame)
                    prev_mask = self.Seg_Tracker_reverse_threading.first_frame_mask
                    self.Seg_Tracker_reverse_threading.update_origin_merged_mask(prev_mask)
                overwrite = False
            self.thread_reverse_video = threading.Thread(target=thread_tracking, args=(
                self.Seg_Tracker_reverse_threading,
                self.video_path_reverse_threading,
                self.video_fps_reverse_threading,
                self.current_idx_reverse_threading,
                self.ab_check_reverse_threading,
                self.end_check_reverse_threading,
                self.entry_end_reverse_threading,
                self.click_stack,
                self.current_label_dict,
                overwrite,
                self.os_env))
            self.thread_reverse_video.start()

    def track_labels_all(self):
        self.track_labels(False)

    def track_labels_frame(self):
        self.track_labels(True)

    def find_previous_abnormal(self):
        if self.idx == 0:
            return
        self.idx -= 1
        self.video_scroll.set(self.abnormal_list[self.idx])
        self.update_video_frame()

    def find_next_abnormal(self):
        if self.idx < len(self.abnormal_list)-1:
            self.idx += 1
            self.video_scroll.set(self.abnormal_list[self.idx])
            self.update_video_frame()
            return

        if self.json_data is None:
            return
        min_area = 50
        prev_center_dict = None
        found = 0
        print("start finding abnormanl area at ",self.video_scroll.get())
        reversed = 0
        #if self.end_check.get() and self.video_scroll.get() > int(self.entry_end.get()):
        #    reversed = 1
        #for idx in range(self.video_scroll.get()+1, len(self.json_data['color_dict'])):
        idx = self.video_scroll.get()+1 if reversed == 0 else self.video_scroll.get()-1
        while True:
            #if self.end_check.get():
                #if reversed == 0 and idx > int(self.entry_end.get()):
            if idx > int(self.entry_end.get()):
                break
                #elif reversed == 1 and idx < int(self.entry_end.get()):
                #    break
            if len(self.abnormal_list) > 0:
                if idx < self.abnormal_list[-1] + 20:
                    continue
            print("detect frame {}".format(idx),end='\r')
            #check mask frames
            total_label_count = [0]*segtracker_args["max_obj_num"]
            cl = self.json_data['color_dict'][idx]['color_label']
            ct = self.json_data['color_dict'][idx]['color_count']
            for i in self.json_data['color_dict'][idx]['label_sum']:
                total_label_count[int(i)-1] = int(self.json_data['color_dict'][idx]['label_sum'][i])
            check_min_area = any(x > 0 and x < min_area for x in total_label_count)
            #print(total_label_count)
            if check_min_area:
                abnormal_label = 0
                for i in range(len(total_label_count)):
                    if total_label_count[i] != 0:
                        abnormal_label = i
                        break
                self.video_scroll.set(idx)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = self.cap.read()
                mask = self.rebuild_mask_from_index(idx)
                self.current_mask = mask
                frame = self.apply_mask_to_frame(frame, mask)
                self.current_frame_in_canvas = frame
                self.display_frame_in_canvas(frame, self.video_canvas)
                menu = self.label_code_menu["menu"]
                messagebox.showinfo("found abnormal", f"found abnormal label {self.current_label_dict[str(i+1)]}: area too small")
                self.abnormal_list.append(idx)
                self.idx = len(self.abnormal_list) - 1
                found = 1
                break

            #check mask distance
            now_mask = self.rebuild_mask_from_index(idx)
            is_apart, centers_dict = chack_group_distance(now_mask)
            current_displacements = dict()
            if prev_center_dict != None:
                for value, center in centers_dict.items():
                    if value in prev_center_dict:
                        displacement = np.sqrt((center[0] - prev_center_dict[value][0])**2 + (center[1] - prev_center_dict[value][1])**2)
                        current_displacements[value] = displacement
            dist_threshold_list = [[0]]*segtracker_args["max_obj_num"]
            for value, dist in is_apart.items():
                dist_threshold_list[value-1] = 1 if dist == True else 0
            for value, dist in current_displacements.items():
                dist_threshold_list[value-1] = 2 if dist > 100 else 0
            if 1 in dist_threshold_list:
                abnormal_label = 0
                for i in range(len(dist_threshold_list)):
                    if dist_threshold_list[i] == 1:
                        abnormal_label = i
                        break
                menu = self.label_code_menu["menu"]
                messagebox.showinfo("found abnormal", f"found abnormal label {self.current_label_dict[str(i+1)]}: apart area")
                self.abnormal_list.append(idx)
                self.idx = len(self.abnormal_list) - 1
                self.video_scroll.set(self.abnormal_list[self.idx])
                self.update_video_frame()
                found = 1
                break
            if 2 in dist_threshold_list:
                abnormal_label = 0
                for i in range(len(dist_threshold_list)):
                    if dist_threshold_list[i] == 2:
                        abnormal_label = i
                        break
                messagebox.showinfo("found abnormal", f"found abnormal label {self.current_label_dict[str(i+1)]}: distance too large")
                self.abnormal_list.append(idx)
                self.idx = len(self.abnormal_list) - 1
                self.video_scroll.set(self.abnormal_list[self.idx])
                self.update_video_frame()
                found = 1
                break
            prev_center_dict = centers_dict
            idx = idx+1 if reversed == 0 else idx-1
        if found == 0:
            messagebox.showinfo('Finish', f'Finding abnormal finish. No abnormal found')

    def find_all_abnormal(self):
        if self.json_data is None:
            return
        min_area = 50
        prev_center_dict = None

        for idx in range(self.current_idx+1, len(self.json_data['color_dict'])):
            if self.end_check.get() and idx > int(self.entry_end.get()):
                break
            if len(self.abnormal_list) > 0:
                if idx < self.abnormal_list[-1] + 20:
                    continue
            print("detect frame {}".format(idx),end='\r')
            #check mask frames
            total_label_count = [0]*segtracker_args["max_obj_num"]
            cl = self.json_data['color_dict'][idx]['color_label']
            ct = self.json_data['color_dict'][idx]['color_count']
            for i in self.json_data['color_dict'][idx]['label_sum']:
                total_label_count[int(i)-1] = int(self.json_data['color_dict'][idx]['label_sum'][i])
            check_min_area = any(x > 0 and x < min_area for x in total_label_count)
            #print(total_label_count)
            if check_min_area:
                print()
                self.video_scroll.set(idx)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = self.cap.read()
                mask = self.rebuild_mask_from_index(idx)
                self.current_mask = mask
                frame = self.apply_mask_to_frame(frame, mask)
                self.current_frame_in_canvas = frame
                self.display_frame_in_canvas(frame, self.video_canvas)
                print("found abnormal min area")
                self.abnormal_list.append(idx)
                continue

            #check mask distance
            now_mask = self.rebuild_mask_from_index(idx)
            distance_dict, centers_dict = chack_group_distance(now_mask)
            current_displacements = dict()
            if prev_center_dict != None:
                for value, center in centers_dict.items():
                    if value in prev_center_dict:
                        displacement = np.sqrt((center[0] - prev_center_dict[value][0])**2 + (center[1] - prev_center_dict[value][1])**2)
                        current_displacements[value] = displacement
            dist_threshold_list = [[0]]*segtracker_args["max_obj_num"]
            for value, dist in distance_dict.items():
                dist_threshold_list[value-1] = 1 if dist == True else 0
            for value, dist in current_displacements.items():
                dist_threshold_list[value-1] = 2 if dist > 100 else 0
            if 1 in dist_threshold_list:
                print()
                print("found abnormal apart area")
                self.abnormal_list.append(idx)
                continue
            if 2 in dist_threshold_list:
                print()
                print("found abnormal distance")
                self.abnormal_list.append(idx)
                continue
            prev_center_dict = centers_dict

        if len(self.abnormal_list) != 0:
            self.idx = 0
            self.video_scroll.set(self.abnormal_list[0])
            self.update_video_frame()
        messagebox.showinfo('Finish', f'Finding abnormal finish.')

    def find_and_jump_to_large_range(self):
        #legacy function
        if self.json_data is None:
            return
        max_length = 200
        earliest_frame_index = -1

        for idx in range(self.current_idx+1, len(self.json_data['color_dict'])):
            label_range = [None, None, None, None, None, None]
            total_count = 0
            cl = self.json_data['color_dict'][idx]['color_label']
            ct = self.json_data['color_dict'][idx]['color_count']
            for a, b in zip(cl, ct):
                total_count += b
                if a == 0:
                    continue
                current_height = total_count // self.width
                current_width = total_count % self.width
                if label_range[a-1] is None:
                    label_range[a-1] = [[current_width, current_height], [current_width, current_height]]
                else:
                    label_range[a-1][0][0] = current_width if current_width < label_range[a-1][0][0] else label_range[a-1][0][0]
                    label_range[a-1][0][1] = current_height if current_height < label_range[a-1][0][1] else label_range[a-1][0][1]
                    label_range[a-1][1][0] = current_width if current_width > label_range[a-1][1][0] else label_range[a-1][1][0]
                    label_range[a-1][1][1] = current_height if current_height > label_range[a-1][1][1] else label_range[a-1][1][1]
            #print(label_range)
            for i in range(segtracker_args["max_obj_num"]):
                if label_range[i] is None:
                    continue
                dist = ((label_range[i][1][0] - label_range[i][0][0])**2 + (label_range[i][0][1] - label_range[i][1][1])**2) ** 0.5
                #print(dist)
                if dist > max_length:
                    self.video_scroll.set(idx)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = self.cap.read()
                    mask = self.rebuild_mask_from_index(idx)
                    self.current_mask = mask
                    frame = self.apply_mask_to_frame(frame, mask)
                    self.current_frame_in_canvas = frame
                    self.display_frame_in_canvas(frame, self.video_canvas)
                    return

    def capture_first_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        self.video_scroll.set(0)
        self.current_idx = 0
        if frame is not None:
            self.current_frame = frame
            self.current_frame_in_canvas = frame
            self.Seg_Tracker.restart_tracker()
            self.labels_history = []
            self.captured_frame = frame.copy()
            self.masked_frame = frame.copy()
            if self.json_data:
                index = int(self.video_scroll.get())
                if index < len(self.json_data['color_dict']):
                    mask = self.rebuild_mask_from_index(0)
                    self.current_mask = mask
                    frame = self.apply_mask_to_frame(frame, mask)
                    self.current_frame_in_canvas = frame
            self.display_frame_in_canvas(self.captured_frame, self.image_canvas)

    def prev_frame(self, event=None):
        now_idx = self.video_scroll.get()
        if now_idx == 0:
            return
        now_idx -= 1
        self.video_scroll.set(now_idx)
        self.update_video_frame()

    def next_frame(self, event=None):
        frame_count = self.video_scroll.cget("to")
        now_idx = self.video_scroll.get()
        if now_idx == frame_count:
            return
        now_idx += 1
        self.video_scroll.set(now_idx)
        self.update_video_frame()

    def output_abnormal_frame(self, event=None):
        #print(self.abnormal_list)
        #path tag to change posix
        Path("./abnormal_list").mkdir(parents=True, exist_ok=True)
        out_file = open("./abnormal_list/myfile.json", "w")
        json.dump(self.abnormal_list, out_file, indent = 6)

        out_file.close()

    def generate_video(self):
        messagebox.showinfo('Select', "Please Select json data to apply on video.")
        filepath = filedialog.askopenfilename()
        if filepath is None:
            return
        with open(filepath) as json_data:
            manual_point_json = json.load(json_data)
        messagebox.showinfo('Select', "Please Select video.")
        video_path = filedialog.askopenfilename()
        if video_path is None:
            return
        vcap = cv2.VideoCapture(video_path)
        length = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vcap.get(cv2.CAP_PROP_FPS)
        width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc =  cv2.VideoWriter_fourcc(*"mp4v")
        video_name = '.'.join(os.path.basename(video_path).split('.')[:-1])
        if self.os_env == 'posix':
            output_path = "/".join(os.getcwd().split("/")[:-1])+f'/output/{video_name}/{video_name}_mask.mp4'
        elif self.os_env == 'nt':
            output_path = os.path.dirname(video_path).replace('assets', 'tracking_results')+f'/{video_name}/{video_name}_mask.mp4'

        print(output_path)

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        counter = 0
        while vcap.isOpened():
            ret, frame  = vcap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            color_dict = manual_point_json['color_dict'][counter]
            color_label = color_dict['color_label']
            color_count = color_dict['color_count']

            mask = self.rebuild_mask(color_label, color_count, width, height)
            frame = self.apply_mask_to_frame(frame, mask)
            out.write(frame)
            print('frame {}/{} writed'.format(counter,length),end='\r')
            counter += 1
        out.release()
        vcap.release()
        #cv2.imwrite("name.jpg", frame)
        print("I am NT")
        print("PASS")
    #for frame_index, mask_data in self.json_data.items():
    #    for _, mask in mask_data['shapes'].items():
    #        area = self.calculate_mask_area(mask)
    #        if area <= min_area:
    #            earliest_frame_index = int(frame_index)
    #            break

    #if earliest_frame is not None:
    #    self.jump_to_frame(earliest_frame)
    #else:
    #    print("None")
if __name__ == '__main__':
    root = tk.Tk()
    root.geometry("1600x900")  # Initial size of the window
    app = DataLabelingApp(root)
    root.mainloop()
