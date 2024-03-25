import tkinter as tk
from tkinter import filedialog
import json
import numpy as np
from PIL import Image, ImageTk
import cv2
from seg_track_anything import aot_model2ckpt, tracking_objects_in_video, draw_mask
from model_args import segtracker_args,sam_args,aot_args
from SegTracker import SegTracker
from tool.transfer_tools import draw_outline, draw_points
import torch

class DataLabelingApp:
    def __init__(self, master):
        self.master = master
        master.title("Data Labeling Tool")

        # Frame for video and image display areas
        self.display_frame = tk.Frame(master)
        self.display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

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
        self.control_frame = tk.Frame(master)
        self.control_frame.pack(fill=tk.X, padx=10, pady=5)

        # Load video button
        self.load_video_button = tk.Button(self.control_frame, text="Load Video", command=self.load_video)
        self.load_video_button.pack(side=tk.LEFT)

        # Scroll bar for video navigation
        self.video_scroll = tk.Scale(self.control_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.update_video_frame)
        self.video_scroll.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Button to capture current frame to mark
        self.capture_frame_button = tk.Button(self.control_frame, text="Capture Frame", command=self.capture_frame)
        self.capture_frame_button.pack(side=tk.LEFT)

        # Labeling buttons
        self.positive_button = tk.Button(self.control_frame, text="Positive Label", command=lambda: self.toggle_label("positive"))
        self.positive_button.pack(side=tk.LEFT)

        self.negative_button = tk.Button(self.control_frame, text="Negative Label", command=lambda: self.toggle_label("negative"))
        self.negative_button.pack(side=tk.LEFT)

        self.cap = None
        self.current_frame = None
        self.captured_frame = None
        self.current_frame_in_canvas = None
        self.label_mode = None  # Can be "positive" or "negative"
        self.current_mask = None
        self.width = None
        self.height = None
        self.masked_frame = None
        self.video_path = None
        self.video_fps = 0

        # Bind the resize event
        self.master.bind("<Configure>", self.on_window_resize)

        # Labeling buttons
        self.reset_button = tk.Button(self.control_frame, text="Reset", command=self.reset_label)
        self.reset_button.pack(side=tk.LEFT)

        self.labels_history = []  # To store the history of label operations
        self.json_data = None  # To store json
        self.label_colors = {
            1: (255, 0, 0),  # Red
            2: (0, 255, 0),  # Green
            3: (0, 0, 255),  # Blue
            4: (255, 255, 0),  # Yellow
            5: (255, 0, 255),  # Purple
            6: (0, 255, 255)   # Cyan
        }

        # Current label code variable
        self.current_label_code = tk.StringVar(master)
        self.current_label_code.set("1")  # Default value
        self.current_label_code.trace("w", self.label_change)

        # Dropdown menu for label codes
        self.label_code_menu = tk.OptionMenu(self.control_frame, self.current_label_code, "1", "2", "3", "4", "5", "6")
        self.label_code_menu.pack(side=tk.LEFT)

        # Frame for controls
        self.control_frame = tk.Frame(master)
        self.control_frame.pack(fill=tk.X, padx=10, pady=5)

        # Load mask button
        self.load_json_button = tk.Button(self.control_frame, text="Load JSON", command=self.load_json_data)
        self.load_json_button.pack(side=tk.LEFT)

        self.track_labels_button = tk.Button(self.control_frame, text="Track Video", command=self.track_labels)
        self.track_labels_button.pack(side=tk.LEFT)

        self.find_small_mask_button = tk.Button(self.control_frame, text="Find next abnormal small frame", command=self.find_and_jump_to_small_mask)
        self.find_small_mask_button.pack(side=tk.LEFT)

        self.find_large_range_button = tk.Button(self.control_frame, text="Find next abnormal range frame", command=self.find_and_jump_to_large_range)
        self.find_large_range_button.pack(side=tk.LEFT)

        self.capture_first_frame_button = tk.Button(self.control_frame, text="Capture First Frame", command=self.capture_first_frame)
        self.capture_first_frame_button.pack(side=tk.LEFT)

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
        self.video_path = file_path
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_scroll.config(to=total_frames - 1)
            self.update_video_frame()
            self.Seg_Tracker = SegTracker(segtracker_args, sam_args, aot_args)
            self.click_stack = [[[],[]],[[],[]],[[],[]],[[],[]],[[],[]],[[],[]]]

    def update_video_frame(self, event=None):
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
        # 创建一个透明层
        overlay = frame.copy()
        for label in np.unique(mask):
            if label == 0:  # 跳过背景
                continue
            # 为每个label应用颜色和透明度
            color = np.array(self.label_colors[label])
            # 仅选择mask中对应label的部分
            mask_layer = (mask == label)
            # 计算透明层的颜色值
            overlay[mask_layer] = np.clip(0.3 * color + 0.7 * frame[mask_layer], 0, 255).astype(np.uint8)
        return overlay

    def capture_frame(self):
        if self.current_frame is not None:
            self.labels_history = []
            self.Seg_Tracker.restart_tracker()
            self.captured_frame = self.current_frame.copy()
            self.masked_frame = self.current_frame.copy()
            self.display_frame_in_canvas(self.captured_frame, self.image_canvas)

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
            click_prompt = self.get_click_prompt(self.click_stack[int(self.current_label_code.get())-1], point)
            print(self.click_stack)
            masked_frame = self.seg_acc_click(self.Seg_Tracker, click_prompt, self.captured_frame)
            self.masked_frame = masked_frame
            self.display_frame_in_canvas(masked_frame, self.image_canvas)

            # Create the oval and store its ID along with label type and code for undo operation
            color = "blue" if self.label_mode == "positive" else "red"
            oval_id = self.image_canvas.create_oval(x - 1, y - 1, x + 1, y + 1, fill=color, outline=color)
            label_info = (oval_id, self.label_mode, self.current_label_code.get())  # Include the selected label code
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
        now_index = int(self.current_label_code.get())
        self.click_stack[now_index-1] = [[],[]]
        self.labels_history = []
        self.masked_frame = self.captured_frame
        self.Seg_Tracker = SegTracker(segtracker_args, sam_args, aot_args)
        for i in range(6):
            if len(self.click_stack[i][0]) != 0:
                 self.Seg_Tracker.curr_idx = i+1
                 prompt = {
                     "points_coord":self.click_stack[i-1][0],
                     "points_mode":self.click_stack[i-1][1],
                     "multimask":"True",
                 }
                 self.masked_frame = self.seg_acc_click(self.Seg_Tracker, prompt, self.masked_frame)
        prev_mask = self.Seg_Tracker.first_frame_mask
        self.Seg_Tracker.update_origin_merged_mask(prev_mask)
        self.Seg_Tracker.curr_idx = now_index
        #self.Seg_Tracker = SegTracker(segtracker_args, sam_args, aot_args)
        self.display_frame_in_canvas(self.masked_frame, self.image_canvas)


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
        if self.current_frame_in_canvas is not None:
            self.display_frame_in_canvas(self.current_frame_in_canvas, self.video_canvas)
        # Adjust the captured image frame according to the window size
        if self.masked_frame is not None:
            self.display_frame_in_canvas(self.masked_frame, self.image_canvas)

    def load_json(self, filepath):
        with open(filepath, 'r') as file:
            data = json.load(file)
        return data

    def rebuild_mask(self, color_label, color_count, width, height):
        mask = np.zeros((height, width), dtype=int)
        current_pos = 0
        for label, count in zip(color_label, color_count):
            for _ in range(count):
                row = current_pos // width
                col = current_pos % width
                mask[row, col] = label
                current_pos += 1
        return mask

    def load_json_data(self):
        if self.cap is None:
            return
        filepath = filedialog.askopenfilename()
        if filepath:
            with open(filepath, 'r') as file:
                self.json_data = json.load(file)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.video_scroll.set(0)
            ret, frame = self.cap.read()
            mask = self.rebuild_mask_from_index(0)
            self.current_mask = mask
            frame = self.apply_mask_to_frame(frame, mask)
            self.current_frame_in_canvas = frame
            self.display_frame_in_canvas(frame, self.video_canvas)

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
                    self.current_label_code.set(str(label))

    def label_change(self, *args):
        #prev_mask = self.Seg_Tracker.first_frame_mask
        #self.Seg_Tracker.update_origin_merged_mask(prev_mask)
        #self.Seg_Tracker.restart_tracker()
        now_index = int(self.current_label_code.get())
        for i in range(6):
            self.Seg_Tracker.reset_origin_merged_mask(None, i)
        self.Seg_Tracker.update_origin_merged_mask(None)
        #self.masked_frame = self.captured_frame

        for i in range(6):
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

    def track_labels(self):
        if self.video_path is not None:
            print(int(self.video_scroll.get()))
            tracking_objects_in_video(self.Seg_Tracker, self.video_path, None, self.video_fps, int(self.video_scroll.get()), False, False)

    def find_and_jump_to_small_mask(self):
        if self.json_data is None:
            return
        min_area = 50
        earliest_frame_index = -1

        for idx in range(self.video_scroll.get()+1, len(self.json_data['color_dict'])):
            total_label_count = [0, 0, 0, 0, 0, 0]
            cl = self.json_data['color_dict'][idx]['color_label']
            ct = self.json_data['color_dict'][idx]['color_count']
            for a, b in zip(cl, ct):
                if a == 0:
                    continue
                total_label_count[a-1] += b
            print(total_label_count)
            check_min_area = any(x > 0 and x < min_area for x in total_label_count)
            if check_min_area:
                print(idx)
                self.video_scroll.set(idx)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = self.cap.read()
                mask = self.rebuild_mask_from_index(idx)
                self.current_mask = mask
                frame = self.apply_mask_to_frame(frame, mask)
                self.current_frame_in_canvas = frame
                self.display_frame_in_canvas(frame, self.video_canvas)
                break

    def find_and_jump_to_large_range(self):
        if self.json_data is None:
            return
        max_length = 200
        earliest_frame_index = -1

        for idx in range(self.video_scroll.get()+1, len(self.json_data['color_dict'])):
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
            print(label_range)
            for i in range(6):
                if label_range[i] is None:
                    continue
                dist = ((label_range[i][1][0] - label_range[i][0][0])**2 + (label_range[i][0][1] - label_range[i][1][1])**2) ** 0.5
                print(dist)
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

root = tk.Tk()
root.geometry("1280x720")  # Initial size of the window
app = DataLabelingApp(root)
root.mainloop()
