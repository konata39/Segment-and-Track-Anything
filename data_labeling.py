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
import wx
import json
from pathlib import Path


def clean():
    return None, None, None, None, None, None, [[], []]

def SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask):
    print("Wu zer tian")
    with torch.cuda.amp.autocast():
        # Reset the first frame's mask
        frame_idx = 0
        Seg_Tracker.restart_tracker()
        Seg_Tracker.add_reference(origin_frame, predicted_mask, frame_idx)
        Seg_Tracker.first_frame_mask = predicted_mask

    return Seg_Tracker

def init_SegTracker(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, origin_frame):

    if origin_frame is None:
        return None, origin_frame, [[], []], ""

    # reset aot args
    aot_args["model"] = aot_model
    aot_args["model_path"] = aot_model2ckpt[aot_model]
    aot_args["long_term_mem_gap"] = long_term_mem
    aot_args["max_len_long_term"] = max_len_long_term
    # reset sam args
    segtracker_args["sam_gap"] = sam_gap
    segtracker_args["max_obj_num"] = max_obj_num
    sam_args["generator_args"]["points_per_side"] = points_per_side

    Seg_Tracker = SegTracker(segtracker_args, sam_args, aot_args)
    Seg_Tracker.restart_tracker()
    print(Seg_Tracker)
    return Seg_Tracker, origin_frame, [[], []], ""

def seg_acc_click(Seg_Tracker, prompt, origin_frame):
    # seg acc to click
    predicted_mask, masked_frame = Seg_Tracker.seg_acc_click(
                                                      origin_frame=origin_frame,
                                                      coords=np.array(prompt["points_coord"]),
                                                      modes=np.array(prompt["points_mode"]),
                                                      multimask=prompt["multimask"],
                                                    )

    Seg_Tracker = SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask)

    return masked_frame

def get_click_prompt(click_stack, point):

    click_stack[0].append(point["coord"])
    click_stack[1].append(point["mode"]
    )

    prompt = {
        "points_coord":click_stack[0],
        "points_mode":click_stack[1],
        "multimask":"True",
    }

    return prompt

def add_new_object(Seg_Tracker):

    prev_mask = Seg_Tracker.first_frame_mask
    Seg_Tracker.update_origin_merged_mask(prev_mask)
    Seg_Tracker.curr_idx += 1

    print("Ready to add new object!")

    return Seg_Tracker, [[], []]

class PhotoCtrl(wx.App):

    def __init__(self, redirect=False, filename=None):
        wx.App.__init__(self, redirect, filename)
        self.frame = wx.Frame(None, title='Photo Control')

        self.panel = wx.Panel(self.frame)
        self.PhotoMaxWSize = 1024
        self.PhotoMaxHSize = 768

        #Segment-and-Tracking data
        self.input_video = None
        self.input_img_seq = None
        self.Seg_Tracker = None
        self.input_first_frame = None
        self.origin_frame = None
        self.drawing_board = None
        self.click_stack = [[], []]

        self.filedict = None
        self.filename = None
        self.rate = 1

        self.createWidgets()
        self.frame.Show()

    def createWidgets(self):
        instructions = 'Browse for an image'
        img = wx.EmptyImage(1024,768)
        self.imageCtrl = wx.StaticBitmap(self.panel, wx.ID_ANY,
                                         wx.BitmapFromImage(img))
        self.imageCtrl.Bind(wx.EVT_LEFT_DOWN, self.onMouseClick_img)

        instructLbl = wx.StaticText(self.panel, label=instructions)
        self.photoTxt = wx.TextCtrl(self.panel, size=(200,-1))

        browseBtn = wx.Button(self.panel, label='Browse')
        browseBtn.Bind(wx.EVT_BUTTON, self.onBrowse)

        saveBtn = wx.Button(self.panel, label='Save Mask')
        saveBtn.Bind(wx.EVT_BUTTON, self.onSave)

        objBtn = wx.Button(self.panel, label='Add object')
        objBtn.Bind(wx.EVT_BUTTON, self.onObj)

        self.check_pn =  wx.CheckBox(self.panel, label='Negative')

        self.mainSizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.mainSizer.Add(wx.StaticLine(self.panel, wx.ID_ANY),
                           0, wx.ALL|wx.EXPAND, 5)
        self.mainSizer.Add(instructLbl, 0, wx.ALL, 5)
        self.mainSizer.Add(self.imageCtrl, 0, wx.ALL, 5)
        self.sizer.Add(self.photoTxt, 0, wx.ALL, 5)
        self.sizer.Add(browseBtn, 0, wx.ALL, 5)
        self.sizer.Add(saveBtn, 0, wx.ALL, 5)
        self.sizer.Add(objBtn, 0, wx.ALL, 5)
        self.sizer.Add(self.check_pn, 0, wx.ALL, 5)
        self.mainSizer.Add(self.sizer, 0, wx.ALL, 5)

        self.panel.SetSizer(self.mainSizer)
        self.mainSizer.Fit(self.frame)
        self.panel.Layout()

    def onBrowse(self, event):
        """
        Browse for file
        """
        wildcard = "PNG files (*.png)|*.png"
        dialog = wx.FileDialog(None, "Choose a file",
                               wildcard=wildcard,
                               style=wx.FD_OPEN)
        if dialog.ShowModal() == wx.ID_OK:
            self.photoTxt.SetValue(dialog.GetPath())
            self.filedict = '\\'.join(dialog.GetPath().split('\\')[:-1]) #self.filedict
            self.filename = dialog.GetPath().split('\\')[-1]
        dialog.Destroy()
        self.onView()

    def onView(self):
        filepath = self.photoTxt.GetValue()
        img = wx.Image(filepath, wx.BITMAP_TYPE_ANY)
        img_px = Image.open(filepath)
        self.origin_frame = np.array(img_px)
        # scale the image, preserving the aspect ratio
        W = img.GetWidth()
        H = img.GetHeight()
        W_rate = W / self.PhotoMaxWSize
        H_rate = H / self.PhotoMaxHSize
        print(W_rate, H_rate)
        if W_rate > H_rate:
            NewW = self.PhotoMaxWSize
            NewH = H / W_rate
            self.rate = W_rate
        else:
            NewW = W / H_rate
            NewH = self.PhotoMaxHSize
            self.rate = H_rate
        print(NewW, NewH)
        img = img.Scale(NewW,NewH)
        self.imageCtrl.SetBitmap(wx.BitmapFromImage(img))
        self.panel.Refresh()

    def onMouseClick_img(self, event):
        point_on_map = event.GetPosition()
        pom = [int(point_on_map[0] * self.rate), int(point_on_map[1] * self.rate)]

        # Default Positive
        if self.check_pn.GetValue() == True:
            point = {
                "coord": pom,
                "mode": 0,
            }
        else:
            point = {
                "coord": pom,
                "mode": 1,
            }

        click_prompt = get_click_prompt(self.click_stack, point)

        if self.Seg_Tracker is None:
            self.aot_model = 'r50_deaotl'
            self.max_len_long_term = 9999
            self.sam_gap = 9999
            self.long_term_mem = 9999
            self.max_obj_num = 50
            self.points_per_side = 16
            self.Seg_Tracker, _, _, _ = init_SegTracker(
                self.aot_model,
                self.long_term_mem,
                self.max_len_long_term,
                self.sam_gap,
                self.max_obj_num,
                self.points_per_side,
                self.origin_frame
            )
        masked_frame = seg_acc_click(self.Seg_Tracker, click_prompt, self.origin_frame)
        masked_frame_img = Image.fromarray(masked_frame, 'RGB')

        W, H = masked_frame_img.size
        W_rate = W / self.PhotoMaxWSize
        H_rate = H / self.PhotoMaxHSize
        if W_rate > H_rate:
            NewW = self.PhotoMaxWSize
            NewH = H / W_rate
            self.rate = W_rate
        else:
            NewW = W / H_rate
            NewH = self.PhotoMaxHSize
            self.rate = H_rate
        masked_frame_img = masked_frame_img.resize((int(NewW),int(NewH)))
        self.imageCtrl.SetBitmap(wx.BitmapFromBuffer(NewW, NewH, masked_frame_img.tobytes()))

    def onSave(self, event):
        output_path = self.filedict + "\\data.json"
        print(output_path)
        #out_file = open(, "w")
        my_file = Path(output_path)
        json_data = dict()
        if my_file.is_file():
            with open(output_path) as f:
                json_data = json.load(f)
        out_file = open(output_path, "w")
        json_data[self.filename] = self.click_stack
        json.dump(json_data, out_file, indent = 6)
        print(self.SegTracker.everything_points)
        print(self.SegTracker.everything_labels)

    def onObj(self, event):
        self.SegTracker, self.click_stack = add_new_object(self.Seg_Tracker)

if __name__ == '__main__':
    app = PhotoCtrl()
    app.MainLoop()
