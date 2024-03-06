from PIL.ImageOps import colorize, scale
import gradio as gr
import importlib
import sys
import os
import pdb
from matplotlib.pyplot import step
import tkinter

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


def clean():
    return None, None, None, None, None, None, [[], []]
