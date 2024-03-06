import os
import cv2
import json
from model_args import segtracker_args,sam_args,aot_args
from PIL import Image
from aot_tracker import _palette
import numpy as np
import torch
import gc
import imageio
from scipy.ndimage import binary_dilation
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def save_prediction(pred_mask,output_dir,file_name):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask.save(os.path.join(output_dir,file_name))

def colorize_mask(pred_mask):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask = save_mask.convert(mode='RGB')
    return np.array(save_mask)


def draw_mask(img, mask, alpha=0.5, id_countour=False):
    img_mask = np.zeros_like(img)
    img_mask = img
    if id_countour:
        # very slow ~ 1s per image
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids!=0]

        for id in obj_ids:
            # Overlay color on  binary mask
            if id <= 255:
                color = _palette[id*3:id*3+3]
            else:
                color = [0,0,0]
            foreground = img * (1-alpha) + np.ones_like(img) * alpha * np.array(color)
            binary_mask = (mask == id)

            # Compose image
            img_mask[binary_mask] = foreground[binary_mask]

            countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
            img_mask[countours, :] = 0
    else:
        binary_mask = (mask!=0)
        countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
        foreground = img*(1-alpha)+colorize_mask(mask)*alpha
        img_mask[binary_mask] = foreground[binary_mask]
        img_mask[countours,:] = 0

    return img_mask.astype(img.dtype)

def draw_block(img, mask, width, height, alpha=1, id_countour=False):
    #img_mask = np.zeros_like(img)
    #img_mask = img

    if id_countour:
        # very slow ~ 1s per image
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids!=0]
        for id in obj_ids:
            # Overlay color on  binary mask
            if id <= 255:
                color = _palette[id*3:id*3+3]
            else:
                color = [0,0,0]
            foreground = img * (1-alpha) + np.ones_like(img) * alpha * np.array(color)
            binary_mask = (mask == id)

            # Compose image
            img_mask[binary_mask] = foreground[binary_mask]

            countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
            img_mask[countours, :] = 0
    else:
        unique_masks = np.unique(mask)
        corners = {}
        for i in unique_masks:
            if i != 0:  # 排除背景
                rows, cols = np.where(mask == i)
                top_left = (min(rows), min(cols))
                bottom_right = (max(rows), max(cols))
                corners[i] = (top_left, bottom_right)

                color = _palette[i*3:i*3+3]
                thickness = 2  # 线条粗细
                cv2.rectangle(img, (top_left[1], top_left[0]), (bottom_right[1],bottom_right[0]) , color, thickness)
        #countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
        #foreground = img*(1-alpha)+colorize_mask(mask)*alpha
        #img_mask[binary_mask] = foreground[binary_mask]
        #img_mask[countours,:] = 0
    img_mask = np.zeros_like(img)
    img_mask = img
    return img_mask.astype(img.dtype)

frame_idx = 0

video_name = 'blackswan'
tracking_result_dir = f'{os.path.join(os.path.dirname(__file__), "tracking_results", f"{video_name}")}'
io_args = {
    'tracking_result_dir': tracking_result_dir,
    'output_mask_dir': f'{tracking_result_dir}/{video_name}_masks_from_json',
    'output_masked_frame_dir': f'{tracking_result_dir}/{video_name}_masked_frames_from_json',
    'output_masked_json_dir': f'{tracking_result_dir}/{video_name}_json_frames',
    'output_video': f'{tracking_result_dir}/{video_name}_seg_from_json.mp4', # keep same format as input video
    'output_video_block': f'{tracking_result_dir}/{video_name}_seg_from_json_with_block.mp4', # keep same format as input video
    'output_gif': f'{tracking_result_dir}/{video_name}_seg.gif',
}

if os.path.isdir(io_args['output_mask_dir']):
    os.system(f"rm -r {io_args['output_mask_dir']}")
if not os.path.isdir(io_args['output_mask_dir']):
    os.makedirs(io_args['output_mask_dir'])
if os.path.isdir(io_args['output_masked_frame_dir']):
    os.system(f"rm -r {io_args['output_masked_frame_dir']}")
if not os.path.isdir(io_args['output_masked_frame_dir']):
    os.makedirs(io_args['output_masked_frame_dir'])
f = open(f"{io_args['output_masked_json_dir']}/mask.json")

data = json.load(f)

print(len(data['color_dict'][0]['color_label']))
f.close()
color_label = []
color_count = []
json_dict = {}
color_dict = []
frame_count = 0
width = data['width']
height = data['height']
total_frame = len(data['color_dict'])
for i in range(total_frame):
    pred_mask = []
    for idx in range(len(data['color_dict'][i]['color_label'])):
        d = data['color_dict'][i]
        pred_mask += ([d['color_label'][idx]] * d['color_count'][idx])
    #print(pred_mask)
    pred_mask = np.array(pred_mask)
    pred_mask = np.reshape(pred_mask, (-1, width))
    save_prediction(pred_mask, f"{io_args['output_mask_dir']}", str(i).zfill(5) + '.png')
    print('json {}/{} writed'.format(i+1,total_frame),end='\r')
print('')
draw_video = True
input_video = f'U:\\sat\\Segment-and-Track-Anything\\assets\\{video_name}.mp4'
if draw_video:
    cap = cv2.VideoCapture(input_video)
    # if frame_num > 0:
    #     for i in range(0, frame_num):
    #         cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc =  cv2.VideoWriter_fourcc(*"mp4v")
    # if input_video[-3:]=='mp4':
    #     fourcc =  cv2.VideoWriter_fourcc(*"mp4v")
    # elif input_video[-3:] == 'avi':
    #     fourcc =  cv2.VideoWriter_fourcc(*"MJPG")
    #     # fourcc = cv2.VideoWriter_fourcc(*"XVID")
    # else:
    #     fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    name_list = io_args['output_video'].split('.') [:-1]
    part_i = 1
    #io_args['output_video'] = ''.join(name_list)+f'_part_{part_i}.mp4'
    print(f"This is output video {io_args['output_video']}")
    out = cv2.VideoWriter(io_args['output_video'], fourcc, fps, (width, height))
    out_block = cv2.VideoWriter(io_args['output_video_block'], fourcc, fps, (width, height))
    #partial video


    frame_idx = 0
    #with open(f"{io_args['output_masked_frame_dir']}/{str(frame_idx).zfill(5)}.json", 'w') as f:
    #    json.dump(data, f)
    color_label = []
    color_count = []
    json_dict = {}
    color_dict = []
    frame_idx = 0
    width = data['width']
    height = data['height']
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        #return from json
        pred_mask = []
        for idx in range(len(data['color_dict'][frame_idx]['color_label'])):
            d = data['color_dict'][frame_idx]
            pred_mask += ([d['color_label'][idx]] * d['color_count'][idx])
        #print(pred_mask)
        pred_mask = np.array(pred_mask)
        pred_mask = np.reshape(pred_mask, (-1, width))
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        masked_frame = draw_mask(frame, pred_mask)
        block_frame = draw_block(frame, pred_mask, width, height)
        cv2.imwrite(f"{io_args['output_masked_frame_dir']}/{str(frame_idx).zfill(5)}.png", masked_frame[:, :, ::-1])
        #print(len(pred_mask))
        #print(len(pred_mask[0]))
        for i in pred_mask:
            for j in i:
                if len(color_label) == 0:
                    color_label.append(int(j))
                    color_count.append(0)
                elif j != color_label[-1]:
                    color_label.append(int(j))
                    color_count.append(0)
                color_count[-1] += 1
        #print(color_label)
        #print(color_count)
        color_dict.append({'color_label':color_label, 'color_count':color_count})
        #masked_pred_list.append(masked_frame)
        masked_frame = cv2.cvtColor(masked_frame,cv2.COLOR_RGB2BGR)
        block_frame = cv2.cvtColor(block_frame,cv2.COLOR_RGB2BGR)
        out.write(masked_frame)
        out_block.write(block_frame)
        print('frame {}/{} writed'.format(frame_idx+1,total_frame),end='\r')
        frame_idx += 1
        #if frame_idx % 900 == 0:
        #    out.release()
        #    part_i += 1
        #    io_args['output_video'] = ''.join(name_list)+f'_part_{part_i}.mp4'
        #    print(f"This is output video {io_args['output_video']}")
        #    out = cv2.VideoWriter(io_args['output_video'], fourcc, fps, (width, height))

    out.release()
    out_block.release()
    cap.release()
    print("\n{} saved".format(io_args['output_video']))
    print('\nfinished')

    # save colorized masks as a gif
    # imageio.mimsave(io_args['output_gif'], masked_pred_list, fps=fps)
    # print("{} saved".format(io_args['output_gif']))


    gc.collect()
