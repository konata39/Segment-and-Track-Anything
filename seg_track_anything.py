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
from scipy.spatial.distance import cdist

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

def dfs(x, y, value):
    if x < 0 or x >= rows or y < 0 or y >= cols or mask[x, y] != value or groups[x, y] != 0:
        return
    groups[x, y] = group_id
    group_counts[group_id] += 1
    for dx, dy in adjacent:
        dfs(x + dx, y + dy, value)

def chack_group_distance(mask):
    rows, cols = mask.shape
    group_id = 1
    groups = np.zeros_like(mask)
    group_counts = {}
    adjacent = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]  # 8方向鄰居
    def dfs_iterative(x, y, value):
        stack = [(x, y)]
        while stack:
            x, y = stack.pop()
            if x < 0 or x >= rows or y < 0 or y >= cols or mask[x, y] != value or groups[x, y] != 0:
                continue
            groups[x, y] = group_id
            group_counts[group_id] += 1
            for dx, dy in adjacent:
                stack.append((x + dx, y + dy))

    for i in range(rows):
        for j in range(cols):
            if mask[i, j] != 0 and groups[i, j] == 0:
                group_counts[group_id] = 0
                dfs_iterative(i, j, mask[i, j])
                group_id += 1

    value_groups = {}
    for gid, count in group_counts.items():
        value = mask[groups == gid][0]
        if value not in value_groups:
            value_groups[value] = []
        value_groups[value].append(gid)

    disconnected_pairs = []
    distances = {}

    for value, gids in value_groups.items():
        if len(gids) > 1:
            for i in range(len(gids) - 1):
                for j in range(i + 1, len(gids)):
                    group_i_points = np.argwhere(groups == gids[i])
                    group_j_points = np.argwhere(groups == gids[j])
                    max_distance = np.min(cdist(group_i_points, group_j_points, 'euclidean'))
                    if value not in distances:
                        distances[value] = -2147483647
                    if max_distance > distances[value]:
                        distances[value] = max_distance
                    disconnected_pairs.append((gids[i], gids[j]))

    group_centers = {}

    for gid in value_groups.items():
        group_points = np.argwhere(mask == gid[0])
        group_points = tuple(tuple(sub) for sub in group_points)
        center_x, center_y = np.mean(group_points, axis=0)
        group_centers[gid[0]] = (center_x, center_y)
    return distances, group_centers


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

def create_dir(dir_path):
    # if os.path.isdir(dir_path):
    #     os.system(f"rm -r {dir_path}")

    # os.makedirs(dir_path)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)



aot_model2ckpt = {
    "deaotb": "./ckpt/DeAOTB_PRE_YTB_DAV.pth",
    "deaotl": "./ckpt/DeAOTL_PRE_YTB_DAV",
    "r50_deaotl": "./ckpt/R50_DeAOTL_PRE_YTB_DAV.pth",
}


def tracking_objects_in_video(SegTracker, input_video, input_img_seq, fps, frame_num=0, delete_dir=True, output_image=True, detect_check=False):

    if input_video is not None:
        video_name = os.path.basename(input_video).split('.')[0]
    elif input_img_seq is not None:
        file_name = input_img_seq.name.split('/')[-1].split('.')[0]
        file_path = f'./assets/{file_name}'
        imgs_path = sorted([os.path.join(file_path, img_name) for img_name in os.listdir(file_path)])
        video_name = file_name
    else:
        return None, None

    # create dir to save result
    tracking_result_dir = f'{os.path.join(os.path.dirname(__file__), "tracking_results", f"{video_name}")}'
    create_dir(tracking_result_dir)

    io_args = {
        'tracking_result_dir': tracking_result_dir,
        'output_mask_dir': f'{tracking_result_dir}/{video_name}_masks',
        'output_masked_frame_dir': f'{tracking_result_dir}/{video_name}_masked_frames',
        'output_masked_json_dir': f'{tracking_result_dir}/{video_name}_json_frames',
        'output_video': f'{tracking_result_dir}/{video_name}_seg.mp4', # keep same format as input video
        'output_gif': f'{tracking_result_dir}/{video_name}_seg.gif',
    }

    if input_video is not None:
        return video_type_input_tracking(SegTracker, input_video, io_args, video_name, frame_num, delete_dir=delete_dir, output_image=output_image, detect_check=detect_check)
    elif input_img_seq is not None:
        return img_seq_type_input_tracking(SegTracker, io_args, video_name, imgs_path, fps, frame_num, delete_dir=delete_dir, output_image=output_image)


def video_type_input_tracking(SegTracker, input_video, io_args, video_name, frame_num=0, delete_dir=True, output_image=True, detect_check=False):

    pred_list = []
    masked_pred_list = []

    # source video to segment
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if frame_num > 0:
        output_mask_name = sorted([img_name for img_name in os.listdir(io_args['output_mask_dir'])])
        output_masked_frame_name = sorted([img_name for img_name in os.listdir(io_args['output_masked_frame_dir'])])

        for i in range(0, frame_num):
            cap.read()
            pred_list.append(np.array(Image.open(os.path.join(io_args['output_mask_dir'], output_mask_name[i])).convert('P')))
            masked_pred_list.append(cv2.imread(os.path.join(io_args['output_masked_frame_dir'], output_masked_frame_name[i])))


    # create dir to save predicted mask and masked frame
    if frame_num == 0:
        if delete_dir:
            if os.path.isdir(io_args['output_mask_dir']):
                os.system(f"rm -r {io_args['output_mask_dir']}")
            if os.path.isdir(io_args['output_masked_frame_dir']):
                os.system(f"rm -r {io_args['output_masked_frame_dir']}")
            if os.path.isdir(io_args['output_masked_json_dir']):
                os.system(f"rm -r {io_args['output_masked_json_dir']}")
    output_mask_dir = io_args['output_mask_dir']
    create_dir(io_args['output_mask_dir'])
    create_dir(io_args['output_masked_frame_dir'])
    create_dir(io_args['output_masked_json_dir'])

    torch.cuda.empty_cache()
    gc.collect()
    sam_gap = SegTracker.sam_gap
    frame_idx = 0
    prev_center_dict = None
    with torch.cuda.amp.autocast():
        while cap.isOpened():
            ret, frame  = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            if frame_idx == 0:
                pred_mask = SegTracker.first_frame_mask
                torch.cuda.empty_cache()
                gc.collect()
            elif (frame_idx % sam_gap) == 0:
                seg_mask = SegTracker.seg(frame)
                torch.cuda.empty_cache()
                gc.collect()
                track_mask = SegTracker.track(frame)
                # find new objects, and update tracker with new objects
                new_obj_mask = SegTracker.find_new_objs(track_mask,seg_mask)
                save_prediction(new_obj_mask, output_mask_dir, str(frame_idx+frame_num).zfill(5) + '_new.png')
                pred_mask = track_mask + new_obj_mask
                # segtracker.restart_tracker()
                print("video_type_input_tracking")
                SegTracker.add_reference(frame, pred_mask)
            else:
                pred_mask = SegTracker.track(frame,update_memory=True)
            torch.cuda.empty_cache()
            gc.collect()
            if output_image:
                save_prediction(pred_mask, output_mask_dir, str(frame_idx + frame_num).zfill(5) + '.png')
            pred_list.append(pred_mask)

            print("processed frame {}, obj_num {}".format(frame_idx + frame_num, SegTracker.get_obj_num()),end='\r')
            frame_idx += 1
            if detect_check == True:
                distance_dict, centers_dict = chack_group_distance(pred_mask)
                current_displacements = dict()
                if prev_center_dict != None:
                    for value, center in centers_dict.items():
                        if value in prev_center_dict:
                            displacement = np.sqrt((center[0] - prev_center_dict[value][0])**2 + (center[1] - prev_center_dict[value][1])**2)
                            current_displacements[value] = displacement
                dist_threshold_list = [0, 0, 0, 0, 0, 0]
                for value, dist in distance_dict.items():
                    dist_threshold_list[value-1] = 1 if dist > 250 else 0
                for value, dist in current_displacements.items():
                    dist_threshold_list[value-1] = 2 if dist > 100 else 0
                if 1 in dist_threshold_list:
                    print("\nDetect abnomal apart part from Frame {}, stop tracking.".format(frame_idx + frame_num))
                    #print(centers_dict)
                    #print(distance_dict)
                    break
                if 2 in dist_threshold_list:
                    print("\nDetect abnomal tracking distance from Frame {}, stop tracking.".format(frame_idx + frame_num))
                    break
                prev_center_dict = centers_dict

            #if mask part is separate, break
        cap.release()
        print('\nfinished')

    ##################
    # Visualization
    ##################

    # draw pred mask on frame and save as a video
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
    #partial video


    output_frame_idx = 0
    #with open(f"{io_args['output_masked_frame_dir']}/{str(frame_idx).zfill(5)}.json", 'w') as f:
    #    json.dump(data, f)
    color_label = []
    color_count = []
    json_dict = {}
    color_dict = []
    frame_count = 0
    json_dict['width'] = width
    json_dict['height'] = height
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        color_label = []
        color_count = []
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        pred_mask = pred_list[output_frame_idx]
        masked_frame = draw_mask(frame, pred_mask)
        if output_image:
            cv2.imwrite(f"{io_args['output_masked_frame_dir']}/{str(output_frame_idx).zfill(5)}.png", masked_frame[:, :, ::-1])
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
        #masked_frame = cv2.cvtColor(masked_frame,cv2.COLOR_RGB2BGR)
        out.write(masked_frame)

        print('frame {} writed'.format(output_frame_idx),end='\r')
        output_frame_idx += 1
        if output_frame_idx >= frame_idx:
            break
        #if frame_idx % 900 == 0:
        #    out.release()
        #    part_i += 1
        #    io_args['output_video'] = ''.join(name_list)+f'_part_{part_i}.mp4'
        #    print(f"This is output video {io_args['output_video']}")
        #    out = cv2.VideoWriter(io_args['output_video'], fourcc, fps, (width, height))
    json_dict['color_dict'] = color_dict
    with open(f"{io_args['output_masked_json_dir']}/mask.json", 'w') as f:
        json.dump(json_dict, f)
    #out.release()
    cap.release()
    print("\n{} saved".format(io_args['output_video']))
    print('\nfinished')

    # save colorized masks as a gif
    # imageio.mimsave(io_args['output_gif'], masked_pred_list, fps=fps)
    # print("{} saved".format(io_args['output_gif']))

    # zip predicted mask(disable zip output for saving time)
    #if os.name == 'posix':
    #    os.system(f"zip -r {io_args['tracking_result_dir']}/{video_name}_pred_mask.zip {io_args['output_mask_dir']}")
    #elif os.name == "nt":
    #    os.system(f"7z a -tzip {io_args['tracking_result_dir']}/{video_name}_pred_mask.zip {io_args['output_mask_dir']}")
    # manually release memory (after cuda out of memory)
    del SegTracker
    torch.cuda.empty_cache()
    gc.collect()

    return io_args['output_video'], f"{io_args['tracking_result_dir']}/{video_name}_pred_mask.zip"


def img_seq_type_input_tracking(SegTracker, io_args, video_name, imgs_path, fps, frame_num=0, delete_dir=True, output_image=True):

    pred_list = []
    masked_pred_list = []

    if frame_num > 0:
        output_mask_name = sorted([img_name for img_name in os.listdir(io_args['output_mask_dir'])])
        output_masked_frame_name = sorted([img_name for img_name in os.listdir(io_args['output_masked_frame_dir'])])
        for i in range(0, frame_num):
            pred_list.append(np.array(Image.open(os.path.join(io_args['output_mask_dir'], output_mask_name[i])).convert('P')))
            masked_pred_list.append(cv2.imread(os.path.join(io_args['output_masked_frame_dir'], output_masked_frame_name[i])))

    # create dir to save predicted mask and masked frame
    if frame_num == 0:
        if delete_dir:
            if os.path.isdir(io_args['output_mask_dir']):
                os.system(f"rm -r {io_args['output_mask_dir']}")
            if os.path.isdir(io_args['output_masked_frame_dir']):
                os.system(f"rm -r {io_args['output_masked_frame_dir']}")
            if os.path.isdir(io_args['output_masked_json_dir']):
                os.system(f"rm -r {io_args['output_masked_json_dir']}")

    output_mask_dir = io_args['output_mask_dir']
    create_dir(io_args['output_mask_dir'])
    create_dir(io_args['output_masked_frame_dir'])


    i_frame_num = frame_num

    torch.cuda.empty_cache()
    gc.collect()
    sam_gap = SegTracker.sam_gap
    frame_idx = 0

    with torch.cuda.amp.autocast():
        for img_path in imgs_path:
            if i_frame_num > 0:
                i_frame_num = i_frame_num - 1
                continue

            frame_name = os.path.basename(img_path).split('.')[0]
            frame = cv2.imread(img_path)
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            if frame_idx == 0:
                pred_mask = SegTracker.first_frame_mask
                torch.cuda.empty_cache()
                gc.collect()
            elif (frame_idx % sam_gap) == 0:
                seg_mask = SegTracker.seg(frame)
                torch.cuda.empty_cache()
                gc.collect()
                track_mask = SegTracker.track(frame)
                # find new objects, and update tracker with new objects
                new_obj_mask = SegTracker.find_new_objs(track_mask,seg_mask)
                save_prediction(new_obj_mask, output_mask_dir, f'{frame_name}_new.png')
                pred_mask = track_mask + new_obj_mask
                # segtracker.restart_tracker()
                print("img_seq_type_input_tracking")
                SegTracker.add_reference(frame, pred_mask)
            else:
                pred_mask = SegTracker.track(frame,update_memory=True)
            torch.cuda.empty_cache()
            gc.collect()

            save_prediction(pred_mask, output_mask_dir, f'{frame_name}.png')
            pred_list.append(pred_mask)

            print("processed frame {}, obj_num {}".format(frame_idx+frame_num, SegTracker.get_obj_num()),end='\r')
            frame_idx += 1
        print('\nfinished')

    ##################
    # Visualization
    ##################

    # draw pred mask on frame and save as a video
    height, width = pred_list[0].shape
    fourcc =  cv2.VideoWriter_fourcc(*"mp4v")
    i_frame_num =frame_num

    out = cv2.VideoWriter(io_args['output_video'], fourcc, fps, (width, height))

    frame_idx = 0
    for img_path in imgs_path:
        # if i_frame_num > 0:
        #     i_frame_num = i_frame_num - 1
        #     continue
        frame_name = os.path.basename(img_path).split('.')[0]
        frame = cv2.imread(img_path)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        pred_mask = pred_list[frame_idx]
        masked_frame = draw_mask(frame, pred_mask)
        masked_pred_list.append(masked_frame)
        cv2.imwrite(f"{io_args['output_masked_frame_dir']}/{frame_name}.png", masked_frame[:, :, ::-1])

        masked_frame = cv2.cvtColor(masked_frame,cv2.COLOR_RGB2BGR)
        out.write(masked_frame)
        print('frame {} writed'.format(frame_name),end='\r')
        frame_idx += 1
    out.release()
    print("\n{} saved".format(io_args['output_video']))
    print('\nfinished')

    # save colorized masks as a gif
    imageio.mimsave(io_args['output_gif'], masked_pred_list, fps=fps)
    print("{} saved".format(io_args['output_gif']))

    # zip predicted mask
    os.system(f"zip -r {io_args['tracking_result_dir']}/{video_name}_pred_mask.zip {io_args['output_mask_dir']}")

    # manually release memory (after cuda out of memory)
    del SegTracker
    torch.cuda.empty_cache()
    gc.collect()


    return io_args['output_video'], f"{io_args['tracking_result_dir']}/{video_name}_pred_mask.zip"
