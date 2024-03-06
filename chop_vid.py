import cv2
import os
str_list = ['cars']
sample_rate = 1
for str_i in str_list:
#    if i == 3:
#        continue
#    str_i = str(i)
#    if i < 10:
#        str_i = '0' + str_i
    cwd = os.getcwd()
    vidcap = cv2.VideoCapture(f'{cwd}/assets/{str_i}.mp4')
    if not os.path.isdir(f'{cwd}/assets/output_{str_i}'):
        os.mkdir(f'{cwd}/assets/output_{str_i}')
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    success,image = vidcap.read()
    count = 0
    print(f"extracting video {str_i}...")
    while success:
      if count % sample_rate == 0:
        output_str = str(count)
        while len(output_str) < 5:
            output_str = '0' + output_str
        cv2.imwrite(f"{cwd}/assets/output_{str_i}/img{output_str}.png", image)     # save frame as JPEG file
      success,image = vidcap.read()
      count += 1
print('all video complete.')
os.system("pause")
