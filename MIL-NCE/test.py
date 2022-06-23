import json
import os
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=7 python test.py

json_file = '/data5/datasets/MSRVTT/annotation/MSR_VTT.json'
video_folder = '/data5/datasets/MSRVTT/videos/all'
split = 'train'


with open(json_file, 'r') as f:
    data = json.load(f)
f.close()

image_id_list = []
id_list = []

for d in data['annotations']:
    image_id_list.append(d['image_id'])
    id_list.append(d['id'])
import pdb;pdb.set_trace()
for i in range(200000):
    if image_id_list.count('video{}'.format(i)) != 5:
        print(i)

print('end')

