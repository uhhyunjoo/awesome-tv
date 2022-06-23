from utils import *

import pandas as pd
import numpy as np
import json

# save train / test csv files
data_folder = '/home/hjlee/workspace/data5_custom_hjlee/msrvtt_data'
video_folder = '/data5/datasets/MSRVTT/videos/all'

split = 'train'

json_path = Path(data_folder) / 'MSRVTT_data.json'
with open(json_path, 'r') as f:
    json_data = json.load(f)
f.close()

annotations_df = pd.DataFrame.from_dict(json_data['sentences'])
import pdb;pdb.set_trace()
split == 'train'
train_csv_path = Path(data_folder) / 'MSRVTT_train.9k.csv'
train_csv_data = pd.read_csv(train_csv_path)
train_video_id_list = train_csv_data['video_id'].values.tolist()
train_df = annotations_df[annotations_df['video_id'].isin(train_video_id_list)]
video_list = train_df['video_id'].values.tolist()
text_list = train_df['caption'].values.tolist()

train_dict = {'video' : video_list, 'text': text_list}
df_train = pd.DataFrame(train_dict)
df_train.to_csv('./msrvtt_train.csv', index = False)


split == 'test'
test_csv_path = Path(data_folder) / 'MSRVTT_JSFUSION_test.csv'
test_csv_data = pd.read_csv(test_csv_path)
video_list = test_csv_data['video_id'].values.tolist()
text_list = test_csv_data['sentence'].values.tolist()

test_dict = {'video' : video_list, 'text': text_list}
df_test = pd.DataFrame(test_dict)
df_test.to_csv('./msrvtt_test.csv', index = False)
