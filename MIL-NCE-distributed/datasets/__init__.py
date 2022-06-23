from utils import *
import torchvision.transforms as transforms
from torch.utils.data.dataloader import default_collate
import math
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import random
import ffmpeg
import time
import re
import pickle

class MSRVTT_Dataset(Dataset):
    """MSRVTT Video-Text loader."""

    def __init__(
            self,
            csv,
            video_root='/data5/datasets/MSRVTT/videos/all',
            num_clip=4,
            fps=16,
            num_frames=32,
            size=224,
            crop_only=False,
            center_crop=True,
            token_to_word_path='/home/hjlee/workspace/Github/awesome-tv/MIL-NCE/datasets/data/dict.npy',
            max_words=30
    ):
        """
        Args:
        """
        assert isinstance(size, int)
        self.csv = pd.read_csv(csv)
        # self.csv = pd.read_csv(csv).iloc[:80, :] ##############################
        self.video_root = video_root
        self.size = size
        self.num_frames = num_frames
        self.fps = fps
        self.num_clip = num_clip
        self.num_sec = self.num_frames / float(self.fps)
        self.crop_only = crop_only
        self.center_crop = center_crop
        self.max_words = max_words
        self.word_to_token = {}
        token_to_word = np.load(token_to_word_path)
        for i, t in enumerate(token_to_word):
            self.word_to_token[t] = i + 1


    def __len__(self):
        return len(self.csv)

    def _get_msrvtt_video(self, video_path):
        #print(">>> ", video_path)
        buffer = self._load_resized_frames(video_path)
        # print('load :', buffer.shape) load : (300, 224, 224, 3)
        # buffer = self._center_crop(buffer)
        # 아 resize 한 다음에 224로 crop 하는 건가?
        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv.flip(buffer[i], flipCode=1) # (h,w,c)
                buffer[i] = frame
        
        buffer = buffer.astype(np.float32)
        # (480, 224, 224, 3)
        
        for i, frame in enumerate(buffer):
            frame = frame / 255.0 # (h,w,c)
            buffer[i] = frame

        buffer = buffer.transpose((3, 0, 1, 2))
        # (3, 480, 224, 224)
        torch_buffer = torch.from_numpy(buffer)
        # (3, 480, 224, 224)

        mean = [0.406, 0.456, 0.485]
        std = [0.225, 0.224, 0.229]

        for i, channel in enumerate(torch_buffer):
            normalized_channel = (channel - mean[i])/std[i]
            torch_buffer[i] = normalized_channel
        # torch.Size([3, 480, 224, 224])
        torch_buffer = torch_buffer.permute((1,0,2,3))

        # torch.Size([480, 3, 224, 224])

        uniform_indicies = np.linspace(0, torch_buffer.shape[0] - 1, self.num_frames)
        uniform_indicies = uniform_indicies.astype('int32')
        clip_buffer = torch.zeros(self.num_frames, torch_buffer.shape[1], self.size, self.size)
        # torch.Size([32, 3, 224, 224])

        for idx, torch_idx in enumerate(uniform_indicies):
            clip_buffer[idx] = torch_buffer[torch_idx]
        # torch.Size([32, 3, 224, 224])
        clip_buffer = clip_buffer.permute((1,0,2,3))
        # # torch.Size([3, 32, 224, 224])

        # # start_indices = np.linspace(0, len(clip_torch_buffer) - self.num_frames, self.num_clip)
        # # start_indices = start_indices.astype('int32') # index 값 4개

        # video = torch.zeros(num_clip, 3, self.num_frames, self.size, self.size)
        # # torch.Size([4, 3, 32, 224, 224])
        # # 이 부분에서 이제 uniform_indices 를 이용해서 비디오 불러오면 됨
        
        # for i in range(self.num_clip):
        #     video[i] = clip_buffer[:, i*self.num_frames: (i+1)*self.num_frames, :, :] # 32, 3, 224, 224
        #     # video[i] = clip_buffer[start_idx: start_idx + self.num_clip]
        # #print(video.shape)
        return clip_buffer

    def _load_resized_frames(self, video_path):
        cap = cv.VideoCapture(video_path)
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)

        frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        # frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        # frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

        # EXTRACT_FREQUENCY = 1

        #unifrom_indices = np.linspace(0, frame_count, self.num_frames)
        #unifrom_indices = unifrom_indices.astype('int32')
        count, i, retaining = 0, 0, True
        video_frames = []
        while count < frame_count and retaining:
            retaining, frame = cap.read()
            if frame is None:
                continue
            # if count in unifrom_indices :
            #     frame = cv.resize(frame, (self.size, self.size))
            #     video_frames.append(frame)
            frame = cv.resize(frame, (self.size, self.size))
            video_frames.append(frame)
            count += 1
        cap.release()

        video_frames = np.asarray(video_frames, dtype=np.uint8) # np.float32
        return video_frames

    def _center_crop(self, buffer):
        # input : (t, h, w, c)
        print('in crop: ', buffer.shape)
        try:
            h_idx = math.floor((buffer.shape[1] - self.size) / 2)
            w_idx = math.floor((buffer.shape[2] - self.size) / 2)

            buffer = buffer[:,
                            h_idx: h_idx + self.size,
                            w_idx: w_idx + self.size,
                            :]
        except:
            print('buffer:', buffer.shape)
        
        if buffer.shape[0] < self.num_frames:
            repeated = self.num_frames // buffer.shape[0] - 1
            remainder = self.num_frames % buffer.shape[0]
            buffered, reverse = buffer, True
            if repeated > 0:
                padded = []
                for _ in range(repeated):
                    if reverse:
                        pad = buffer[::-1, :, :, :]
                        reverse = False
                    else:
                        pad = buffer
                        reverse = True
                    padded.append(pad)
                padded = np.concatenate(padded, axis=0)
                buffer = np.concatenate((buffer, padded), axis=0)
            if reverse:
                pad = buffered[::-1, :, :, :][:remainder, :, :, :]
            else:
                pad = buffered[:remainder, :, :, :]
            buffer = np.concatenate((buffer, pad), axis=0)
        return buffer

    def _random_crop(self, buffer):
        # input : (t, h, w, c)

        # random_h_idx = math.floor((buffer.shape[1] - self.crop_size) / 2)
        # random_w_idx = math.floor((buffer.shape[2] - self.crop_size) / 2)

        rand_w_idx = random.randrange(0, self.resize_width - self.crop_size + 1)
        rand_h_idx = random.randrange(0, self.resize_height - self.crop_size + 1)
        # rand_l_idx = random.randrange(0, (buffer.shape)[0] - self.clip_len + 1)

        #  buffer = buffer[0: self.clip_len,
        buffer = buffer[:,
                        rand_h_idx: rand_h_idx + self.crop_size,
                        rand_w_idx: rand_w_idx + self.crop_size,
                        :]

        if buffer.shape[0] < self.clip_len:
            repeated = self.clip_len // buffer.shape[0] - 1
            remainder = self.clip_len % buffer.shape[0]
            buffered, reverse = buffer, True
            if repeated > 0:
                padded = []
                for _ in range(repeated):
                    if reverse:
                        pad = buffer[::-1, :, :, :]
                        reverse = False
                    else:
                        pad = buffer
                        reverse = True
                    padded.append(pad)
                padded = np.concatenate(padded, axis=0)
                buffer = np.concatenate((buffer, padded), axis=0)
            if reverse:
                pad = buffered[::-1, :, :, :][:remainder, :, :, :]
            else:
                pad = buffered[:remainder, :, :, :]
            buffer = np.concatenate((buffer, pad), axis=0)
        return buffer


    def _get_video_start(self, video_path, start):
        start_seek = start
        cmd = (
            ffmpeg
            .input(video_path, ss=start_seek, t=self.num_sec + 0.1)
            .filter('fps', fps=self.fps)
        )
        if self.center_crop:
            aw, ah = 0.5, 0.5
        else:
            aw, ah = random.uniform(0, 1), random.uniform(0, 1)
        if self.crop_only:
            cmd = (
                cmd.crop('(iw - {})*{}'.format(self.size, aw),
                         '(ih - {})*{}'.format(self.size, ah),
                         str(self.size), str(self.size))
            )
        else:
            cmd = (
                cmd.crop('(iw - min(iw,ih))*{}'.format(aw),
                         '(ih - min(iw,ih))*{}'.format(ah),
                         'min(iw,ih)',
                         'min(iw,ih)')
                .filter('scale', self.size, self.size)
            )
        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, quiet=True)
        )
        video = np.frombuffer(out, np.uint8).reshape([-1, self.size, self.size, 3])
        video = torch.from_numpy(video)
        video = video.permute(3, 0, 1, 2)
        if video.shape[1] < self.num_frames:
            zeros = torch.zeros((3, self.num_frames - video.shape[1], self.size, self.size), dtype=torch.uint8)
            video = torch.cat((video, zeros), axis=1)
        return video[:, :self.num_frames]

    def _split_text(self, sentence):
        w = re.findall(r"[\w']+", str(sentence))
        return w

    def _words_to_token(self, words):
        words = [self.word_to_token[word] for word in words if word in self.word_to_token]
        if words:
            we = self._zero_pad_tensor_token(torch.LongTensor(words), self.max_words)
            return we
        else:
            return torch.zeros(self.max_words, dtype=torch.long)

    def _zero_pad_tensor_token(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = torch.zeros(size - len(tensor)).long()
            return torch.cat((tensor, zero), dim=0)

    def words_to_ids(self, x):
        return self._words_to_token(self._split_text(x))

    # msrvtt
    # def _get_duration(self, video_path):
    #     probe = ffmpeg.probe(video_path)
    #     return probe['format']['duration']


    # msrvtt
    def __getitem__(self, idx):
        # import pdb;pdb.set_trace()
        video_id = self.csv['video'].values[idx]
        caption = self.csv['text'].values[idx] # 1 문장
        video_path = os.path.join(self.video_root, video_id + '.mp4')
        # duration = self._get_duration(video_path)
        text = self.words_to_ids(caption)
        # video = self._get_video(video_path, 0, float(duration), self.num_clip)
        video = self._get_msrvtt_video(video_path)
        # video : torch.Size([4, 3, 32, 224, 224])
        # text : torch.Size([30])
        #print(idx, video.shape, text.shape)
        return video, text

def collate_wrapper(batch):
    # batch = list(filter(lambda x: x[0] is not None, batch))
    # print(len(batch))
    return default_collate(batch)
