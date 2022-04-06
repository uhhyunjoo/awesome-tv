from utils import *
from model import *
from datasets import *

import wandb
import argparse

def main():
    # OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='7' python datasets_sample.py
    ucf_folder = '/data5/datasets/ucf101'
    output_folder = './output/test_datasets'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    lines = ['ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi 1',
            'Nunchucks/v_Nunchucks_g25_c06.avi 56',
            'ParallelBars/v_ParallelBars_g08_c01.avi 57',
            'Skiing/v_Skiing_g25_c05.avi 81',
            'Skijet/v_Skijet_g08_c01.avi 82',
            'YoYo/v_YoYo_g25_c05.avi 101']
    video_list = []
    label_list = []
    for line in lines:
        video_name, label_name = line.split()
        video_list.append(video_name)
        label_list.append(int(label_name)-1) # to make label list in range(0,101)
    
    for idx in range(len(lines)):
        
        video_path = Path(ucf_folder) / 'video' / video_list[idx]
        # video_path = os.path.join(ucf_folder, 'video/{}'.format(video_list[idx]))
        label = np.array(label_list[idx])
        buffer = sample_uniform_load_resized_frames(video_path = str(video_path)) # 0 ~ 255 의 값
        buffer = sample_random_crop(buffer) # 0 ~ 255 의 값
        # shape sum min max

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv.flip(buffer[i], flipCode=1) # (h,w,c)
                buffer[i] = frame
        buffer = buffer.astype(np.float32)
        print(buffer.shape, buffer.sum(), buffer.min(), buffer.max(), np.average(buffer))
        # (16, 112, 112, 3) 52937470.0 1.0 255.0 87.91964
        # (16, 112, 112, 3) 58764090.0 0.0 255.0 97.5966

        # 53353404.0
        normalize_type = 'imagenet'
        # normalize_type = 'imagenet'
        if normalize_type == 'imagenet':
            for i, frame in enumerate(buffer):
                frame = frame / 255.0 # (h,w,c)
                buffer[i] = frame
            print(buffer.shape, buffer.sum(), buffer.min(), buffer.max(), np.average(buffer))
            # (16, 112, 112, 3) 246510.53 0.0 0.99607843

            buffer = buffer.transpose((3, 0, 1, 2)) # (3, 32, 112, 112)
            print(buffer.shape, buffer.sum(), buffer.min(), buffer.max(), np.average(buffer))
            # (3, 16, 112, 112) 246510.53 0.0 0.99607843

            # 'bgr'
            mean = [0.406, 0.456, 0.485]
            std = [0.225, 0.224, 0.229]

            torch_buffer = torch.from_numpy(buffer)

            # import pdb;pdb.set_trace()
            for i, channel in enumerate(torch_buffer):
                normalized_channel = (channel - mean[i])/std[i]
                torch_buffer[i] = normalized_channel
            print(torch_buffer.shape, torch_buffer.sum(), torch_buffer.min(), torch_buffer.max(), torch.mean(torch_buffer))
            # torch.Size([3, 16, 112, 112]) tensor(-105382.6562) tensor(-2.1179) tensor(2.6226)

            out_buffer = torch_buffer
            out_label = torch.from_numpy(label)
            # torch.Size([3, 16, 112, 112]) torch.Size([])
            # out_label : tensor(0)

            print(out_buffer.shape, out_label.shape)
        elif normalize_type =='ucf101':
            # 'bgr'
            mean = [90.0, 98.0, 102.0]

            for i, frame in enumerate(buffer):
                frame = frame.transpose((2,0,1))
                # (c, h, w)
                for j, channel in enumerate(frame):
                    normalized_channel = (channel - mean[j])
                    frame[j] = normalized_channel
                frame = frame.transpose((1,2,0))
                # (h, w, c)
                frame = frame / 255.0
                buffer[i] = frame
            print(buffer.shape, buffer.sum(), buffer.min(), buffer.max(), np.average(buffer))
            # (16, 112, 112, 3) 207597.94 0.003921569 1.0 0.34478292

            buffer = buffer.transpose((3, 0, 1, 2)) # (3, 32, 112, 112)
            print(buffer.shape, buffer.sum(), buffer.min(), buffer.max(), np.average(buffer))
            # (3, 16, 112, 112) 207597.94 0.003921569 1.0 0.34478292

            torch_buffer = torch.from_numpy(buffer)
            torch_label = torch.from_numpy(label)
            # torch.Size([3, 16, 112, 112]) tensor(-277394.5938) tensor(-2.1008) tensor(2.6400) tensor(-0.4607)
            # out_label : tensor(0)

            print(torch_buffer.shape, torch_label.shape)

def sample_uniform_load_resized_frames(video_path, clip_len = 16, resize_width = 171, resize_height = 128):
        cap = cv.VideoCapture(video_path)
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)

        frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        # frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        # frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

        # EXTRACT_FREQUENCY = 1

        unifrom_indices = np.linspace(0, frame_count, clip_len)
        unifrom_indices = unifrom_indices.astype('int32')
        count, i, retaining = 0, 0, True
        video_frames = []
        while count < frame_count and retaining:
            retaining, frame = cap.read()
            if frame is None:
                continue
            if count in unifrom_indices :
                frame = cv.resize(frame, (resize_width, resize_height))
                video_frames.append(frame)
            count += 1
        cap.release()


        video_frames = np.asarray(video_frames, dtype=np.uint8) # np.float32
        return video_frames

def sample_random_crop(buffer, resize_width = 171, resize_height = 128, crop_size = 112, clip_len= 16):
    # input : (t, h, w, c)

    # random_h_idx = math.floor((buffer.shape[1] - self.crop_size) / 2)
    # random_w_idx = math.floor((buffer.shape[2] - self.crop_size) / 2)

    rand_w_idx = random.randrange(0, resize_width - crop_size + 1)
    rand_h_idx = random.randrange(0, resize_height - crop_size + 1)
    # rand_l_idx = random.randrange(0, (buffer.shape)[0] - self.clip_len + 1)

    #  buffer = buffer[0: self.clip_len,
    buffer = buffer[:,
                    rand_h_idx: rand_h_idx + crop_size,
                    rand_w_idx: rand_w_idx + crop_size,
                    :]

    if buffer.shape[0] < clip_len:
        repeated = clip_len // buffer.shape[0] - 1
        remainder = clip_len % buffer.shape[0]
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
if __name__ == "__main__":
    main()
