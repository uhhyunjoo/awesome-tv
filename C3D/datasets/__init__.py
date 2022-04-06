from utils import *
import torchvision.transforms as transforms
from torch.utils.data.dataloader import default_collate
import math

class UCF101Dataset(Dataset):
    def __init__(self, ucf_folder, split, split_num, resize_width, resize_height, crop_size, clip_len, sampling, normalize_type):
        super(UCF101Dataset, self)
        self.video_folder = Path(ucf_folder) / 'video'
        self.split = split

        split_txt_paths = [str(Path(ucf_folder) / 'annotation'/ '{}list0{}'.format(split, str(i))) + '.txt' for i in split_num]

        self.video_list = []
        self.label_list = []
        self.label_to_num = dict()
        
        with open ('/data5/datasets/ucf101/annotation/classInd.txt', 'r') as f:
            for line in f:
                num, label = line.split()
                self.label_to_num[label] = int(num) - 1
        f.close()
        print(self.label_to_num)

        for split_txt_path in split_txt_paths:
            with open(split_txt_path, 'r') as f:
                for line in f:
                    if self.split == 'train':
                        video_name, label_name = line.split()
                        self.video_list.append(video_name)
                        self.label_list.append(int(label_name)-1) # to make label list in range(0,101)
                    elif self.split == 'test':
                        video_name= line.rstrip()
                        self.video_list.append(video_name)
                        self.label_list.append(self.label_to_num[(video_name.split('/'))[0]]) # to make label list in range(0,101)
            f.close()

        self.resize_width = resize_width
        self.resize_height = resize_height
        self.crop_size = crop_size
        self.clip_len = clip_len
        self.sampling = sampling
        self.normalize_type = normalize_type
    
    def __len__(self):
        return len(self.video_list)
    
    def __getitem__(self, idx):
        video_path = os.path.join(self.video_folder, self.video_list[idx])
        label = np.array(self.label_list[idx])

        if self.sampling == 'freq4_all_center':
            buffer = self.load_resized_frames(video_path)
            buffer = self.all_center_crop(buffer)
        elif self.sampling == 'uniform_center':
            buffer = self.uniform_load_resized_frames(video_path)
            buffer = self.center_crop(buffer)
        elif self.sampling == 'uniform_random':
            buffer = self.uniform_load_resized_frames(video_path)
            buffer = self.random_crop(buffer)
        elif self.sampling == 'freq4_all_random':
            buffer = self.load_resized_frames(video_path)
            buffer = self.all_random_crop(buffer)
        
        if self.split == 'train':
            if np.random.random() < 0.5:
                for i, frame in enumerate(buffer):
                    frame = cv.flip(buffer[i], flipCode=1) # (h,w,c)
                    buffer[i] = frame
        buffer = buffer.astype(np.float32)

        normalize_type = 'imagenet'
        if normalize_type == 'imagenet':
            for i, frame in enumerate(buffer):
                frame = frame / 255.0 # (h,w,c)
                buffer[i] = frame
            
            # buffer : (32, 112, 112, 3) 
            buffer = buffer.transpose((3, 0, 1, 2)) # (3, 32, 112, 112)
            
            mean = [0.406, 0.456, 0.485]
            std = [0.225, 0.224, 0.229]
            
            torch_buffer = torch.from_numpy(buffer)

            for i, channel in enumerate(torch_buffer):
                normalized_channel = (channel - mean[i])/std[i]
                torch_buffer[i] = normalized_channel
            
            torch_label = torch.from_numpy(label)
        elif normalize_type == 'ucf101':

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

            buffer = buffer.transpose((3, 0, 1, 2))
            print(buffer.shape, buffer.sum(), buffer.min(), buffer.max(), np.average(buffer))

            torch_buffer = torch.from_numpy(buffer)
            torch_label = torch.from_numpy(label)

            print(torch_buffer.shape, torch_label.shape)

        return torch_buffer, torch_label

    def load_resized_frames(self, video_path):
        cap = cv.VideoCapture(video_path)
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)

        frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        # frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        # frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

        EXTRACT_FREQUENCY = 4
        if frame_count // EXTRACT_FREQUENCY <= self.clip_len:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= self.clip_len:
                EXTRACT_FREQUENCY -= 1
                if frame_count // EXTRACT_FREQUENCY <= self.clip_len:
                    EXTRACT_FREQUENCY -= 1

        count, i, retaining = 0, 0, True
        video_frames = []
        while count < frame_count and retaining:
            retaining, frame = cap.read()
            if frame is None:
                continue

            if count % EXTRACT_FREQUENCY == 0:
                frame = cv.resize(frame, (self.resize_width, self.resize_height))
                video_frames.append(frame)
            count += 1
        cap.release()


        video_frames = np.asarray(video_frames, dtype=np.uint8)
        return video_frames

    def all_center_crop(self, buffer):

        if buffer.shape[0] > self.clip_len:
            t_idx = math.floor((buffer.shape[0] - self.clip_len) / 2)
        else:
            t_idx = 0
        h_idx = math.floor((buffer.shape[1] - self.crop_size) / 2)
        w_idx = math.floor((buffer.shape[2] - self.crop_size) / 2)

        buffer = buffer[t_idx: t_idx + self.clip_len,
                        h_idx: h_idx + self.crop_size,
                        w_idx: w_idx + self.crop_size,
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

    def center_crop(self, buffer):

        h_idx = math.floor((buffer.shape[1] - self.crop_size) / 2)
        w_idx = math.floor((buffer.shape[2] - self.crop_size) / 2)

        buffer = buffer[:,
                        h_idx: h_idx + self.crop_size,
                        w_idx: w_idx + self.crop_size,
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

    def uniform_load_resized_frames(self, video_path):
        cap = cv.VideoCapture(video_path)
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)

        frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        # frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        # frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

        # EXTRACT_FREQUENCY = 1

        unifrom_indices = np.linspace(0, frame_count, self.clip_len)
        unifrom_indices = unifrom_indices.astype('int32')
        count, i, retaining = 0, 0, True
        video_frames = []
        while count < frame_count and retaining:
            retaining, frame = cap.read()
            if frame is None:
                continue
            if count in unifrom_indices :
                frame = cv.resize(frame, (self.resize_width, self.resize_height))
                video_frames.append(frame)
            count += 1
        cap.release()


        video_frames = np.asarray(video_frames, dtype=np.uint8) # np.float32
        return video_frames
    
    def random_crop(self, buffer):
        # input : (t, h, w, c)

        rand_w_idx = random.randrange(0, self.resize_width - self.crop_size + 1)
        rand_h_idx = random.randrange(0, self.resize_height - self.crop_size + 1)

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
    
    def all_random_crop(self, buffer):
        # input : (t, h, w, c)

        rand_t_idx = random.randrange(0, buffer.shape[0] - self.clip_len + 1)
        rand_w_idx = random.randrange(0, self.resize_width - self.crop_size + 1)
        rand_h_idx = random.randrange(0, self.resize_height - self.crop_size + 1)

        buffer = buffer[rand_t_idx: rand_t_idx + self.clip_len,
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

    def to_tensor(buffer):
        return buffer.transpose((3, 0, 1, 2))

def collate_wrapper(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    return default_collate(batch)