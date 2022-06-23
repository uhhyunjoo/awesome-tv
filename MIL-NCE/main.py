from utils import *
from model import *
from datasets import *

import wandb
import argparse

from loss import MILNCELoss
from torchsummary import summary
from s3dg import *

import gc

ckpt = None
epochs = 300

# workers = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# data_folder = '/data5/datasets/MSRVTT'
output_folder = './output/'
wandb_id = wandb.util.generate_id()



def main(args):
    torch.cuda.empty_cache()
    
    global ckpt, device, output_folder
    global epochs, wandb_id

    config = {
      "batch_size" : args.batch_size,
      "epochs" : args.epochs,
      "lr" : args.learning_rate,
      "step_size" : args.step_size
    }

    seed_everything()
    torch.cuda.empty_cache()
    gc.collect()
    # import pdb;pdb.set_trace()
    train_dataset = MSRVTT_Dataset(csv = args.train_csv, video_root=args.video_root)
    valid_dataset = MSRVTT_Dataset(csv = args.test_csv, video_root=args.video_root)
    

    check_dataset(train_dataset, valid_dataset)
    
    # collate_fn None
    # train_data_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers=args.workers, worker_init_fn  = seed_worker, pin_memory=True, drop_last = True)
    # valid_data_loader = DataLoader(valid_dataset, batch_size = args.batch_size, shuffle = False, num_workers=args.workers, worker_init_fn  = seed_worker, pin_memory=True, drop_last = False)
    
    # collate_fn Y
    train_data_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = False, collate_fn = collate_wrapper, num_workers=args.workers, worker_init_fn  = seed_worker, pin_memory=False, drop_last = True)
    valid_data_loader = DataLoader(valid_dataset, batch_size = args.batch_size, shuffle = False, collate_fn = collate_wrapper, num_workers=args.workers, worker_init_fn  = seed_worker, pin_memory=False, drop_last = False)
    
    if ckpt is None:
        seed_everything()
        model = S3D(num_classes = 512)
        wandb.init(id = wandb_id, resume = "allow", project= 's3d', config = config, entity="uhhyunjoo")
    else:
        checkpoint = torch.load(ckpt)
        model = S3D(num_classes = 512)
        optimizer = optim.SGD(model.parameters(), lr = args.learning_rate, momentum = 0.9, weight_decay = 5e-4)
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        model.load_state_dict(torch.load(ckpt))

    loss_fn = MILNCELoss()
    
    optimizer = optim.SGD(model.parameters(), lr = args.learning_rate, momentum = 0.9, weight_decay = 5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    model = model.to(device)

    wandb.watch(models = model, criterion = loss_fn, log = 'all')
    
    # import pdb;pdb.set_trace()
    with torch.autograd.set_detect_anomaly(True):
        for epoch in tqdm(range(epochs)):
            # train
            model.train()
            train_loss = 0.0
            #train_corrects = 0.0
            for video, text in tqdm(train_data_loader):
                # import pdb; pdb.set_trace()
                # loader 에서 data 가져오는 게 오래걸린다
                # video : ([3, 32, 224, 224])
                # 이걸 batch size 만큼이니까 # [batch_size, 3, num_frames, 224, 224] 어 맞네!
                # import pdb;pdb.set_trace()
                video = video.to(device) # [batch_size, 3, num_frames, 224, 224]
                text = text.to(device) # torch.Size([16, 30])

                # torch.Size([4, 3, 32, 224, 224])
                # torch.Size([4, 30])

                text = text.view(-1, text.shape[-1])
                
                video_embed, text_embed = model(video, text)

                # torch.Size([4, 512])
                # torch.Size([4, 512])

                optimizer.zero_grad()

                loss = loss_fn(video_embed, text_embed)

                loss.backward()
                optimizer.step()

                # pred = torch.argmax(output, dim = 1)
                train_loss += loss.item() * (video.shape)[0]
                # train_corrects += torch.sum(pred == y)
                del video
                del text
                del video_embed
                del text_embed

            import pdb;pdb.set_trace()
            train_epoch_loss = train_loss / len(train_data_loader.dataset)
            #train_epoch_acc = 100* train_corrects.double() / len(train_data_loader.dataset)
            print('Train Epoch: {}, Loss: {:.6f}'.format(epoch, train_epoch_loss))
            
            if epoch == 0:
                id_folder = output_folder + str(wandb_id) + '/'
                if not os.path.exists(id_folder):
                    os.makedirs(id_folder)
        
           
            if epoch % 10 == 0:
                ckpt_file_path = '{}/ckpt_{}.pth.tar'.format(id_folder, str(epoch))
                save_checkpoint(epoch, model, optimizer, scheduler, loss, ckpt_file_path)
            
            # val
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                # val_corrects = 0.0
                for val_data in valid_data_loader:

                    val_video = val_data["video"].to(device) # [16, 4, 3, 32, 224, 224] # [batch_size, 3, num_frames, 224, 224]
                    val_text = val_data["text"].to(device) # torch.Size([16, 30])

                    # torch.Size([4, 3, 32, 224, 224])
                    # torch.Size([4, 30])

                    val_text = val_text.view(-1, val_text.shape[-1])
                    
                    val_video_embed, val_text_embed = model(val_video, val_text)

                    # torch.Size([4, 512])
                    # torch.Size([4, 512])

                    val_loss_out = loss_fn(val_video_embed, val_text_embed)

                    val_loss += val_loss_out.item() * (val_video.shape)[0]
                    
                    del val_video
                    del val_text
                val_epoch_loss = val_loss / len(valid_data_loader.dataset)

                scheduler.step()

            print('Valid Epoch: {}, Loss: {:.6f}'.format(epoch, val_epoch_loss))

            wandb.log({'epoch': epoch, 'train_loss' : train_epoch_loss, 'val_loss': val_epoch_loss})

def check_dataset(train_dataset, valid_dataset):
    train_set = train_dataset.csv['video']
    valid_set = valid_dataset.csv['text']

    try :
        len(set(train_set + valid_set)) == len(train_set) + len(valid_set)
        print('Dataset : ok')
    except :
        print('train: {}, test : {}, set : {}'.format(len(train_set), len(valid_set), len(set(train_set + valid_set))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Ours from MIL-NCE HowTo100M with MSRVTT dataset')

    seed_everything()

    parser.add_argument('--batch_size', default= 256, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--step_size', default=100, type=int)

    parser.add_argument('--train_csv', default='/home/hjlee/workspace/Github/awesome-tv/MIL-NCE/msrvtt_train.csv', type=str)
    parser.add_argument('--test_csv', default='/home/hjlee/workspace/Github/awesome-tv/MIL-NCE/msrvtt_test.csv', type=str)
    parser.add_argument('--video_root', default='/data5/datasets/MSRVTT/videos/all', type=str)
    
    parser.add_argument('--workers', default=8, type=int)

    args = parser.parse_args()
    main(args)