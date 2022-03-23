from utils import *
from model import *
from datasets import *

import wandb
import argparse

ckpt = None
epochs = 200
#batch_size = 100
#lr = 0.005
#use_bn = False
#dropout_p = 0.5
#clips_per_video = 1
workers = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ucf_folder = '/data5/datasets/ucf101'
output_folder = './output/'
wandb_id = wandb.util.generate_id()

def train_and_val(model, train_data_loader, valid_data_loader):

    loss_fn = nn.CrossEntropyLoss()
    
    if args.optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr = args.learning_rate, momentum = 0.9, weight_decay = 5e-4)
    elif args.optimizer_type == 'Adam':
        optim_configs = [{'params': model.parameters(), 'lr': args.learning_rate}]
        optimizer = optim.Adam(optim_configs, lr=args.learning_rate, weight_decay=5e-4) # lr = 1e-4
    elif args.optimizer_type =='SGD_params':
        train_params = [{'params': get_1x_lr_params(model), 'lr': args.learning_rate},
                {'params': get_10x_lr_params(model), 'lr': args.learning_rate * 10}]
        optimizer = optim.SGD(train_params, lr = args.learning_rate, momentum = 0.9, weight_decay = 5e-4)
    
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, verbose=True)

    if args.scheduler == 'default':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    elif args.scheduler == 'reduce':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, verbose=True)
    model = model.to(device)
    wandb.watch(models = model, criterion = loss_fn, log = 'all')

    
    with torch.autograd.set_detect_anomaly(True):
        for epoch in tqdm(range(epochs)):
            #epoch_correct = 0.0
            model.train()
            train_loss = 0.0
            train_corrects = 0.0
            for data in tqdm(train_data_loader):
                X, y = data
                X = X.to(device) 
                y = y.to(device)
                # import pdb;pdb.set_trace()
                optimizer.zero_grad()

                output = model(X) # grad_fn=<SoftmaxBackward>

                loss = loss_fn(output, y) # 4.6184

                loss.backward()
                optimizer.step()

                pred = torch.argmax(output, dim = 1)
                #epoch_correct += (pred == y).sum().item()
                train_loss += loss.item() * (X.shape)[0]
                # loss.item() : 4.618359565734863, X.shape[0] = 20
                train_corrects += torch.sum(pred == y)
                del X
                del y
            
            
            # acc = 100 * epoch_correct / (len(train_data_loader) * args.batch_size)
            train_epoch_loss = train_loss / len(train_data_loader.dataset) # 19123
            train_epoch_acc = 100* train_corrects.double() / len(train_data_loader.dataset) # 19123
            print('Train Epoch: {}, Acc: {:.6f}, Loss: {:.6f}'.format(epoch, train_epoch_acc, train_epoch_loss))
            
            if epoch == 0:
                id_folder = output_folder + str(wandb_id) + '/'
                if not os.path.exists(id_folder):
                    os.makedirs(id_folder)
        
            ckpt_file_path = '{}/ckpt_{}.pth.tar'.format(id_folder, str(epoch))
            save_checkpoint(epoch, model, optimizer, loss, ckpt_file_path)
            
            model.eval()
            #val_corrects = 0.0
            with torch.no_grad():
                val_loss = 0.0
                val_corrects = 0.0
                for val_data in valid_data_loader:
                    val_X, val_y = val_data
                    val_X = val_X.to(device)
                    val_y = val_y.to(device)
                    val_output = model(val_X)
                    val_loss_out = loss_fn(val_output, val_y)
                    val_loss += val_loss_out.item() * (val_X.shape)[0]
                    val_pred = torch.argmax(val_output, dim = 1)
                    # val_pred = torch.max(nn.Softmax(dim = 1)(val_output), 1)[1]
                    val_corrects += torch.sum(val_pred == val_y)
                    del val_X
                    del val_y
                val_epoch_loss = val_loss / len(valid_data_loader.dataset)
                val_epoch_acc = 100 * val_corrects.double() / len(valid_data_loader.dataset)
            
            if args.scheduler == 'reduce':
                scheduler.step(val_epoch_loss)
            else:
                scheduler.step()

            print('Valid Epoch: {}, Acc: {:.6f}'.format(epoch, val_epoch_acc))

            wandb.log({'epoch': epoch, 'train_acc' : train_epoch_acc, 'train_loss' : train_epoch_loss, 'val_acc' : val_epoch_acc, 'val_loss': val_epoch_loss})
            # wandb.log({'epoch': epoch, 'acc' : acc, 'loss' : loss})

            #wandb.log({'epoch': epoch, 'acc' : acc, 'loss' : loss})


def main(args):
    torch.cuda.empty_cache()
    
    global ckpt, device, ucf_folder, output_folder
    global epochs, workers, wandb_id

    config = {
      "batch_size" : args.batch_size,
      "resize_width" : args.resize_width,
      "resize_height" : args.resize_height,
      "crop_size" : args.crop_size,
      "clip_len" : args.clip_len,
      "optimizer_type": args.optimizer_type,
      "lr" : args.learning_rate,
      "step_size" : args.step_size,
      "sampling" : args.sampling,
      "scheduler" : args.scheduler,
      "use_vr" : args.use_vr
    }

    seed_everything()

    train_dataset = UCF101Dataset(ucf_folder = ucf_folder, split= 'train', split_num = [1,2], resize_width = args.resize_width, resize_height = args.resize_height, crop_size = args.crop_size, clip_len = args.clip_len, sampling = args.sampling, use_vr = args.use_vr)
    valid_dataset = UCF101Dataset(ucf_folder = ucf_folder, split= 'train', split_num = [3], resize_width = args.resize_width, resize_height = args.resize_height, crop_size = args.crop_size, clip_len = args.clip_len, sampling = args.sampling, use_vr = args.use_vr)
    
    train_data_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, collate_fn = collate_wrapper, num_workers=workers, worker_init_fn  = seed_worker, pin_memory=True, drop_last = True)
    valid_data_loader = DataLoader(valid_dataset, batch_size = args.batch_size, shuffle = False, collate_fn = collate_wrapper, num_workers=workers, worker_init_fn  = seed_worker, pin_memory=True, drop_last = False)
    
    if ckpt is None:
        seed_everything()
        model = C3DNet(num_classes = 101)
        wandb.init(id = wandb_id, resume = "allow", project= 'c3d', config = config, entity="uhhyunjoo")
    
    train_and_val(model, train_data_loader, valid_data_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train C3D with UCF101')

    # 250 default setting
    parser.add_argument('--batch_size', default= 20, type=int)
    parser.add_argument('--resize_width', default = 171, type=int)
    parser.add_argument('--resize_height', default = 128, type=int)
    parser.add_argument('--crop_size', default = 112, type=int)
    parser.add_argument('--clip_len', default = 16, type=int)
    parser.add_argument('--optimizer_type', default='SGD', type=str, choices=['SGD', 'Adam', 'SGD_params'])
    parser.add_argument('--channel_order', default='bgr', type=str)
    
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--step_size', default=20, type=int)
    parser.add_argument('--scheduler', default = 'default', type = str, choices = ['default', 'reduce'])


    # uniform random 이 나을듯?
    parser.add_argument('--sampling', default='uniform_random', type=str, choices=['freq4_all_center', 'freq4_all_random', 'uniform_center', 'uniform_random'])
    parser.add_argument('--use_vr', default=False, type = bool)

    
    args = parser.parse_args()
    main(args)