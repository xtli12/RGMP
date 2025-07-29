import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import os
import time
import shutil
import torchvision
from torchvision import transforms
from utils_swin1 import AverageMeter, initialize_logger, save_checkpoint, record_loss1

from ARGN import *
import cv2
import numpy as np
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = '2,3,0,1'
# RobotArmDataset implementation
class RobotArmDataset(Dataset):
    def __init__(self, image_folder):
        if not os.path.exists(image_folder):
            raise FileNotFoundError(f"The specified path does not exist: {image_folder}")
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_files[idx])
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"The image file cannot be loaded: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (480, 480))  # Resize image to 224x224
        image = image.transpose((2, 0, 1))     # Convert to (C, H, W)
        image = torch.FloatTensor(image) / 255.0  # Normalize [0,1]

        # Extract joint angle information from filename
        filename = os.path.splitext(self.image_files[idx])[0]
        joint_angles = [float(angle) for angle in filename.split(',')]
        joint_angles = torch.FloatTensor(joint_angles)

        return image, joint_angles

# Trainer class for training and validation
class Trainer:
    def __init__(self, init_lr, decay_power, max_iter, outf):
        self.init_lr = init_lr
        self.decay_power = decay_power
        self.max_iter = max_iter
        self.outf = outf

    def poly_lr_scheduler(self, optimizer, iteration, lr_decay_iter=1):
        if iteration % lr_decay_iter or iteration > self.max_iter:
            return self.init_lr * (1 - iteration / self.max_iter) ** self.decay_power
        lr = self.init_lr * (1 - iteration / self.max_iter) ** self.decay_power
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def train_one_epoch(self, train_loader, model, criterion, optimizer, epoch, iteration):
        model.train()
        losses = AverageMeter()
        for i, (images, labels) in enumerate(train_loader):
            labels = labels.cuda()
            images = images.cuda()

            lr = self.poly_lr_scheduler(optimizer, iteration)
            iteration += 1
            output = model(images)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())
            print('[Epoch:%02d],[Process:%d/%d],[iter:%d],lr=%.9f,train_losses.avg=%.9f' %
                  (epoch, i, len(train_loader), iteration, lr, losses.avg))
        return losses.avg, iteration, lr

    def validate(self, val_loader, model, criterion):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                images = images.cuda()
                labels = labels.cuda()
                output = model(images)
                loss = criterion(output, labels)
                total_loss += loss.item()
        val_loss = total_loss / len(val_loader)
        return val_loss


def main():
    parser = argparse.ArgumentParser(description="SSR")
    parser.add_argument("--batchSize", type=int, default=1, help="batch size")
    parser.add_argument("--end_epoch", type=int, default=666, help="number of epochs")
    parser.add_argument("--init_lr", type=float, default=0.001, help="initial learning rate")
    parser.add_argument("--decay_power", type=float, default=0.9, help="decay power")
    parser.add_argument("--max_iter", type=float, default=400000, help="max_iter")
    parser.add_argument("--outf", type=str, default="./Results/ARGN/", help='path log files')
    parser.add_argument("--train_folder", type=str, default="./dataset/train/", help="train data folder")
    parser.add_argument("--valid_folder", type=str, default="./dataset/valid/", help="valid data folder")
    opt = parser.parse_args()

    if not os.path.exists(opt.outf):
        os.makedirs(opt.outf)

    loss_csv = open(os.path.join(opt.outf, 'loss.csv'), 'a+')
    log_dir = os.path.join(opt.outf, 'train.log')
    logger = initialize_logger(log_dir)
    print('save_path is already')

    torch.backends.cudnn.benchmark = True
    shutil.copyfile(os.path.basename(__file__), opt.outf + os.path.basename(__file__))

    print("\nloading dataset ...")
    # Use RobotArmDataset for data loading
    dataset_train = RobotArmDataset(opt.train_folder)
    data_loader = DataLoader(dataset_train, batch_size=opt.batchSize, shuffle=True, num_workers=2, pin_memory=True)

    dataset_test = RobotArmDataset(opt.valid_folder)
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    print("Train samples:", len(dataset_train))
    print("Validation samples:", len(dataset_test))
    opt.max_iter = opt.end_epoch * len(data_loader)
    print("\nbuilding model ...")

    model = SwinTransformer(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), window_size=15, num_classes=6)
    model.cuda()

    # Use MSELoss for regression task
    criterion_train = nn.MSELoss().cuda()

    optimizer = optim.SGD(model.parameters(), lr=opt.init_lr, momentum=0.9, weight_decay=1e-4)

    best_model_path = os.path.join(opt.outf, 'best_argn_model.pth')
    start_epoch = 0
    iteration = 0
    best_loss = float('inf')

    # Try to load previous model weights
    if os.path.exists(best_model_path):
        print("Loading model weights from previous training...")
        model.load_state_dict(torch.load(best_model_path))

    trainer = Trainer(opt.init_lr, opt.decay_power, opt.max_iter, opt.outf)

    record_loss = 0

    for epoch in range(start_epoch + 1, opt.end_epoch):
        start_time = time.time()
        train_loss, iteration, lr = trainer.train_one_epoch(data_loader, model, criterion_train, optimizer, epoch, iteration)
        val_loss = trainer.validate(data_loader_test, model, criterion_train)

        # Use validation MSE Loss for model saving decision
        if val_loss < record_loss:
            record_loss = val_loss
            save_checkpoint(opt.outf, epoch, iteration, model, optimizer)

        end_time = time.time()
        epoch_time = end_time - start_time
        print("Epoch [%02d], Iter[%06d], Time:%.9f, learning rate : %.9f, Train Loss: %.9f, Val Loss: %.9f " %
              (epoch, iteration, epoch_time, lr, train_loss, val_loss))
        record_loss1(loss_csv, epoch, iteration, epoch_time, lr, train_loss, val_loss)
        logger.info("Epoch [%02d], Iter[%06d], Time:%.9f, learning rate: %.9f, Train Loss: %.9f, Val Loss: %.9f" %
                    (epoch, iteration, epoch_time, lr, train_loss, val_loss))


if __name__ == '__main__':
    main()
    print(torch.__version__)
