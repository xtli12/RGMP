import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import time
from utils_swin import AverageMeter, initialize_logger, save_checkpoint, record_loss
import torchvision
from torchvision import transforms
import shutil

from GuesNet import *
import timm

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,0,3'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'



parser = argparse.ArgumentParser(description="SSR")
parser.add_argument("--batchSize", type=int, default=20, help="batch size")
parser.add_argument("--end_epoch", type=int, default=666, help="number of epochs")
parser.add_argument("--init_lr", type=float, default=0.001, help="initial learning rate")
parser.add_argument("--decay_power", type=float, default=0.9, help="decay power")
parser.add_argument("--max_iter", type=float, default=400000, help="max_iter")
parser.add_argument("--outf", type=str, default="./Results/fuzzy_swin_t/",
                    help='path log files')
opt = parser.parse_args()


def main():
    if not os.path.exists(opt.outf):
        os.makedirs(opt.outf)
    loss_csv = open(os.path.join(opt.outf, 'loss.csv'), 'a+'); log_dir = os.path.join(opt.outf, 'train.log')
    logger = initialize_logger(log_dir); print('save_path is already')

    torch.backends.cudnn.benchmark = True
    shutil.copyfile(os.path.basename(__file__), opt.outf + os.path.basename(__file__))

    print("\nloading dataset ...")
    dataset, train_sampler = tra.load_data('./datasets/train_rm_dup_rmerror_fuzzy_n5_tgood_erosionx8/', is_train=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, sampler=train_sampler, num_workers=2,
                                              pin_memory=True)
    dataset_test, test_sampler = tra.load_data('./datasets/validdata_rm_dup_rmerror_fuzzy_n5_tgood_erosionx8', is_train=False)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, sampler=test_sampler, num_workers=2,
                                                   pin_memory=True)
    print("Train:%d" % (len(data_loader)))
    print("Validation set samples: ", len(data_loader_test))
    opt.max_iter = opt.end_epoch * len(data_loader)
    print("\nbuilding models_baseline ...")

    model = SwinTransformer(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), growth_rate=32,
                            block_config=(6, 12, 24, 16),
                            num_init_features=64, bn_size=4, drop_rate=0)


  '''
  re-implement of ConvNext, CoAtNet, EVA-02, and MaxViT
  '''
    # model = timm.create_model('convnext_xxlarge.clip_laion2b_soup_ft_in1k', pretrained=False, num_classes=14)
    criterion_train = nn.CrossEntropyLoss()
    torch.backends.cudnn.benchmark = True

    if torch.cuda.is_available():
        model.cuda();
        criterion_train.cuda();
    print('Model is already and parameters number is ', sum(param.numel() for param in model.parameters()))

    start_epoch = 0;
    iteration = 0;
    record_acc = 0.5

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    # optimizer = optim.Adam(model.parameters(), lr=opt.init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    resume_file = os.path.join(os.path.join(opt.outf), 'mae_finetuned_vit_base.pth')
    if resume_file:
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume_file))
            checkpoint = torch.load(resume_file, map_location=lambda storage, loc: storage.cuda(0))
            # start_epoch = checkpoint['epoch']
            # iteration = checkpoint['iter']
            model.load_state_dict(checkpoint['model'], strict=False)
            # model.load_state_dict(checkpoint, strict = False)
            # optimizer.load_state_dict(checkpoint['optimizer'])


    for epoch in range(start_epoch + 1, opt.end_epoch):
        start_time = time.time()
        # train
        train_loss, iteration, lr = tra.train(data_loader, model, criterion_train, optimizer, epoch, iteration,
                                              opt.init_lr, opt.decay_power)
        acc = tra.validate(data_loader_test, model)  # 调用validate
        if abs(acc - record_acc) < 0.001 or acc > record_acc or epoch % 50 == 0: 
            save_checkpoint(opt.outf, epoch, iteration, model, optimizer)  
            if acc > record_acc:
                record_acc = acc
        end_time = time.time()
        epoch_time = end_time - start_time
        print("Epoch [%02d], Iter[%06d], Time:%.9f, learning rate : %.9f, Train Loss: %.9f acc: %.9f " % (
        epoch, iteration, epoch_time, lr, train_loss, acc))
        record_loss1(loss_csv, epoch, iteration, epoch_time, lr, train_loss, acc)
        logger.info("Epoch [%02d], Iter[%06d], Time:%.9f, learning rate: %.9f, Train Loss: %.9f acc: %.9f " % (
        epoch, iteration, epoch_time, lr, train_loss, acc))  





class tra(nn.Module):

    def train(self, train_loader, model, criterion, optimizer, epoch, iteration, init_lr,
              decay_power): 
        model.train()  
        losses = AverageMeter()  
        for i, (images, labels) in enumerate(train_loader): 
            labels = labels.cuda();
            labels = Variable(labels)
            images = images.cuda();
            images = Variable(images) 
            lr = tra.poly_lr_scheduler(optimizer, init_lr, iteration, max_iter=opt.max_iter,
                                       power=decay_power) 
            iteration = iteration + 1  # iter+1
            output = model(images) 
            # output = model.forward_encoder(images, mask_ratio=0.75)
            loss = criterion(output, labels) 
            optimizer.zero_grad() 
            loss.backward()  
            optimizer.step()  
            losses.update(loss.data)  
            print('[Epoch:%02d],[Process:%d/%d],[iter:%d],lr=%.9f,train_losses.avg=%.9f' % (
            epoch, i, len(train_loader), iteration, lr, losses.avg))  
        return losses.avg, iteration, lr

    def validate(self, val_loader, model):
        model.eval() 
        ac = 0
        idx_to_class = {0: '10.0', 1: '10.5', 2: '11.0', 3: '11.5', 4: '12.0', 5: '12.5', 6: '13.0', 7: '6.5', 8: '7.0',
                        9: '7.5', 10: '8.0', 11: '8.5', 12: '9.0', 13: '9.5'}
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda();
            target = target.cuda()
            with torch.no_grad():  
                output = model(input)  
                # output = model.forward_encoder(input, mask_ratio=0.75)
                _, pred = output.topk(1, 1, True, True)
                pred = pred.flatten().cpu().numpy()
                target = target.cpu().numpy()
                target = torch.tensor([float(idx_to_class[target[i]]) for i in range(len(target))])
                pre_label = torch.tensor([float(idx_to_class[pred[i]]) for i in range(len(pred))])
                acc0, acc0_5 = tra.accuracy(pre_label, target)
                ac += (0.4 * acc0 + 0.6 * acc0_5) / len(val_loader)
        return ac

    def accuracy(self, output, target):
        pre_label = output.cuda()
        gt_label = target.cuda()
        acc0 = torch.le(abs(pre_label - gt_label), 0)
        acc0_5 = torch.le(abs(pre_label - gt_label), 0.5)
        return acc0, acc0_5

    def poly_lr_scheduler(self, optimizer, init_lr, iteraion, lr_decay_iter=1, max_iter=100, power=0.9):  
        if iteraion % lr_decay_iter or iteraion > max_iter:
            return optimizer
        lr = init_lr * (1 - iteraion / max_iter) ** power
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def load_data(self, dir, is_train):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if is_train:
            dataset = torchvision.datasets.ImageFolder(dir, transforms.Compose([
                transforms.Resize(224), transforms.CenterCrop(224),  
                transforms.ToTensor(), normalize]))
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            dataset = torchvision.datasets.ImageFolder(dir, transforms.Compose([
                transforms.Resize(224), transforms.CenterCrop(224),  
                transforms.ToTensor(), normalize]))
            sampler = torch.utils.data.SequentialSampler(dataset)
        return dataset, sampler




if __name__ == '__main__':
    if torch.cuda.is_available():
        tra = tra().cuda()
    main()  
    print(torch.__version__)  
