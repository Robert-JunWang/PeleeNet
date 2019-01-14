import argparse 
import os 
import shutil 
import time 
import math 
import sys 

import numpy as np

import torch 
import torch.nn as nn 
import torch.nn.parallel 
import torch.backends.cudnn as cudnn 
import torch.distributed as dist 
import torch.optim 
import torch.utils.data 
import torch.utils.data.distributed 
import torchvision.transforms as transforms 
import torchvision.datasets as datasets 

from peleenet import PeleeNet 


model_names = [ 'peleenet'] 
engine_names = [ 'caffe', 'torch'] 

parser = argparse.ArgumentParser(description='PeleeNet ImageNet Evaluation') 
parser.add_argument('data', metavar='DIR', help='path to dataset') 
parser.add_argument('--arch', '-a', metavar='ARCH', default='peleenet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: peleenet)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--deploy', '-m', metavar='ARCH', default='caffe/peleenet.prototxt',
                    help='model file ' )

parser.add_argument('--engine', '-e', metavar='ENGINE', default='caffe', choices=engine_names,
                    help='engine type ' +
                        ' | '.join(engine_names) +
                        ' (default: caffe)')
parser.add_argument('--weights', type=str, metavar='PATH', default='caffe/peleenet.caffemodel',
                    help='path to init checkpoint (default: none)')

parser.add_argument('--input-dim', default=224, type=int,
                    help='size of the input dimension (default: 224)')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    print( 'args:',args)

    # Data loading code
    # Val data loading
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(args.input_dim+32),
            transforms.CenterCrop(args.input_dim),
            transforms.ToTensor(),
            normalize,
        ]))

    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    num_classes = len(val_dataset.classes)
    print('Total classes: ',num_classes)

    # create model
    print("=> creating {} model '{}'".format(args.engine, args.arch))
    model = create_model(num_classes, args.engine)

    if args.engine == 'torch':
        validate_torch(val_loader, model)
    else:
        validate_caffe(val_loader, model)



def create_model(num_classes, engine='torch'):

    if engine == 'torch':
        if args.arch == 'peleenet':
                model = PeleeNet(num_classes=num_classes)
        else:
                print("=> unsupported model '{}'. creating PeleeNet by default.".format(args.arch))
                model = PeleeNet(num_classes=num_classes)

        # print(model)

        model = torch.nn.DataParallel(model).cuda()

        if args.weights:
            if os.path.isfile(args.weights):
                print("=> loading checkpoint '{}'".format(args.weights))
                checkpoint = torch.load(args.weights)
                model.load_state_dict(checkpoint['state_dict'])

            else:
                print("=> no checkpoint found at '{}'".format(args.weights))



        cudnn.benchmark = True

    else:
        # create caffe model
        import caffe 
        caffe.set_mode_gpu()
        caffe.set_device(0)

        model_def = args.deploy
        model_weights = args.weights 

        model = caffe.Net(model_def,      # defines the structure of the model
                        model_weights,  # contains the trained weights
                        caffe.TEST)     # use test mode (e.g., don't perform dropout)

    return model

def validate_torch(val_loader, model):


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def validate_caffe(val_loader, net):
    batch_time = AverageMeter()

    batch_time = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for i, (inputs, target) in enumerate(val_loader):

        batch = inputs.numpy()[:, ::-1, ...]

        net.blobs['data'].reshape(len(batch),        # batch size
                              3,         # 3-channel (BGR) images
                              args.input_dim, args.input_dim)  

        net.blobs['data'].data[...] = batch

        output = net.forward()

        # measure elapsed time
        batch_time.update(time.time() - end)

        pre = np.array([x.argmax() for x in output['prob']])
        correct = np.sum(pre == target.numpy()) * 1.0/len(batch)

        top1.update(correct)


        #if i % args.print_freq == 0:
        print('Test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
               i, len(val_loader), batch_time=batch_time, top1=top1))

        end = time.time()

    print( ' * Prec@1 {top1.avg:.3f} '.format(top1=top1))

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
