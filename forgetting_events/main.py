from model import resnet
import data as dataset
import utils
import plot_curves
import numpy as np
import time

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

parser = argparse.ArgumentParser(description='Confidence Aware Learning')
parser.add_argument('--batch_size', default= 256, type=int, help='Batch size')
parser.add_argument('--epochs', default= 150, type=int, help='Total number of epochs to run')
parser.add_argument('--model', default='res', type=str, help='Models name to use [res, dense, vgg]')
parser.add_argument('--save_path', default='./remove_test/', type=str, help='Savefiles directory')
parser.add_argument('--gpu', default='7', type=str, help='GPU id to use')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--remove', default='remove', type=str, help='Mode = [remove]')
parser.add_argument('--mode', default='forgettable', type=str, help='Mode = [forgettable, inforgettable]')
parser.add_argument('--sort', default='reverse', type=str, help='Mode = [reverse, default]')
parser.add_argument('--remove_ratio', default= 10, type=int, help='Remove data ratio(%)')
parser.add_argument('--forget_histroy', default='./train_forgetting.pth', type=str, help='Forgetting Event history')
args = parser.parse_args()

def main():
    # set GPU ID
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cudnn.benchmark = True

    # check save path
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # make dataloader
    train_loader, test_loader = dataset.get_loader(args)

    # set model
    if args.model == 'res':
        model = resnet.ResNet18().cuda()

    # set criterion
    criterion = nn.CrossEntropyLoss().cuda()

    # set optimizer (default:sgd)
    optimizer = optim.SGD(model.parameters(),
                          lr=0.1,
                          momentum=0.9,
                          weight_decay=0.0001,
                          nesterov=True)
    # set scheduler
    scheduler = MultiStepLR(optimizer,
                            milestones=[80,120],
                            gamma=0.1)

    # make logger
    train_logger = utils.Logger(os.path.join(save_path, 'train.log'))
    test_logger = utils.Logger(os.path.join(save_path, 'test.log'))

    # forgetting
    forgetting_histroy = utils.Forgetting_Events(data_size=len(train_loader.dataset))

    # Start Train
    for epoch in range(1, args.epochs + 1):
        # scheduler
        scheduler.step()
        # Train
        train(train_loader, model, criterion, optimizer, epoch, forgetting_histroy, train_logger)
        validate(test_loader, model, criterion, epoch, test_logger, 'test')
        # Save Model each epoch
        if epoch == int(args.epochs):
            torch.save(model.state_dict(), os.path.join(save_path, '{0}_{1}.pth'.format('model', epoch)))
    # Finish Train
    torch.save(forgetting_histroy, os.path.join(save_path,'train_forgetting.pth'))
    # Draw Plot
    plot_curves.draw_plot(save_path)

def train(train_loader, model, criterion, optimizer, epoch, forgetting_histroy, logger):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()


    end = time.time()

    model.train()
    for i, (input, target, idx, file_name) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # set input ,target
        input, target = input.cuda(), target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec, correct = utils.accuracy(output, target)
        losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        # forgetting update
        forgetting_histroy.update(idx, correct, file_name, epoch=epoch)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'            
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))


    logger.write([epoch, losses.avg, top1.avg])



def validate(val_loader, model, criterion, epoch, logger, mode):
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    end = time.time()
    total_acc = 0

    model.eval()
    with torch.no_grad():
        for i, (input, target, _, _) in enumerate(val_loader):
            input, target = input.cuda(), target.cuda()
            # compute output
            output = model(input)
            loss = criterion(output, target)

            pred = output.data.max(1, keepdim=True)[1]
            total_acc += pred.eq(target.data.view_as(pred)).sum()

            # measure accuracy and record loss
            prec, correct = utils.accuracy(output, target)
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                print(mode, ': [{0}/{1}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses))
    total_acc = 100. * total_acc / len(val_loader.dataset)

    print('Accuracy {:.2f}'.format(total_acc))

    logger.write([epoch, losses.avg, total_acc.item()])


if __name__ == "__main__":
    main()



