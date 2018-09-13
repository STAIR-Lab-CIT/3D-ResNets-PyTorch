import torch
from torch import nn
from torch.autograd import Variable
import time
import sys

from utils import AverageMeter, calculate_accuracy

def conf_matrix(epoch, outputs,targets, confmat):
    for i in range(len(targets)):
        if torch.__version__ == '0.4.1':
            xx = targets[i].item()
        else:
            xx = targets[i].data[0]
        confmat[xx] = torch.add(confmat[xx],1.0,outputs[i].data)

    return confmat

def val_epoch(epoch, data_loader, model, criterion, opt, logger):
    model.eval()

    if opt.conf_matrix:
        print('Confusion matrix epoch {}'.format(epoch))
        confmat = torch.cuda.FloatTensor(opt.n_classes, opt.n_classes)
        sm = nn.Softmax(1)
    else:
        print('validation at epoch {}'.format(epoch))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
#        if i >= len(data_loader)-1:
#            print('avoiding last loop trap and exit')
#            break
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets = targets.cuda(async=True)
        if torch.__version__ == '0.4.1':
            with torch.no_grad():
                outputs = model(inputs)
        else:
            inputs = Variable(inputs, volatile=True)
            targets = Variable(targets, volatile=True)
            outputs = model(inputs)
        loss = criterion(outputs, targets)
        if opt.conf_matrix:
            probs = sm(outputs)
            confmat = conf_matrix(epoch, probs, targets, confmat)
        acc = calculate_accuracy(outputs, targets)

        if torch.__version__ == '0.4.1':
            losses.update(loss.item(), inputs.size(0))
        else:
            losses.update(loss.data[0], inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                  epoch, i + 1, len(data_loader), batch_time=batch_time,
                  data_time=data_time, loss=losses, acc=accuracies))

    print('val done')
    if not opt.conf_matrix:
        logger.log({
            'epoch': epoch,
            'loss': losses.avg,
            'acc': accuracies.avg
        })

    if opt.conf_matrix:
        cmfile = open('conf-matrix.txt','w')
        torch.save(confmat, 'conf_matrix.pt')
        for ii in range(opt.n_classes):
            for jj in range(opt.n_classes):
                cmfile.write(str(confmat[ii][jj]))
                cmfile.write('\t')
            cmfile.write('\n')
        cmfile.close()

    return losses.avg

