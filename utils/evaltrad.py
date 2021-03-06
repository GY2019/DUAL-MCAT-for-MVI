import time
import torch
from skimage.feature import local_binary_pattern

def exactLBPfeatruebatch(data):
    npdata = data.numpy()
    radius = 1 
    n_points = 8 * radius 
    #train_features =[]
    #test_features =[]
    for i in range(0,npdata.shape[0]):
        for j in range(0,npdata.shape[1]):
            each_img = npdata[i,j,:,:]
            lbp_feature = local_binary_pattern(each_img, n_points, radius)
            npdata[i,j,:,:] = lbp_feature
    nplbpfeature = torch.from_numpy(npdata)
    return nplbpfeature

def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (data, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        datalbp = exactLBPfeatruebatch(data)
        
        input_var = (data.cuda(),datalbp.cuda())
        input_var = tuple(d.cuda() for d in input_var)
        input_var = tuple(torch.autograd.Variable(d, volatile=True) for d in input_var)
        target_var = torch.autograd.Variable(target, volatile=True)
        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data[0], data.size(0))
        top1.update(prec1[0], data.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

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
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res