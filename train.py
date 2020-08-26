from __future__ import print_function
from __future__ import division

import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from torch.nn import CTCLoss
import os
import utils
import dataset
import models.crnn as crnn

## 1.  创建数据集：1. cd tool 2. python3 create_dataset.py train_val.txt test_val.txt
## 2.1 训练线性模型：
## python3 train.py  --adadelta  --trainRoot tool/train/ --valRoot tool/test/ --pretrained weights/crnn_dense.pth --densemodel
## 2.2 训练RNN模型
## python3 train.py  --adadelta  --trainRoot tool/train/ --valRoot tool/test/ --pretrained weights/crnn_lstm.pth

parser = argparse.ArgumentParser()
parser.add_argument('--trainRoot', required=True, help='path to dataset')
parser.add_argument('--valRoot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--nepoch', type=int, default=5000000000, help='number of epochs to train for')
# TODO(meijieru): epoch -> iter
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrained', default='', help="path to pretrained model (to continue training)")
parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz')
parser.add_argument('--expr_dir', default='data', help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=50, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=2, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=50, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=50, help='Interval to be displayed')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate for Critic, not used by adadealta')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiemnt')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
parser.add_argument('--densemodel', action='store_true', help='use linear layer')
opt = parser.parse_args()

opt.alphabet = utils.alphabetChinese

if not os.path.exists(opt.expr_dir):
    os.makedirs(opt.expr_dir)

random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

train_dataset = dataset.lmdbDataset(root=opt.trainRoot)
assert train_dataset
if not opt.random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler = None
    
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=True, sampler=None,
    num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))

test_dataset = dataset.lmdbDataset(
    root=opt.valRoot, transform=dataset.resizeNormalize((100, 32)))

nclass = len(opt.alphabet) + 1 #训练字符集的个数
nc = 1 #输入图片的通道，nc=1是一个通道灰图

converter = utils.strLabelConverter(opt.alphabet)
criterion = CTCLoss(zero_infinity = True)

# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

lstmFlag = True
if opt.densemodel: lstmFlag = False
print('lstm layer used:', lstmFlag)
crnn = crnn.CRNN(opt.imgH, nc, opt.alphabet, opt.nh, lstmFlag=lstmFlag) 
crnn.apply(weights_init)

if opt.pretrained != '':
    print('loading pretrained model from %s' % opt.pretrained)
    crnn.load_state_dict(torch.load(opt.pretrained))

image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
text = torch.IntTensor(opt.batchSize * 5)
length = torch.IntTensor(opt.batchSize)

if opt.cuda:
    crnn.cuda()
    crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
    image = image.cuda()
    criterion = criterion.cuda()

image = Variable(image)
text = Variable(text)
length = Variable(length)

# loss averager
loss_avg = utils.averager()

# setup optimizer

params = crnn.parameters()
#训练最后一层
#params = crnn.rnn.parameters()
opt.adadelta = 1

if opt.adam:
    print('use optim.Adam')
    optimizer = optim.Adam(params, lr=opt.lr,
                           betas=(opt.beta1, 0.999))
elif opt.adadelta:
    print('use optim.Adadelta')
    optimizer = optim.Adadelta(params)
else:
    print('use optim.RMSprop')
    optimizer = optim.RMSprop(params, lr=opt.lr)

def val(net, dataset, criterion, max_iter=100):
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers))
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    loss_avg = utils.averager()

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)
       
        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
       
        log_preds = torch.nn.functional.log_softmax(preds,dim=2)
        cost = criterion(log_preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)  
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        
        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target.lower():
                n_correct += 1
       
        raw_preds = converter.decode(preds.data, preds_size.data, raw=True)
        for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
            print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * opt.batchSize)
    print('Test loss: %f, accuray: %.2f%%' % (loss_avg.val(), accuracy*100))


def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
   
    t, l = converter.encode(cpu_texts)
    
    utils.loadData(text, t)
    utils.loadData(length, l)
    
    #print("text, length = ", text, length) , 标注结果
    
    preds = crnn(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    
    #print("preds:", preds.shape) #26 x N=5 x 37

    log_preds = torch.nn.functional.log_softmax(preds,dim=2)
    cost = criterion(log_preds, text, preds_size, length) / batch_size
    
    crnn.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


for epoch in range(opt.nepoch):
    train_iter = iter(train_loader)
    i = 0
    while i < len(train_loader):
        for p in crnn.parameters():
            p.requires_grad = True
        crnn.train()

        cost = trainBatch(crnn, criterion, optimizer)
        #print(cost)
        loss_avg.add(cost)
        i += 1

        if i % opt.displayInterval == 0:
            print('[%d/%d][%d/%d] Loss: %f' %
                  (epoch, opt.nepoch, i, len(train_loader), loss_avg.val()))
            loss_avg.reset()

        if i % opt.valInterval == 0:
            val(crnn, test_dataset, criterion)

        # do checkpointing
        if i % opt.saveInterval == 0:
            torch.save(
                crnn.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format(opt.expr_dir, epoch, i))

    if epoch % 1 == 0:
        torch.save(crnn.state_dict(), '{0}/netCRNN_{1}.pth'.format(opt.expr_dir, str(lstmFlag)))
        print('[%d/%d][%d/%d] Loss: %.9f' %(epoch, opt.nepoch, i, len(train_loader), loss_avg.val()))
        val(crnn, test_dataset, criterion)
        loss_avg.reset()

#torch.save(crnn.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format(opt.expr_dir, opt.nepoch, i))