import torch.nn as nn
import torch
import utils
import dataset
from collections import OrderedDict
from torch.autograd import Variable


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):

    def __init__(self, imgH=32, nc=1, alphabet=None, nh=256, n_rnn=2, leakyRelu=False, lstmFlag=True):
        
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        self.lstmFlag = lstmFlag
        if alphabet == None:
            alphabet = utils.alphabetChinese
            
        nclass = len(alphabet) + 1       
        self.converter = utils.strLabelConverter(alphabet)

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0) 
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x50 , h(32)->h(16),  w(100) -> w(50)
        
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x25 h(16)->h(8), w(50)->w(25)
        
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x26 h(8)->h(4), w(25) -> w(26)
        
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x27 h(4)->h(2), w(26) -> w(27)  
        
        
        convRelu(6, True)  # Conv2d(kernel_size=(2, 2), stride=(1, 1)),  512x1x26 h(2)->h(1), w(27) -> w(26)  
        
        #### tensor shape总结 ######
        # 1. cnn input [N, 1, 32, 100] => output [N, 512, 1, 26]  
        # 2. w = input_w/4 + 1 + 1 - 1 = input_w/4 + 1,   h = input_h/16-1(最后输出必须为1)
        # 3. 默认input [w(100), h(32)] -> [w(26), h(1)]

        self.cnn = cnn
        if self.lstmFlag:
            self.rnn = nn.Sequential(
                BidirectionalLSTM(512, nh, nh),
                BidirectionalLSTM(nh, nh, nclass))
        else:
            self.linear = nn.Linear(nh*2, nclass)

    def forward(self, input):
        # conv features
        conv = self.cnn(input) # input [N, 1, 32, 100] => output [N, 512, 1, 26], T = 26
        
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2) # remove dim(2), h -> [N, 512, 26]  
        conv = conv.permute(2, 0, 1)  # [w, b, c] => [26, N, 512]
        if self.lstmFlag:
            # rnn features
            output = self.rnn(conv) # input [T(26), N, 512] => [w(26), N, nclass(37)]
        
        else:
            T, b, h = conv.size()
            t_rec = conv.contiguous().view(T * b, h)
            output = self.linear(t_rec)  # [T * b, nOut]
            output = output.view(T, b, -1)
        
        return output
    
    def load_weights_old(self,path):   
        
        trainWeights = torch.load(path,map_location=lambda storage, loc: storage)
        modelWeights = OrderedDict()
        for k, v in trainWeights.items():
            name = k.replace('module.','') # remove `module.`
            modelWeights[name] = v      
        self.load_state_dict(modelWeights)
        if torch.cuda.is_available() and self.GPU:
            self.cuda()
        self.eval()
        
    def load_weights(self,path):    
        self.load_state_dict(torch.load(path))  
        if torch.cuda.is_available() and self.GPU:
            self.cuda()
        self.eval()  
     
    def predict(self,image):
        img_w = 100 #2 * image.size[0] // image.size[1] 
        transformer = dataset.resizeNormalize((img_w, 32))  
        image = transformer(image)
        
        if torch.cuda.is_available():
            image = image.cuda()     
            
        image       = image.view(1, *image.size())
        image       = Variable(image)
        
        if image.size()[-1]<8:
            return ''
        
        preds       = self(image)
        max_val, preds = preds.max(2)
        preds = preds.view(-1)
        
        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        raw_pred = self.converter.decode(preds.data, preds_size.data, raw=True)
        sim_pred = self.converter.decode(preds.data, preds_size.data, raw=False)
        
        #sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
        return preds, raw_pred,sim_pred
    
