import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image

import models.crnn as crnn


model_path = './data/netCRNN.pth'
img_path = './data/demo2.png'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

model = crnn.CRNN(32, 1, 37, 256)

if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((95, 32))
image = Image.open(img_path).convert('L')

image = transformer(image)

if torch.cuda.is_available():
    image = image.cuda()
image = image.view(1, *image.size())
image = Variable(image)

model.eval()
print('image shape', image.shape)
preds = model(image)

_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

preds_size = Variable(torch.IntTensor([preds.size(0)]))
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))
