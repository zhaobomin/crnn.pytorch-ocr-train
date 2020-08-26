from PIL import Image
import models.crnn as crnn

model_path = './data/crnn_lstm.pth'
img_path = './data/demo.png'

model = crnn.CRNN()
print('loading pretrained model from %s' % model_path)
model.load_weights(model_path)

image = Image.open(img_path).convert('L')
preds, raw_pred,sim_pred = model.predict(image)


print('### 模型结果')
print('- raw_pred size: %d, sim_pred size: %d' %(len(raw_pred), len(sim_pred)))
print('- decode result: %-20s => %-20s' % (raw_pred, sim_pred))