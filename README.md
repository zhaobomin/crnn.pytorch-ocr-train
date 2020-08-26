CRNN网络模型-CNN+RNN OCR
======================================

基于CRNN的OCR模型：
1. 支持横向和竖向文字OCR（中文和英文）
2. 提供数据集创建工具和训练代码
3. 提供预训练模型（LSTM版本和Linear版本）
4. 基于Pytorch CTCLoss，不依赖warp_ctc_pytorch（简化依赖环境）

此项目参考:
- CRNN模型Pytorch版本：https://github.com/meijieru/crnn.pytorch
- chineseocr: https://github.com/chineseocr/chineseocr

模型使用
--------
1. 先下载预训练模型,拷贝到weights/文件夹，百度网盘下载:https://pan.baidu.com/s/1nvRw665LwS6a9dxApvT_Xw  密码:uqqy
2. 运行:python3 demo.py

- 输入图片:
![Example Image](./data/demo.png)

- 结果输出:
    loading pretrained model from ./data/crnn.pth
    a-----v--a-i-l-a-bb-l-ee-- => available

Train模型
-----------------
1. 创建数据集：
    - cd tool
    - python3 create_dataset.py train_val.txt test_val.txt
    
2. 模型训练：
    - 训练Linear版本：python3 train.py  --adadelta  --trainRoot tool/train/ --valRoot tool/test/ --pretrained weights/crnn_dense.pth --densemodel
    - 训练LSTM版本：python3 train.py  --adadelta  --trainRoot tool/train/ --valRoot tool/test/ --pretrained weights/crnn_lstm.pth

