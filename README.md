# Handwritten_Digit_Recognition
手写数字识别
markdown

# MNIST 手写数字识别系统 (PyTorch实现)

## 项目描述
本项目使用PyTorch实现了一个卷积神经网络(CNN)，用于识别MNIST数据集中的手写数字。系统包含训练脚本和带可视化功能的测试脚本。

## 文件结构
项目/
├── CNN.py # CNN模型定义
├── pytorch_mnist.py # 训练脚本
├── mnist_test.py # 测试和可视化脚本
└── model/ # 模型保存目录
└── mnist_model.pkl # 训练好的模型文件


## 环境要求
- Python 3.6+
- PyTorch 1.0+
- torchvision
- OpenCV (用于可视化)
- NumPy

安装依赖：
```bash
pip install torch torchvision opencv-python numpy
使用说明
1. 训练模型
运行训练脚本：

bash

python pytorch_mnist.py
训练过程说明：

10个训练周期(epoch)

批大小(batch size): 64

使用Adam优化器(学习率0.01)

交叉熵损失函数

模型将保存到model/mnist_model.pkl

2. 测试模型
运行测试脚本：

bash

python mnist_test.py
测试功能说明：

显示每个测试图像及预测结果

在控制台显示真实标签

按任意键查看下一张图片

计算整体准确率

模型架构
python

CNN(
  (卷积层): Sequential(
    (0): 卷积层(1→32通道, 5x5卷积核, 边缘填充2)
    (1): 批归一化(32通道)
    (2): ReLU激活函数
    (3): 最大池化(2x2)
  )
  (全连接层): 线性层(6272输入→10输出)
)
主要特点
使用torchvision进行数据加载和预处理

自定义CNN实现

训练过程监控

交互式测试可视化

模型保存/加载功能

预期输出
训练过程中：


当前为第1轮，当前批次为1/937，loss为2.3025851249694824
当前为第1轮，当前批次为2/937，loss为2.3025851249694824
...
测试过程中：


预测值为7
真实值为7
[显示数字7的图像]
在MNIST测试集上最终准确率通常能达到>98%。

注意事项
如有GPU设备，可取消代码中的CUDA相关注释以启用GPU加速

模型会自动下载MNIST数据集(首次运行需要联网)

测试时按任意键可查看下一张图片

按ESC键可提前退出测试过程

模型文件(.pkl)需从可信来源获取，加载时可能需设置weights_only=False
