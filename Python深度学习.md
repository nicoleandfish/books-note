# ch4 机器学习基础
## 4.1 机器学习的四个分支
### 4.1.1 监督学习（supervised learning）
主要包括分类和回归。例如：二分类问题、多分类问题和标量回归问题，以及，光学字符识别、语音识别、图像分类和语言翻译。
- 序列生成（sequence generation）：给定一张图像，预测描述图像的文字。序列生成有时可以被重新表示为一系列分类问题，比如反复预测序列中的单词或标记。
- 语法树预测（syntax tree prediction）：给定一个橘子，预测其分解生成的语法树。
- 目标检测（object detection）：给定一张图像，在图中特定目标的周围画一个边界框，可表示为分类问题（给定多个候选边界框，对每个框内的目标进行分类）或分类与回归联合问题（用向量回归来预测边界框的坐标）。
- 图像分割（image segmentation）：给定一张图像，在特定物体上画一个像素级的掩模（mask）。

### 4.1.2 无监督学习
降维（dimensionality reduction）和聚类（clustering）为无监督学习方法。

### 4.1.3 自监督学习
自编码器（autoencoder）为一个自监督学习的例子。

### 4.1.4 强化学习
主要集中在研究领域，除游戏外还没有取得实践上的重大成功。但应用场景在未来会很丰富：自动驾驶汽车、机器人、资源管理等。

术语
- 样本（sample）或输入（input）：进入模型的数据点
- 预测（prediction）或输出（output）：经过模型得到的结果
- 目标（target）真实值。

## 4.2 评估机器学习模型
机器学习的目的是得到可泛化（generalize）的模型。
### 4.2.1 训练集、验证集和测试集

## 4.4 过拟合

# ch5 深度学习用于计算机视觉
## 5.1卷积神经网络简介
卷积神经网络（convnet）
卷积神经网络接受形状为（image_height,image_width,image_channels)的输入张量（不包括批量维度）。
### 5.1.1 卷积运算
密集连接层和卷积层的根本区别在于，Dense层从输入特征空间中学到的全局模式（全局模式就是涉及所有像素的模式），而卷积层学到的是局部模式。
卷积神经网络具有以下两个性质：
- 平移不变性（translation invariant）。
- 模式的空间层次结构（spatial hierarchies of patterns）。
对于包含两个空间轴（高度和宽度）和一个深度轴（或称为通道轴）的3D张量

# ch6 深度学习用于文本和序列
用于处理序列的两种基本的深度学习算法分别是循环神经网络（recurrent neural network）和一维卷积神经网络（1D convnet）。
## 6.1 处理文本数据
文本向量化（vectorize）是指将文本转换为数值张量的过程。
- 将文本分割为单词，并将每个单词转换为一个向量。
- 将文本分割为字符，并将每个字符转换为一个向量。
- 提取单词或字符的n-gram，并将每个n-gram转换为一个向量。n-gram是多个连续单词或字符的集合（n-gram之间可重叠）。
标记（token）：将文本分解而成的单元（单词、字符或n-gram）。
分词（tokenization）：将文本分解成标记的过程。
将向量与标记相关联的方法：one-hot编码（one-hot encoding）与标记嵌入（token embedding，适用于单词为词嵌入（word embedding））。
一般从文本到标记再到向量。

二元语法袋（bag-of-2-grams）及三元语法袋（bag-of-3-grams）举例：
"the cat sat on the mat"
bag-of-2-grams:
{"the","the cat","cat","cat sat","sat","sat on","on","on the","the","the mat","mat"}
bag-of-3-grams:
{"the","the cat","cat","cat sat","the cat sat","sat","sat on","on","cat sat on","on the","the","sat on the","the mat","mat","on the mat"}
词袋（bag-of-words）是不保存顺序的粉刺方法，用于千层的语言处理模型，而非深度学习模型。在使用轻量级的浅层文本处理模型时（比如logisitc回归喝随机森林），n-gram是一种较为性能较好的特征工程工具。

### 6.1.1 单词和字符的one-hot编码
one-hot编码是将标记转换为向量的最常用、最基本的方法。
```
#one-hot coding sample one
import numpy as np
samples = ['the cat sat on the mat.','the dog ate my homework.']
token_index={} #create the index of token
for sample in samples:
  for word in sample.split(): #method split to tokenization, in real apply need reduce the symboland special marks.
    if word not in token_index:
      token_index[word] = len(token_index)+1 #for each word to create unique index, pay attention index start from 1 not 0.
max_length=10 #only take 10 word for each sample

results = np.zeros(shape=(len(samples),max_length,max(token_index.values())+1))

for i,sample in enumerate(samples):
  for j,word in list(eumerate(sample.split()))[:max_length]:
    index = token_index.get(word)
    results[i,j,index]=1.
```

```
import string
samples=['the cat sat on the mat.','the dog ate my homework.']
characters = string.printable
token_index=dict(zip(rang(1,len(characters)+1),characters))

max_length=50
results = np.zeros((len(samples),max_length,max(token_index.keys())+1))
for i,sample in enumerate(samples):
  for j,character in enumerate(sample):
    index=token_index.get(character)
    results[i,j,index]=1.
```
```
from keras.preprocessing.text import Tokenizer
samples=['','']
tokenizer=Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples)
sequences=tokenizer.texts_to_sequences(samples)
one_hot_results = tokenizer.texts_to_matrix(samples,mode='binary')
word_index=tokenizer.word_index
print('Found %s unique tokens.' %len(word_index))
```
one-hot编码存在一种变体为one-hot散列技巧（one-hot hashing trick），适用于词表中唯一标记数量太大的场景。但会存在散列冲突（hash collision）的问题，即两个不同的单词可能具有相同的散列值。

### 6.1.2
```
from keras.preprocessing.text import Tokenizer
from kears.preprocessing.sequence import pad_sequences
import numpy as np

maxlen=100
training_samples=200
validation_samples=10000
max_words=10000

tokenizer=Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences=tokenizer.texts_to_sequences(texts)

word_index=tokenizer.word_index
print('found %s unique tokens.' %len(word_index))

data=pad_sequences(sequences, maxlen=maxlen)

labels=np.asarray(labels)
print('shape of data tensor:' data.shape)
print('shape of label tensor:' labels.shape)

indices=np.arange(data.shape[0])
np.random.shuffle(indices)
data=data[indices]
labels=labels[indices]
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples+validation_samples]
y_val = labels[training_samples: training_samples +validation_samples]

```

# 在EC2 GPU实例上运行Jupyter notebook
需要AWS账号。
- 打开EC2控制面板（https://console.aws.amazon.com/ec2/v2），点击Launch Instance。
- 选择AWS Marketplace，并搜索"deep learning"。找到Deep Learning AMI Ubuntu Version的Amazon系统映像（AMI）
- 选择p2.xlarge实例，此实例允许访问单个GPU。
- Configure Instance\Add Storage\Add Tags可以保留默认配置。Configure Security Group需要自定义配置，新增一个TCP规则来允许8888端口被本地计算机访问，为保证个人信息安全，设置本笔记本电脑IP进行访问。
首次创建实例，需要创建新密钥并下载。
- 连接实例需要在EC2控制面上选择，并Connect。
- 通过SSH登陆到实例上，建议创建ssl目录，并打开。（mkdir ssl/cd ssl）
- 使用OpenSSL创建一个新的SSL证书，并在当前ssl目录下创建cert.key和cert.pem文件。
openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout "cert.key" -out "cert.pem" -batch.

配置Jupyter
需要修改远程实例上的默认配置。
- 生成新的Jupyter配置文件。
jupyter notebook --generate-config
- （可选）为笔记本生成Jupyter密码。打开IPython shell，运行以下代码。
from IPython.lib import passwd
passwd()
exit
- passwd()命令会要求输入密码并确认。完成后，会显示密码的散列值。
- 使用vi或任何文本编辑器编辑Jupyter配置文件。
vi /.jupyter/jupyter_notebook_config.py
- 将下列代码增加到文本开头。
c = get_config()
c.NotebookApp.certfile = u'/home/ubuntu/ssl/cert.pem'
c.NotebookApp.keyfile = u'/home/ubuntu/ssl/cert.key'
c.IPKernelApp.pylab = 'inline'
c.NotebookApp.ip = '*'

c.NotebookApp.open_browser=False
c.NotebookApp.password = '之前生成的密码散列值'

安装Keras
默认自带，需升级keras版本
sudo pip install keras --upgrade
如果是python3，则需要pip3

设置本地端口转发
在本地计算机的shell中，将本地的443端口（HTTPS端口）转发到远程实例的8888端口。
sudo ssh -i awsKeys.pem -L local_port:local_machine:remote_port remote_machine
sudo ssh -i awsKeys.pen -L 443:127.0.0.1:8888 ubuntu@ec2-54-147-126-214.compute-1.amazonaws.com
远程机器在ssh的connect时会出现。

在远程实例上将本书相关的Jupyter的Github仓库克隆下来：
git clone https://github.com/fchollet/deep-learning-with-python-notebooks.git
cd deep-learning-with-python-notebooks
在远程实例上启动Jupyter
jupyter notebook
在本地浏览器https://127.0.0.1
