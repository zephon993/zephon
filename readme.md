# 项目介绍
原先的项目构思是通过摄像头收集人脸情绪进行识别进而智能更换音乐歌曲，但由于情绪识别往往是一瞬间的情感（我们很少会一直笑，一直激动，有可能上一秒机器识别到情绪为高兴而放喜庆的音乐，但下一秒我们的情绪已经平复导致原本将要播放的歌曲一下子又更换了，也可能放到高潮部分因为情绪的错误识别而将歌曲替换，导致效果不佳，所以最终决定采用更为稳定的手势识别方式来进行对音乐播放器的控制）本项目旨在通过手势识别来控制音乐播放器，以满足某些场合下，不方便直接操控音乐软件的需求。

# 灵感来源
    在一些公共的场合如理发店、奶茶店、餐厅等都会播放音乐来调节气氛，而音乐的控制方式是直接运用电脑来操控（固定位置操控），有时候碰到不想听的、让客人反感的歌曲，或者歌曲音量过大过低的情况，往往需要工作人员亲自过去更换，如果正在忙走不开，也只能等歌曲播放完自动切换（如理发师正在理发、服务员正在为客户点餐），为了使用户的体验更佳，并且给予客人对音乐环境的控制权力（客人也可以通过自己的手势来控制音乐的播放及切换），我想部署一个可以通过手势控制的智能音乐盒在此类场景，便于对背景音乐进行控制，后期将基于文字识别模型，让用户可自行点歌（客户对音乐盒摄像头展现出想听的歌曲名字，机器进行识别播放）。

# 一、数据集说明

* 本项目使用的数据集是：自定义的手势数据集，数据来源于aistudio中的公开数据集，并将这些数据集进行重新整合。每一种手势对应一种音乐播放器的控制操作。

该数据集已加载至本环境中，位于：**data/data103092/hand-music.zip**


```python
## 解压数据集（解压一次即可，请勿重复解压）
!unzip -q -o data/data103092/hand-music.zip
```

解压完成后，左侧文件夹处会多一个名为hand-music的文件夹，该文件夹下有2个子文件夹：

1. **Annotations图像文件**
1. **Annotations标注文件**



```python
# 查看数据集文件结构
!tree hand-music -L 1
```

    hand-music
    ├── Annotations
    └── JPEGImages
    
    2 directories, 0 files


# 二、数据准备

本基线系统使用的数据格式是PascalVOC格式，开发者基于PaddleX开发目标检测模型时，无需对数据格式进行转换，开箱即用。

但为了进行训练，还需要将数据划分为训练集、验证集和测试集。划分之前首先需要**安装PaddleX**。


```python
#安装paddlex
!pip install paddlex
```


```python
#调用一些要用到的库
import paddlex as pdx
import cv2
import numpy as np
```


```python
#数据划分效果
!tree hand-music -L 1
```

    hand-music
    ├── Annotations
    └── JPEGImages
    
    2 directories, 0 files


使用如下命令即可将数据划分为70%训练集，20%验证集和10%的测试集。


```python
!paddlex --split_dataset --format VOC --dataset_dir hand-music --val_value 0.2 --test_value 0.1
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/setuptools/depends.py:2: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      import imp
    Dataset Split Done.
    Train samples: 883
    Eval samples: 251
    Test samples: 125
    Split files saved in hand-music



```python
#数据划分效果
!tree hand-music -L 1
```

    hand-music
    ├── Annotations
    ├── JPEGImages
    ├── labels.txt
    ├── test_list.txt
    ├── train_list.txt
    └── val_list.txt
    
    2 directories, 4 files


# 三、数据预处理

在训练模型之前，对目标检测任务的数据进行操作，从而提升模型效果。可用于数据处理的API有：
- **Normalize**：对图像进行归一化
- **ResizeByShort**：根据图像的短边调整图像大小
- **RandomHorizontalFlip**：以一定的概率对图像进行随机水平翻转
- **RandomDistort**：以一定的概率对图像进行随机像素内容变换

更多关于数据处理的API及使用说明可查看文档：
[https://paddlex.readthedocs.io/zh_CN/release-1.3/apis/transforms/det_transforms.html](https://paddlex.readthedocs.io/zh_CN/release-1.3/apis/transforms/det_transforms.html)


```python
from paddlex.det import transforms
# 定义训练和验证时的transforms
# API说明 https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/det_transforms.html
train_transforms = transforms.Compose([
    # 此处需要补充图像预处理代码
    transforms.Resize(target_size=608, interp='LINEAR'),
    transforms.RandomHorizontalFlip(prob=0.5),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], min_val=[0., 0., 0.], max_val=[255., 255., 255.])

])

eval_transforms = transforms.Compose([
    # 此处需要补充图像预处理代码
    transforms.Resize(target_size=608, interp='LINEAR'),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], min_val=[0., 0., 0.], max_val=[255., 255., 255.])
])


```


```python
# 定义训练和验证所用的数据集
train_dataset = pdx.datasets.VOCDetection(
    data_dir='hand-music',
    file_list='hand-music/train_list.txt',
    label_list='hand-music/labels.txt',
    transforms=train_transforms,
    shuffle=True)

eval_dataset = pdx.datasets.VOCDetection(
    data_dir='hand-music',
    file_list='hand-music/val_list.txt',
    label_list='hand-music/labels.txt',
    transforms=eval_transforms,
    shuffle=False)
```

# 四、模型训练

PaddleX目前提供了FasterRCNN和YOLOv3两种检测结构，多种backbone模型。本项目以骨干网络为MobileNetV1的YOLOv3算法进行训练。


```python
#初始化模型
num_classes = len(train_dataset.labels)
model = pdx.det.YOLOv3(num_classes=num_classes, backbone='MobileNetV1')
```


```python
#模型训练及相关参数
model.train(
    num_epochs=270,
    train_dataset=train_dataset,
    train_batch_size=2,
    eval_dataset=eval_dataset,
    learning_rate=0.001 / 8,
    warmup_steps=1000,
    warmup_start_lr=0.0,
    save_interval_epochs=1,
    lr_decay_epochs=[210,240],
    save_dir='output/yolov3_mobilenet',
    use_vdl='ture'
    )
```

# 五、模型检测



```python
image_name = 'test1.jpg'
# 模型保存位置
model = pdx.load_model('output/yolov3_mobilenet/best_model')

img = cv2.imread(image_name)
result = model.predict(img)


keep_results = []
areas = []
for dt in np.array(result):
    cname, bbox, score = dt['category'], dt['bbox'], dt['score']
    if score < 0.5:                    #准确率低于0.5的结果不记录
        continue
    keep_results.append(dt)
    areas.append(bbox[2] * bbox[3])
areas = np.asarray(areas)
sorted_idxs = np.argsort(-areas).tolist()
keep_results = [keep_results[k]
                for k in sorted_idxs] if len(keep_results) > 0 else []
print(keep_results)

```

# 六、模型导出
模型导出结果位于inference_model文件中


```python
!paddlex --export_inference --model_dir=output/yolov3_mobilenet/best_model --save_dir=./inference_model
```


```python
#测试导出模型
image_name = 'test1.jpg'
# 模型保存位置
model = pdx.load_model('inference_model')

img = cv2.imread(image_name)
result = model.predict(img)


keep_results = []
areas = []
for dt in np.array(result):
    cname, bbox, score = dt['category'], dt['bbox'], dt['score']
    if score < 0.5:
        continue
    keep_results.append(dt)
    areas.append(bbox[2] * bbox[3])
areas = np.asarray(areas)
sorted_idxs = np.argsort(-areas).tolist()
keep_results = [keep_results[k]
                for k in sorted_idxs] if len(keep_results) > 0 else []
print(keep_results)


```

    2021-08-17 12:16:37 [INFO]	Model[YOLOv3] loaded.
    [{'category_id': 2, 'bbox': [40.19819641113281, 4.5550384521484375, 236.24424743652344, 326.36256408691406], 'score': 0.7120453119277954, 'category': 'pause'}]


# 七、结果可视化
导出测试结果位于根目录下


```python
#结果可视化
image_name = 'test1.jpg'
img = cv2.imread(image_name)
pdx.det.visualize(img, result, threshold=0.5, save_dir='./', color=None)
```

    2021-08-17 12:18:29 [INFO]	The visualized result is saved as ./visualize_1629173909067.jpg


![](https://ai-studio-static-online.cdn.bcebos.com/f216bb6f38af4008b058235e2f983f1b78f0611dbf8a4ac78055e956b0d632e5)



```python
#模型生成压缩包 下载到本地部署
!zip -r inference_model.zip inference_model
```

# 八、本地简单部署成果
[视频](https://www.bilibili.com/video/BV1uM4y1L763/)
<iframe style="width:100%;height: 640px;" src="//player.bilibili.com/player.html?aid=588161536&bvid=BV1uM4y1L763&cid=340211724&page=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> 


**音乐播放器代码位于根目录中**
[代码来源](https://blog.csdn.net/qq_44614026/article/details/88833953?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522162895165116780265427641%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=162895165116780265427641&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v29-2-88833953.ecpm_v1_rank_v29&utm_term=python%E9%9F%B3%E4%B9%90%E6%92%AD%E6%94%BE%E5%99%A8&spm=1018.2226.3001.4187)

项目中的音乐播放器在仅在上述播放器基础上添加了手势识别功能

# 九、开发过程遇到的问题
1. 模型精确度不够高，容易出现错误的识别，影响对播放器的操控。------要训练更精确、高效的模型
2. 摄像头读取信息，并交由模型检测的过程需要一定时间，最终导致摄像头读取的帧数堆积，在cv2.imshow('frame', new_img)时，实时图像延迟严重--------------------网上解决办法：采用多线程，一个进程展示实时图像，一个进程进行数据处理，输出结果。


# 十、总结
1. 模型训练调参是提高模型精确度的关键，本项目的模型准确度不高，当有外界干扰时，很难做出准确的判断。
2. 本地部署环境配置很复杂，远比在aistudio上开发困难
3. 要全面发展，运用多线程，解决图像延迟问题。

aistudio账号
zephon993