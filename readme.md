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

    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Collecting paddlex
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/d6/a2/07435f4aa1e51fe22bdf06c95d03bf1b78b7bc6625adbb51e35dc0804cc7/paddlex-1.3.11-py3-none-any.whl (516kB)
    [K     |████████████████████████████████| 522kB 12.9MB/s eta 0:00:01
    [?25hCollecting xlwt (from paddlex)
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/44/48/def306413b25c3d01753603b1a222a011b8621aed27cd7f89cbc27e6b0f4/xlwt-1.3.0-py2.py3-none-any.whl (99kB)
    [K     |████████████████████████████████| 102kB 24.2MB/s ta 0:00:01
    [?25hRequirement already satisfied: opencv-python in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (4.1.1.26)
    Collecting pycocotools; platform_system != "Windows" (from paddlex)
      Downloading https://mirror.baidu.com/pypi/packages/de/df/056875d697c45182ed6d2ae21f62015896fdb841906fe48e7268e791c467/pycocotools-2.0.2.tar.gz
    Requirement already satisfied: psutil in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (5.7.2)
    Collecting shapely>=1.7.0 (from paddlex)
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/98/f8/db4d3426a1aba9d5dfcc83ed5a3e2935d2b1deb73d350642931791a61c37/Shapely-1.7.1-cp37-cp37m-manylinux1_x86_64.whl (1.0MB)
    [K     |████████████████████████████████| 1.0MB 14.9MB/s eta 0:00:01
    [?25hCollecting paddlehub==2.1.0 (from paddlex)
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/7a/29/3bd0ca43c787181e9c22fe44b944b64d7fcb14ce66d3bf4602d9ad2ac76c/paddlehub-2.1.0-py3-none-any.whl (211kB)
    [K     |████████████████████████████████| 215kB 29.1MB/s eta 0:00:01
    [?25hRequirement already satisfied: sklearn in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (0.0)
    Requirement already satisfied: tqdm in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (4.36.1)
    Requirement already satisfied: flask-cors in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (3.0.8)
    Requirement already satisfied: colorama in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (0.4.4)
    Collecting paddleslim==1.1.1 (from paddlex)
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/d1/77/e257227bed9a70ff0d35a4a3c4e70ac2d2362c803834c4c52018f7c4b762/paddleslim-1.1.1-py2.py3-none-any.whl (145kB)
    [K     |████████████████████████████████| 153kB 28.3MB/s eta 0:00:01
    [?25hRequirement already satisfied: visualdl>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (2.2.0)
    Requirement already satisfied: pyyaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (5.1.2)
    Requirement already satisfied: numpy>=1.14.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from opencv-python->paddlex) (1.20.3)
    Requirement already satisfied: setuptools>=18.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pycocotools; platform_system != "Windows"->paddlex) (56.2.0)
    Requirement already satisfied: cython>=0.27.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pycocotools; platform_system != "Windows"->paddlex) (0.29)
    Requirement already satisfied: matplotlib>=2.1.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pycocotools; platform_system != "Windows"->paddlex) (2.2.3)
    Requirement already satisfied: rarfile in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (3.1)
    Requirement already satisfied: packaging in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (20.9)
    Requirement already satisfied: flask>=1.1.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (1.1.1)
    Requirement already satisfied: pyzmq in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (18.1.1)
    Requirement already satisfied: easydict in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (1.9)
    Requirement already satisfied: colorlog in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (4.1.0)
    Requirement already satisfied: filelock in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (3.0.12)
    Collecting paddle2onnx>=0.5.1 (from paddlehub==2.1.0->paddlex)
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/37/80/aa6134b5f36aea45dc1b363e7af941dccabe4d7e167ac391ff046f34baf1/paddle2onnx-0.7-py3-none-any.whl (94kB)
    [K     |████████████████████████████████| 102kB 32.2MB/s ta 0:00:01
    [?25hRequirement already satisfied: gunicorn>=19.10.0; sys_platform != "win32" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (20.0.4)
    Requirement already satisfied: paddlenlp>=2.0.0rc5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (2.0.0rc7)
    Requirement already satisfied: gitpython in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (3.1.14)
    Requirement already satisfied: Pillow in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (7.1.2)
    Requirement already satisfied: scikit-learn in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from sklearn->paddlex) (0.24.2)
    Requirement already satisfied: Six in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask-cors->paddlex) (1.15.0)
    Requirement already satisfied: protobuf>=3.11.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (3.14.0)
    Requirement already satisfied: pre-commit in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (1.21.0)
    Requirement already satisfied: requests in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (2.22.0)
    Requirement already satisfied: pandas in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (1.1.5)
    Requirement already satisfied: flake8>=3.7.9 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (3.8.2)
    Requirement already satisfied: bce-python-sdk in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (0.8.53)
    Requirement already satisfied: shellcheck-py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (0.7.1.1)
    Requirement already satisfied: Flask-Babel>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (1.0.0)
    Requirement already satisfied: python-dateutil>=2.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib>=2.1.0->pycocotools; platform_system != "Windows"->paddlex) (2.8.0)
    Requirement already satisfied: cycler>=0.10 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib>=2.1.0->pycocotools; platform_system != "Windows"->paddlex) (0.10.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib>=2.1.0->pycocotools; platform_system != "Windows"->paddlex) (1.1.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib>=2.1.0->pycocotools; platform_system != "Windows"->paddlex) (2.4.2)
    Requirement already satisfied: pytz in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib>=2.1.0->pycocotools; platform_system != "Windows"->paddlex) (2019.3)
    Requirement already satisfied: Werkzeug>=0.15 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.0->paddlehub==2.1.0->paddlex) (0.16.0)
    Requirement already satisfied: itsdangerous>=0.24 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.0->paddlehub==2.1.0->paddlex) (1.1.0)
    Requirement already satisfied: Jinja2>=2.10.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.0->paddlehub==2.1.0->paddlex) (2.10.1)
    Requirement already satisfied: click>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.0->paddlehub==2.1.0->paddlex) (7.0)
    Requirement already satisfied: seqeval in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (1.2.2)
    Requirement already satisfied: jieba in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (0.42.1)
    Requirement already satisfied: h5py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (2.9.0)
    Requirement already satisfied: gitdb<5,>=4.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from gitpython->paddlehub==2.1.0->paddlex) (4.0.5)
    Requirement already satisfied: scipy>=0.19.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn->sklearn->paddlex) (1.6.3)
    Requirement already satisfied: joblib>=0.11 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn->sklearn->paddlex) (0.14.1)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn->sklearn->paddlex) (2.1.0)
    Requirement already satisfied: aspy.yaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (1.3.0)
    Requirement already satisfied: cfgv>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (2.0.1)
    Requirement already satisfied: toml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (0.10.0)
    Requirement already satisfied: virtualenv>=15.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (16.7.9)
    Requirement already satisfied: identify>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (1.4.10)
    Requirement already satisfied: nodeenv>=0.11.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (1.3.4)
    Requirement already satisfied: importlib-metadata; python_version < "3.8" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (0.23)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->paddlex) (1.25.6)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->paddlex) (2019.9.11)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->paddlex) (3.0.4)
    Requirement already satisfied: idna<2.9,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->paddlex) (2.8)
    Requirement already satisfied: mccabe<0.7.0,>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0->paddlex) (0.6.1)
    Requirement already satisfied: pyflakes<2.3.0,>=2.2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0->paddlex) (2.2.0)
    Requirement already satisfied: pycodestyle<2.7.0,>=2.6.0a1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0->paddlex) (2.6.0)
    Requirement already satisfied: pycryptodome>=3.8.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl>=2.0.0->paddlex) (3.9.9)
    Requirement already satisfied: future>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl>=2.0.0->paddlex) (0.18.0)
    Requirement already satisfied: Babel>=2.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl>=2.0.0->paddlex) (2.8.0)
    Requirement already satisfied: MarkupSafe>=0.23 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Jinja2>=2.10.1->flask>=1.1.0->paddlehub==2.1.0->paddlex) (1.1.1)
    Requirement already satisfied: smmap<4,>=3.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from gitdb<5,>=4.0.1->gitpython->paddlehub==2.1.0->paddlex) (3.0.5)
    Requirement already satisfied: zipp>=0.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata; python_version < "3.8"->pre-commit->visualdl>=2.0.0->paddlex) (0.6.0)
    Requirement already satisfied: more-itertools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from zipp>=0.5->importlib-metadata; python_version < "3.8"->pre-commit->visualdl>=2.0.0->paddlex) (7.2.0)
    Building wheels for collected packages: pycocotools
      Building wheel for pycocotools (setup.py) ... [?25ldone
    [?25h  Created wheel for pycocotools: filename=pycocotools-2.0.2-cp37-cp37m-linux_x86_64.whl size=278362 sha256=040b3946276260ad69d71d529cd6d51bfda0ffc3573ac2ec3d996fa5b10b805c
      Stored in directory: /home/aistudio/.cache/pip/wheels/fb/44/67/8baa69040569b1edbd7776ec6f82c387663e724908aaa60963
    Successfully built pycocotools
    Installing collected packages: xlwt, pycocotools, shapely, paddle2onnx, paddlehub, paddleslim, paddlex
      Found existing installation: paddlehub 2.0.4
        Uninstalling paddlehub-2.0.4:
          Successfully uninstalled paddlehub-2.0.4
    Successfully installed paddle2onnx-0.7 paddlehub-2.1.0 paddleslim-1.1.1 paddlex-1.3.11 pycocotools-2.0.2 shapely-1.7.1 xlwt-1.3.0



```python
#调用一些要用到的库
import paddlex as pdx
import cv2
import numpy as np
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/utils.py:26: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      def convert_to_list(value, n, name, dtype=np.int):


使用如下命令即可将数据划分为70%训练集，20%验证集和10%的测试集。


```python
!paddlex --split_dataset --format VOC --dataset_dir hand-music --val_value 0.2 --test_value 0.1
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/utils.py:26: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      def convert_to_list(value, n, name, dtype=np.int):
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

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/__init__.py:107: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import MutableMapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/rcsetup.py:20: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Iterable, Mapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/colors.py:53: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Sized
    2021-08-16 15:35:34,868 - INFO - font search path ['/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf', '/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/afm', '/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/pdfcorefonts']
    2021-08-16 15:35:35,288 - INFO - generated new fontManager


    2021-08-16 15:35:35 [INFO]	Starting to read file list from dataset...
    2021-08-16 15:35:36 [INFO]	883 samples in file hand-music/train_list.txt
    creating index...
    index created!
    2021-08-16 15:35:36 [INFO]	Starting to read file list from dataset...
    2021-08-16 15:35:36 [INFO]	251 samples in file hand-music/val_list.txt
    creating index...
    index created!


# 四、模型训练

PaddleX目前提供了FasterRCNN和YOLOv3两种检测结构，多种backbone模型。本基线系统以骨干网络为MobileNetV1的YOLOv3算法为例。


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
生成result.txt记录结果


```python
image_name = 'test1.jpg'
# 模型保存位置
model = pdx.load_model('output/yolov3_mobilenet/best_model')

img = cv2.imread(image_name)
result = model.predict(img)


keep_results = []
areas = []
f = open('./output/yolov3_mobilenet/result.txt', 'a')
for dt in np.array(result):
    cname, bbox, score = dt['category'], dt['bbox'], dt['score']
    if score < 0.5:                    #准确率低于0.5的结果不记录
        continue
    keep_results.append(dt)
    f.write(str(dt) + '\n')
    f.write('\n')
    areas.append(bbox[2] * bbox[3])
areas = np.asarray(areas)
sorted_idxs = np.argsort(-areas).tolist()
keep_results = [keep_results[k]
                for k in sorted_idxs] if len(keep_results) > 0 else []
print(keep_results)
f.close()

```

    2021-08-16 15:36:07 [INFO]	Model[YOLOv3] loaded.
    [{'category_id': 2, 'bbox': [40.198211669921875, 4.55499267578125, 236.2442626953125, 326.3627014160156], 'score': 0.7120435237884521, 'category': 'pause'}]


# 六、模型导出
模型导出结果位于inference_model文件中


```python
!paddlex --export_inference --model_dir=output/yolov3_mobilenet/best_model --save_dir=./inference_model
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/utils.py:26: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      def convert_to_list(value, n, name, dtype=np.int):
    W0816 15:36:21.716179   993 device_context.cc:362] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
    W0816 15:36:21.720713   993 device_context.cc:372] device: 0, cuDNN Version: 7.6.
    2021-08-16 15:36:27 [INFO]	Model[YOLOv3] loaded.
    2021-08-16 15:36:27 [INFO]	Model for inference deploy saved in ./inference_model.



```python
#测试导出模型
image_name = 'test1.jpg'
# 模型保存位置
model = pdx.load_model('inference_model')

img = cv2.imread(image_name)
result = model.predict(img)


keep_results = []
areas = []
f = open('./output/yolov3_mobilenet/result.txt', 'a')
for dt in np.array(result):
    cname, bbox, score = dt['category'], dt['bbox'], dt['score']
    if score < 0.3:
        continue
    keep_results.append(dt)
    f.write(str(dt) + '\n')
    f.write('\n')
    areas.append(bbox[2] * bbox[3])
areas = np.asarray(areas)
sorted_idxs = np.argsort(-areas).tolist()
keep_results = [keep_results[k]
                for k in sorted_idxs] if len(keep_results) > 0 else []
print(keep_results)
f.close()

```

    2021-08-16 15:36:44 [INFO]	Model[YOLOv3] loaded.
    [{'category_id': 2, 'bbox': [40.198211669921875, 4.55499267578125, 236.2442626953125, 326.3627014160156], 'score': 0.7120435237884521, 'category': 'pause'}]


# 七、结果可视化
导出测试结果位于根目录下


```python
#结果可视化
image_name = 'test1.jpg'
img = cv2.imread(image_name)
pdx.det.visualize(img, result, threshold=0, save_dir='./', color=None)
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/cbook/__init__.py:2349: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      if isinstance(obj, collections.Iterator):


    2021-08-16 15:36:48 [INFO]	The visualized result is saved as ./visualize_1629099408886.jpg



```python
#模型生成压缩包 下载到本地部署
!zip -r inference_model.zip inference_model
```

      adding: inference_model/ (stored 0%)
      adding: inference_model/.success (stored 0%)
      adding: inference_model/__params__ (deflated 7%)
      adding: inference_model/model.yml (deflated 50%)
      adding: inference_model/__model__ (deflated 96%)


# 八、本地简单部署成果
[视频](https://www.bilibili.com/video/BV1uM4y1L763/)
<iframe style="width:100%;height: 640px;" src="//player.bilibili.com/player.html?aid=588161536&bvid=BV1uM4y1L763&cid=340211724&page=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> 


#音乐播放器代码位于data/Music Player.zip中
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
