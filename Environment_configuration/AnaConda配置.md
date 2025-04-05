# AnaConda配置

## 软件包安装

**安装地址清华园链接**

https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/

**添加环境变量**

```
\Anaconda3\Scripts
\Anaconda3\Library\bin
\Anaconda3\Library\mingw-w64
\Anaconda3\Library\usr\bin
\Anaconda3
```

**添加.condarc文件**

```cmd
conda config --set show_channel_urls yes
```

**修改.condarc文件**

![image-20250405161401369](.\source\image-20250405161401369.png)

寻找对应目录下的.condarc文件，修改成

```cmd
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

从而使用清华镜像下载pkg

## 创建和加载虚拟环境

**创建名为code_learner的虚拟环境**

```cmd
conda create -n code_learner python=[version]
```

**查看虚拟环境**

```
conda env list
```

**加载虚拟环境**

```
conda activate code_learner
```