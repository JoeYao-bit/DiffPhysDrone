
ubuntu 24.04复现NMI论文

```
# DiffPhysDrone 完整搭建文档（Ubuntu20.04 Docker容器）
> 目标：搭建可微分无人机仿真环境，编译自定义CUDA扩展 quadsim_cuda，成功运行单/多智能体训练
> 前置条件：宿主机已安装 docker + nvidia-container-toolkit，宿主机NVIDIA驱动支持CUDA11.8
> 容器名称：diffphysdrone

## 一、宿主机操作：拉取镜像 + 创建容器
宿主机终端执行
```bash
# 拉取ubuntu20.04镜像
docker pull ubuntu:20.04

# 创建并启动容器，挂载源码、软件目录，开启GPU直通
docker run -it --gpus all \
-v /宿主机DiffPhysDrone源码路径:/workspace/DiffPhysDrone \
-v /宿主机software目录:/home/software \
--name diffphysdrone \
ubuntu:20.04 /bin/bash
```

### 再次进入容器（宿主机）

```
# 启动容器 + 交互式bash进入
docker start diffphysdrone && docker exec -it diffphysdrone /bin/bash
# 容器内退出：exit，仅退出shell，容器不停止
# ❌ 禁止 docker attach，Ctrl+C会直接杀死容器
```

## 二、容器内：基础系统依赖安装

```
apt update
apt install -y build-essential wget curl git
```

## 三、容器内：安装 Miniconda

```
# 下载安装脚本
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# 静默安装到 /opt/miniconda3
bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3
```

## 四、创建 Python3.11 conda 环境

```
# 临时初始化conda
eval "$(/opt/miniconda3/bin/conda shell.bash hook)"
# 创建py311环境
conda create -n py311 python=3.11 -y
# 激活环境
conda activate py311
```

## 五、安装 CUDA Toolkit 11.8（仅 nvcc 编译工具，不装显卡驱动）

> 
> 驱动复用宿主机 NVIDIA 驱动

```
# cuda安装包放在挂载目录 /home/software
sh /home/software/cuda_11.8.0_520.61.02_linux.run --toolkit --silent --override
```

校验 nvcc

```
nvcc -V
# 预期输出 release 11.8, V11.8.89
```

## 六、写入～/.bashrc 自动加载环境（交互式终端自动配置）

```
cat >> ~/.bashrc <<'EOF'
# ===== Conda + CUDA + Torch lib auto setup for DiffPhysDrone =====
# only run for interactive shell
if [[ $- == *i* ]]; then
    # conda init
    eval "$(/opt/miniconda3/bin/conda shell.bash hook)"
    conda activate py311
    # CUDA Toolkit path
    export PATH=/usr/local/cuda-11.8/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
    # Torch lib: fix custom cpp extension libc10.so missing
    export TORCH_LIB=/opt/miniconda3/envs/py311/lib/python3.11/site-packages/torch/lib
    export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"
fi
EOF

source ~/.bashrc
```

## 七、安装 PyTorch 2.2.2+cu118

```
conda activate py311
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu118
```

校验 torch 与 GPU

```
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'GPU available:', torch.cuda.is_available())"
```

### 修复 numpy 版本冲突（Torch2.2.2 仅支持 numpy1.x）

```
pip uninstall numpy -y
pip install "numpy<2.0"
```

## 八、编译 DiffPhysDrone 自定义 CUDA 扩展 quadsim_cuda

> 
> ⚠️ 必须加 --no-build-isolation，否则构建沙盒找不到 torch

```
cd /workspace/DiffPhysDrone
pip install -e src --no-build-isolation
```

✅ 成功标志：`Successfully built quadsim_cuda` + `Successfully installed quadsim_cuda-0.0.0`

验证扩展导入

```
python -c "import quadsim_cuda; print('quadsim_cuda loaded OK')"
```

## 九、安装项目运行依赖

```
pip install matplotlib tensorboard tqdm scipy gym
```

## 十、启动训练脚本

```
# 单智能体
python main_cuda.py $(cat configs/single_agent.args)

# 多智能体
python main_cuda.py $(cat configs/multi_agent.args)
```

> 
> 说明：`torch.cross without specifying the dim arg` 为弃用警告，不影响仿真计算，可忽略。

## 十一、后台长时间训练（防止 SSH 断开中断任务）

```
nohup python main_cuda.py $(cat configs/single_agent.args) > train_single.log 2>&1 &
# 实时查看日志
tail -f train_single.log
```

Tensorboard 可视化查看 loss 曲线

```
tensorboard --logdir=runs --bind_all
```


# 踩坑记录

1. CUDA run 安装：**只安装 toolkit，不要勾选 driver**，驱动使用宿主机驱动。
2. numpy>=2 与 torch2.2.2 存在 ABI 冲突，必须锁定`numpy<2.0`。
3. 编译 quadsim_cuda：缺少`--no-build-isolation`会报 ModuleNotFoundError: No module named 'torch'。
4. 导入扩展报错`libc10.so: cannot open shared object file`：conda activate 不会自动加载 torch/lib，需配置 LD_LIBRARY_PATH 写入 bashrc。
5. 不要使用`docker attach`进入容器，极易误按 Ctrl+C 杀死整个容器。
6. 环境变量放在`if [[ $- == *i* ]]`内，**仅交互式终端生效，后台训练脚本不受影响**。

# 环境版本清单

- 基础镜像：ubuntu:20.04
- Python：3.11
- CUDA Toolkit(nvcc)：11.8.89
- Pytorch：2.2.2+cu118
- Numpy：1.26.x
- 自定义 CUDA 扩展：quadsim_cuda


# 不是，**训练过程中就可以实时看曲线，不需要等全部训练结束**

TensorBoard 是**边训练边写日志**，每迭代若干步就把 loss 写入 `runs/` 目录，你随时可以打开网页查看。

## 操作步骤（容器内执行）

```
tensorboard --logdir=runs --bind_all
```

- `--bind_all`：允许宿主机浏览器访问（非常关键，不加只能容器内部访问）
- 启动后会输出地址，类似：`http://localhost:6006`

打开宿主机浏览器访问：

```
http://容器IP:6006
```

> 
> 如果是本地机器，容器 IP 可以用宿主机`localhost:6006`；远程服务器填服务器公网 IP:6006


## 宿主机终端执行（新开一个宿主机窗口！不要在正在跑训练的那个窗口输入）

```
docker exec -it diffphysdrone /bin/bash
```

# DiffPhysDrone 训练产出结果格式

## 1. 模型权重（最核心，用于断点续训、评估推理）

后缀：`.pth` / `.pt`，**PyTorch 标准序列化文件**

- 内部存储：网络权重、optimizer 状态、迭代次数、loss 记录等
- 读写方式：`torch.load()` / `torch.save()`
- 使用：`--resume xxx.pth` 加载做续训或者评估

> 
> 示例文件名：`model_50000.pth`，`ckpt_100000.pt`

## 2. TensorBoard 日志（loss 曲线数据）

存放目录：`runs/`，**不是图片，是一组 event 二进制文件**

- 文件名称类似：`events.out.tfevents.xxxx`
- 格式：TensorFlow protobuf 二进制日志
- 用途：TensorBoard 读取，绘制 loss、奖励等标量曲线
- 特点：不能直接双击打开；可以离线回看，也能用 python 脚本读取里面的标量数据导出 csv

## 3. 文本日志（控制台输出）

- `train_single.log`：**纯文本 utf-8**
保存训练打印的 loss、迭代信息，可用`tail/grep`查看，可导出到 Excel 做离线绘图。

## 4. 评估输出（仅开启 eval/--eval_only 才会生成）

### 4.1 轨迹图片

`*.png` 静态图片（matplotlib 保存），**只有评估阶段才生成**，训练阶段默认没有。

### 4.2 仿真轨迹数据（可选，看代码实现）

很多时候代码会额外保存轨迹数值：

- `.npy`：numpy 二进制数组，存储无人机位置、速度、姿态时间序列，python 直接加载做后处理
- `.csv`：纯文本表格，位置 / 速度随时间，方便导入 matlab、excel 分析


# `checkpoint0000.pth` 就是训练保存的模型权重文件

> 
> 是 PyTorch 标准 ckpt，包含网络参数，一般还附带优化器状态、迭代计数，用来**断点续训**和**评估仿真**。

## 几个关键点

1. **文件名 `checkpoint0000.pth`**
0000 通常代表**初始保存 / 第 0 个断点**，要看日志确认它对应的迭代步数；
后面继续训练会陆续生成 `checkpoint0001.pth`、`checkpoint0002.pth`。
2. 验证这个文件能不能加载（评估 / 续训命令模板）


# 断点继续训练
python main_cuda.py $(cat configs/single_agent.args) --resume ./checkpoint0000.pth

# 仅评估，不训练，生成轨迹
python main_cuda.py $(cat configs/single_agent.args) --resume ./checkpoint0000.pth --eval_only

`checkpoint0001.pth` 是**在 0000 保存时刻的基础上，继续跑更多迭代之后，新保存的快照**。