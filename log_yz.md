[TOC]

ubuntu 24.04复现NMI论文


# DiffPhysDrone 完整搭建文档（Ubuntu20.04 Docker容器）
> 目标：搭建可微分无人机仿真环境，编译自定义CUDA扩展 quadsim_cuda，成功运行单/多智能体训练
> 前置条件：宿主机已安装 docker + nvidia-container-toolkit，宿主机NVIDIA驱动支持CUDA11.8
> 容器名称：diffphysdrone

## 一、宿主机操作：拉取镜像 + 创建容器
宿主机终端执行
bash
docker pull ubuntu:20.04 # 拉取ubuntu20.04镜像


# 创建并启动容器，挂载源码、软件目录，开启GPU直通
docker run -it --gpus all \
-v /宿主机DiffPhysDrone源码路径:/workspace/DiffPhysDrone \
-v /宿主机software目录:/home/software \
--name diffphysdrone \
ubuntu:20.04 /bin/bash


### 再次进入容器（宿主机）


# 启动容器 + 交互式bash进入
docker start diffphysdrone && docker exec -it diffphysdrone /bin/bash
# 容器内退出：exit，仅退出shell，容器不停止
# ❌ 禁止 docker attach，Ctrl+C会直接杀死容器


## 二、容器内：基础系统依赖安装


apt update
apt install -y build-essential wget curl git


## 三、容器内：安装 Miniconda


# 下载安装脚本
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# 静默安装到 /opt/miniconda3
bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3


## 四、创建 Python3.11 conda 环境


# 临时初始化conda
eval "$(/opt/miniconda3/bin/conda shell.bash hook)"
# 创建py311环境
conda create -n py311 python=3.11 -y
# 激活环境
conda activate py311


## 五、安装 CUDA Toolkit 11.8（仅 nvcc 编译工具，不装显卡驱动）

> 
> 驱动复用宿主机 NVIDIA 驱动


# cuda安装包放在挂载目录 /home/software
sh /home/software/cuda_11.8.0_520.61.02_linux.run --toolkit --silent --override


校验 nvcc


nvcc -V
# 预期输出 release 11.8, V11.8.89


## 六、写入～/.bashrc 自动加载环境（交互式终端自动配置）


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


## 七、安装 PyTorch 2.2.2+cu118


conda activate py311
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu118


校验 torch 与 GPU


python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'GPU available:', torch.cuda.is_available())"


### 修复 numpy 版本冲突（Torch2.2.2 仅支持 numpy1.x）


pip uninstall numpy -y
pip install "numpy<2.0"


## 八、编译 DiffPhysDrone 自定义 CUDA 扩展 quadsim_cuda

> 
> ⚠️ 必须加 --no-build-isolation，否则构建沙盒找不到 torch


cd /workspace/DiffPhysDrone
pip install -e src --no-build-isolation


✅ 成功标志：`Successfully built quadsim_cuda` + `Successfully installed quadsim_cuda-0.0.0`

验证扩展导入


python -c "import quadsim_cuda; print('quadsim_cuda loaded OK')"


## 九、安装项目运行依赖


pip install matplotlib tensorboard tqdm scipy gym


## 十、启动训练脚本


# 单智能体
python main_cuda.py $(cat configs/single_agent.args)

# 多智能体
python main_cuda.py $(cat configs/multi_agent.args)


> 
> 说明：`torch.cross without specifying the dim arg` 为弃用警告，不影响仿真计算，可忽略。

## 十一、后台长时间训练（防止 SSH 断开中断任务）


nohup python main_cuda.py $(cat configs/single_agent.args) > train_single.log 2>&1 &
# 实时查看日志
tail -f train_single.log


Tensorboard 可视化查看 loss 曲线


tensorboard --logdir=runs --bind_all



踩坑记录

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


tensorboard --logdir=runs --bind_all


- `--bind_all`：允许宿主机浏览器访问（非常关键，不加只能容器内部访问）
- 启动后会输出地址，类似：`http://localhost:6006`

打开宿主机浏览器访问：


http://容器IP:6006


> 
> 如果是本地机器，容器 IP 可以用宿主机`localhost:6006`；远程服务器填服务器公网 IP:6006


宿主机终端执行（新开一个宿主机窗口！不要在正在跑训练的那个窗口输入）


docker exec -it diffphysdrone /bin/bash


### DiffPhysDrone 训练产出结果格式

1. 模型权重（最核心，用于断点续训、评估推理）

后缀：`.pth` / `.pt`，**PyTorch 标准序列化文件**

- 内部存储：网络权重、optimizer 状态、迭代次数、loss 记录等
- 读写方式：`torch.load()` / `torch.save()`
- 使用：`--resume xxx.pth` 加载做续训或者评估

> 
> 示例文件名：`model_50000.pth`，`ckpt_100000.pt`

2. TensorBoard 日志（loss 曲线数据）

存放目录：`runs/`，**不是图片，是一组 event 二进制文件**

- 文件名称类似：`events.out.tfevents.xxxx`
- 格式：TensorFlow protobuf 二进制日志
- 用途：TensorBoard 读取，绘制 loss、奖励等标量曲线
- 特点：不能直接双击打开；可以离线回看，也能用 python 脚本读取里面的标量数据导出 csv

 3. 文本日志（控制台输出）

- `train_single.log`：**纯文本 utf-8**
保存训练打印的 loss、迭代信息，可用`tail/grep`查看，可导出到 Excel 做离线绘图。

4. 评估输出（仅开启 eval/--eval_only 才会生成）

### 4.1 轨迹图片

`*.png` 静态图片（matplotlib 保存），**只有评估阶段才生成**，训练阶段默认没有。

### 4.2 仿真轨迹数据（可选，看代码实现）

很多时候代码会额外保存轨迹数值：

- `.npy`：numpy 二进制数组，存储无人机位置、速度、姿态时间序列，python 直接加载做后处理
- `.csv`：纯文本表格，位置 / 速度随时间，方便导入 matlab、excel 分析


### `checkpoint0000.pth` 就是训练保存的模型权重文件

> 
> 是 PyTorch 标准 ckpt，包含网络参数，一般还附带优化器状态、迭代计数，用来**断点续训**和**评估仿真**。

### 几个关键点

1. **文件名 `checkpoint0000.pth`**
0000 通常代表**初始保存 / 第 0 个断点**，要看日志确认它对应的迭代步数；
后面继续训练会陆续生成 `checkpoint0001.pth`、`checkpoint0002.pth`。
2. 验证这个文件能不能加载（评估 / 续训命令模板）


### 断点继续训练
python main_cuda.py $(cat configs/single_agent.args) --resume ./checkpoint0000.pth

### 仅评估，不训练，生成轨迹
python main_cuda.py $(cat configs/single_agent.args) --resume ./checkpoint0000.pth --eval_only

`checkpoint0001.pth` 是**在 0000 保存时刻的基础上，继续跑更多迭代之后，新保存的快照**。

可以，训练过程中支持做评估，两种方式

> 
> 评估 = 加载当前网络权重，跑仿真、算指标、生成轨迹图；**评估阶段不会更新网络参数，不会破坏训练**。


 GitHub Release 页面在哪里（网页端操作）

仓库地址：`https://github.com/JoeYao-bit/DiffPhysDrone`

1. 打开这个仓库主页
2. 页面右侧栏，找到 **Releases** 栏目（一般在 Star/Fork 下面），点进去就是 Release 页面
   - 分支`yz_20261014`只是代码分支，Release 是**整个仓库**的发布包，**不是在分支里面**
   - 很多人找不到：因为这个 fork 仓库**可能根本没有上传任何 Release 附件！**

> 
> 重点：`JoeYao-bit/DiffPhysDrone`是原作者的 fork 分支，**Release 在原始仓库 HenryHuYu/DiffPhysDrone**
> 原始仓库：[https://github.com/HenryHuYu/DiffPhysDrone](https://github.com/HenryHuYu/DiffPhysDrone)
> 👉 原始仓库右侧 Releases：[https://github.com/HenryHuYu/DiffPhysDrone/releases](https://github.com/HenryHuYu/DiffPhysDrone/releases)

> 
> 你 Readme 里面写的仿真验证包（LinuxNoEditor 虚幻引擎 Blocks.sh）是放在**原始仓库 Release 的附件**。

训练结果评估

Evaluation

You need to download the simulation validation code from the GitHub release page. To evaluate the trained model in multi-agent settings, use the following command to launch the simulator:

在<path to multi agent code supplementary>，

例如/home/yaozhuo/code/validation_code/swarm_v1/swarm/中运行脚本，唤起测试环境

cd <path to multi agent code supplementary>
./LinuxNoEditor/Blocks.sh -ResX=896 -ResY=504 -windowed -WinX=512 -WinY=304 -settings=$PWD/settings.json

测试指定模型(/home/yaozhuo/code/DiffPhysDrone/checkpoint0004.pth)

(airsim_eval) ➜  swarm python eval.py --resume /home/yaozhuo/code/DiffPhysDrone/checkpoint0004.pth --target_speed 2.5


high_speed_flight_v1是告诉测试，需要20.04相关ros版本

swarm_v1是多个机器人测试，不需要两行命令就能跑：
./LinuxNoEditor/Blocks.sh -ResX=896 -ResY=504 -windowed -WinX=512 -WinY=304 -settings=$PWD/settings.json
(airsim_eval) ➜  swarm python eval.py --resume /home/yaozhuo/code/DiffPhysDrone/checkpoint0004.pth --target_speed 2.5


配置文件settings.json解析

{
  "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/main/docs/settings.md",
  "SettingsVersion": 1.2,       // AirSim配置文件版本号，固定1.2
  "LogMessagesVisible": false,  // 关闭仿真引擎内部日志输出，减少刷屏
  "SimMode": "Multirotor",      // 仿真模式：多旋翼无人机（不是汽车）
  "ViewMode": "Fpv",            // 默认主视图：FPV第一人称视角
  "ClockSpeed": 0.25,           // ⭐【非常关键】仿真时间缩放系数：0.25 = 仿真速度是真实世界的1/4，慢放仿真。
  // 比如真实1s，仿真只前进0.25s；用于给神经网络推理留出足够计算时间，防止推理延迟导致控制丢包。

  "CameraDefaults": {
    "CaptureSettings": [
      {
        "ImageType": 1,         // ImageType=1 = 灰度图(Grayscale)
        "Width": 48,
        "Height": 36,
        "FOV_Degrees": 79       // 视场角79°，小分辨率，是模型输入尺寸！DiffPhysDrone网络输入就是48×36灰度图
      },
      {
        "ImageType": 3,         // ImageType=3 = 深度图(DepthPerspective)
        "FOV_Degrees": 79
      }
    ]
  },

  "Vehicles": { // 定义所有无人机，一共6架 drone_1 ~ drone_6
    "drone_1": {
      "VehicleType": "SimpleFlight", // SimpleFlight：AirSim内置简化多旋翼动力学模型（DiffPhysDrone用这个）
      "Cameras": {
        "front_center_custom": { // 每架无人机挂载同一个名字的相机 front_center_custom
          "X": 0.25, "Y": 0.0, "Z": -0.1, // 相机在机体坐标系位置：机头向前0.25m，向下0.1m
          "Pitch": 10, "Roll": 0, "Yaw": 0 // 相机向下俯10°
        }
      },
      "X": 0, "Y": 0, "Z": 0, // 【注意】这里写的坐标只是默认值！
      "Pitch": 0, "Roll": 0, "Yaw": 0
    },
    // drone_2 ~ drone_6 和drone_1完全一致，6架无人机硬件/相机配置完全相同
  },

  "ExternalCameras": { // 全局外部相机，不属于任何无人机，鸟瞰俯视BEV
    "BEV": {
      "CaptureSettings": [{
        "ImageType": 0, // 0=RGB图
        "Width": 512,
        "Height": 288,
        "OrthoWidth": 8,
        "ProjectionMode": "orthographic" // ⭐正交投影鸟瞰图，没有透视变形，用于可视化全局多机位置
      }],
      "X": 3.0, "Y": 0.0, "Z": -6.0,    // 相机位置，高度6米俯视
      "Pitch": -90, "Roll": 0, "Yaw": -90 // 镜头垂直朝下
    }
  },

  "SubWindows": [ // 就是你截图里底部3个小预览窗口！对应界面的三色框
    {"WindowID": 0, "CameraName": "front_center_custom", "ImageType": 3, "VehicleName": "drone_2", "Visible": true, "External": false},
    // 窗口0：drone_2机载相机，深度图（ImageType3），截图里绿色框
    {"WindowID": 1, "CameraName": "BEV", "ImageType": 0, "VehicleName": "", "Visible": true, "External": true},
    // 窗口1：全局BEV鸟瞰RGB图，截图黄色框
    {"WindowID": 2, "CameraName": "front_center_custom", "ImageType": 3, "VehicleName": "drone_6", "Visible": true, "External": false}
    // 窗口2：drone_6机载相机，深度图，截图紫色框
  ]
}


## eval.py(网络测试脚本)解析


import argparse
from datetime import datetime
import json
import math
import os
import random
import signal
import subprocess
from time import sleep, time
import airsim
from airsim.types import Pose, Vector3r, Quaternionr
from airsim.types import AngleLevelControllerGains, PIDGains, AngleRateControllerGains
import numpy as np
from tqdm import tqdm

===================== 视频录制工具类：保存深度图可视化视频 =====================
class VideoRecorder:
    def __init__(self, output, w, h, fps=15, pix_fmt='rgb24') -> None:
        # ffmpeg子进程，用于把图像帧写入视频文件
        self.p = None
        self.output = output
        # 构造ffmpeg命令，从管道读取raw图像流编码成视频
        command = [
            "/usr/bin/ffmpeg",
            '-y',  # 覆盖已有输出文件，不弹窗询问
            '-f', 'rawvideo',
            '-vcodec','rawvideo',
            '-s', f'{w}x{h}',  # 单帧图像分辨率
            '-pix_fmt', pix_fmt,
            '-r', f'{fps}',  # 帧率
            '-i', '-',  # 输入来源：标准输入管道
            '-s', f'{w//2*2}x{h//2*2}',
            '-an',  # 不编码音频
            '-loglevel', 'error', # 只输出错误信息，屏蔽多余日志
            '-pix_fmt', 'yuv420p'
        ]
        self.p = subprocess.Popen(command + [self.output], stdin=subprocess.PIPE)

    def add_image(self, image):
        # 写入一帧图像到ffmpeg管道
        self.p.stdin.write(image)

    def close(self):
        # 关闭管道并等待ffmpeg完成编码
        if self.p is not None:
            self.p.stdin.close()
            self.p.wait()

===================== 定频控制器：维持固定循环频率 =====================
class Rate:
    def __init__(self, hz) -> None:
        self.hz = hz
        self.t0 = time()

    def sleep(self):
        # 计算需要sleep的时间，保证循环稳定hz频率
        while True:
            to_sleep = 1 / self.hz - time() + self.t0
            if to_sleep < 0.01:
                break
            sleep(to_sleep)
        self.t0 += max(1 / self.hz, 0.5 / self.hz - to_sleep)

===================== 模型与数学工具函数 =====================
import torch
import torch.nn.functional as F
from model import Model

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    四元数 -> 旋转矩阵，来自pytorch3d
    Args:
        quaternions: 实部在前的四元数 tensor(...,4)
    Returns:
        旋转矩阵 tensor(...,3,3)
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),

            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),

            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

===================== 命令行参数解析 =====================
parser = argparse.ArgumentParser()
parser.add_argument('--resume', default='exps/run1/checkpoint0004.pth', help="模型权重文件路径")
parser.add_argument('--env', default='swap', help="环境任务名称，当前只有swap场景")
parser.add_argument('--target_speed', default=2, type=float, help='m/s，无人机期望飞行速度上限')
parser.add_argument('--margin', default=0.15, type=float, help='m，机体安全半径，给网络作为避障安全距离输入')
parser.add_argument('--smoothness', default=0.5, type=float, help='平滑项参数(代码里未实际使用)')
parser.add_argument('--clockspeed', default=0.25, type=float, help='仿真时间缩放，必须和settings.json ClockSpeed保持一致')
parser.add_argument('--sr', default=3, type=int, help='下采样倍数，深度图降采样给神经网络输入')
parser.add_argument('--no_odom', default=False, action='store_true', help='是否关闭里程计输入，关闭则模型输入维度减少')
args = parser.parse_args()
print(args)

 悬停基准油门，SimpleFlight动力学悬停所需基础油门
hover_thr = 0.297
 hover_thr = 0.593

 创建本次评估的日志文件夹
datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f'exps_{args.target_speed}/{datetime_str}/'
os.makedirs(log_dir)

 =====================【核心：任务定义，起点与目标航点】====================
 swap场景：列表内每个元组 (无人机名称, [起点,终点])
 坐标格式 [X,Y,Z] AirSim NED坐标系
 原版：左右两组无人机相向交叉飞行，swap交换位置，碰撞概率高
agents = {
"swap": [
        ("drone_1", [[ 0,  1.2, 0], [ 6, -1.2, 0]]),
        ("drone_2", [[ 0,  0.0, 0], [ 6,  0.0, 0]]),
        ("drone_3", [[ 0, -1.2, 0], [ 6,  1.2, 0]]),
        ("drone_4", [[ 6, -1.2, 0], [ 0,  1.2, 0]]),
        ("drone_5", [[ 6,  0.0, 0], [ 0,  0.0, 0]]),
        ("drone_6", [[ 6,  1.2, 0], [ 0, -1.2, 0]]),
    ],
}[args.env]

 深度图可视化视频保存器
depth_recorder = VideoRecorder(f'{log_dir}/depth.mp4', 16*args.sr, len(agents)*12*args.sr, pix_fmt='y8')

B = len(agents)  # B=6 无人机数量
agent_names, agent_waypoints = zip(*agents)
target_pos = [w[-1] for w in agent_waypoints] # 提取每架无人机最终目标终点

traj_history = {agent_name: [] for agent_name in agent_names} # 保存所有无人机轨迹，最后写入json

 连接AirSim仿真后端
client = airsim.MultirotorClient()
client.confirmConnection()
client.reset()

 模型加载到GPU
device = torch.device('cuda')
 模型输入维度：no_odom=7，带odom=10；输出6维动作
model = Model(7 if args.no_odom else 10, 6).eval().to(device)
if args.resume:
    model.load_state_dict(torch.load(args.resume, map_location=device))

@torch.no_grad()
def main():
    h = None
    # 预热模型，初始化LSTM/GRU隐藏状态
    for _ in range(10):
        _, _, h = model(
            torch.zeros(B, 1, 12, 16, device=device),
            torch.zeros(B, model.v_proj.in_features, device=device),
            h)
    sleep(1)
    # 初始化所有无人机：放置起点 + 设置初始机头yaw朝向目标 
    for agent_name, waypoints in agents:
        sleep(0.1)
        client.enableApiControl(True, agent_name)
        client.armDisarm(True, agent_name)
        client.moveByVelocityAsync(0, 0, 0, 0.5, vehicle_name=agent_name)
        # 计算初始yaw：机头预先对准【起点指向终点】的方向
        yaw = math.atan2(waypoints[1][1] - waypoints[0][1], waypoints[1][0] - waypoints[0][0])
        start_pt = waypoints.pop(0)
        # 在原始起点上增加小随机扰动，增加评估鲁棒性 ±0.1~±0.25
        start_pt = [
            start_pt[0] + random.random() * 0.2 - 0.1,
            start_pt[1] + random.random() * 0.2 - 0.1,
            start_pt[2] + random.random() * 0.5 - 0.25,
        ]
        # 把无人机放置到起点位置，忽略放置瞬间碰撞
        for _ in range(3):
            sleep(0.1)
            client.simSetVehiclePose(Pose(
                Vector3r(*start_pt),
                Quaternionr(0, 0, math.sin(yaw / 2), math.cos(yaw / 2))),
            ignore_collision=True, vehicle_name=agent_name)

        # 设置SimpleFlight多旋翼PID控制参数，对齐PX4风格
        client.setAngleRateControllerGains(AngleRateControllerGains(
            roll_gains=PIDGains(0.2, 0.01, 0.001),
            pitch_gains=PIDGains(0.2, 0.01, 0.001),
            yaw_gains=PIDGains(0.2, 0.01, 0.001),
        ))
        client.setAngleLevelControllerGains(AngleLevelControllerGains(
            roll_gains=PIDGains(2, 0, 0),
            pitch_gains=PIDGains(2, 0, 0),
            yaw_gains=PIDGains(2, 0, 0),
        ), agent_name)
    sleep(0.5)

    # 清空碰撞记录
    for agent_name, waypoints in agents:
        client.simGetCollisionInfo(agent_name)

    # 预分配张量存储状态
    p_target = torch.empty((B, 3)) # 每架无人机当前目标点坐标
    last_p = torch.empty((B, 3))   # 上一时刻位置
    forward_vec = torch.empty((B, 3))
    v = torch.empty((B, 3))       # 机体速度
    R = torch.empty((B, 3, 3))     # 世界->机体旋转矩阵

    traveled_distance = [0 for _ in agents] # 飞行总路程
    traveled_time = [0 for _ in agents]     # 任务耗时
    done_flag = [False for _ in agents]      # 是否到达目标标记
    has_collided = [set() for _ in agents]   # 碰撞记录集合

    extra = torch.tensor([[args.margin]]).repeat(B, 1) # 安全距离margin送入网络

    # 初始化每架无人机的第一个目标航点
    for i, (agent_name, waypoints) in enumerate(agents):
        x, y, z = waypoints.pop(0)
        p_target[i] = torch.as_tensor([x, -y, -z])
        # 获取初始位姿
        state = client.getMultirotorState(agent_name)
        q = state.kinematics_estimated.orientation
        p = state.kinematics_estimated.position
        q = torch.as_tensor([q.w_val, q.x_val, -q.y_val, -q.z_val])
        last_p[i] = torch.as_tensor([p.x_val, -p.y_val, -p.z_val])
        forward_vec[i] = quaternion_to_matrix(q)[:, 0]

    pbar = tqdm()
    hidden_state = None # 神经网络时序隐藏状态
    rate = Rate(15 * args.clockspeed) # 主循环频率 15Hz * 仿真时钟缩放
    t_begin_real = time()
    t_begin = state.timestamp / 1e9
    t_now = t_begin
    t_end = t_begin + 30 # 最大仿真时长30秒
    ctl_error = torch.randn((len(agents), 3)) * 0.17 # 控制扰动项

    # ====================== 主循环 15Hz 控制周期 ======================
    while t_now < t_end:
        pbar.update()
        # 读取每架无人机机载深度相机
        depth = []
        for agent in agent_names:
            responses = client.simGetImages([
                airsim.ImageRequest("front_center_custom", airsim.ImageType.DepthPlanar, True)
            ], agent)
            depth.append(airsim.get_pfm_array(responses[0]))
        depth = np.stack(depth)
        depth_viz = np.uint8(np.clip(depth / 10 * 255, 0, 255))
        depth_recorder.add_image(depth_viz.reshape(-1, 16*args.sr))

        # 读取所有无人机状态、更新轨迹记录
        for i, (agent_name, waypoints) in enumerate(agents):
            state = client.getMultirotorState(agent_name)
            t_now = state.timestamp / 1e9
            p = state.kinematics_estimated.position
            q = state.kinematics_estimated.orientation
            _v = state.kinematics_estimated.linear_velocity
            traj_history[agent_name].append([p.x_val, p.y_val, p.z_val, q.w_val, q.x_val, q.y_val, q.z_val])

            # 坐标转换，存入tensor
            p = torch.as_tensor([p.x_val, -p.y_val, -p.z_val])
            duration = t_now - t_begin
            if not done_flag[i]:
                traveled_distance[i] += torch.norm(p - last_p[i]).item()
                traveled_time[i] = duration
            last_p[i] = p
            v[i] = torch.as_tensor([_v.x_val, -_v.y_val, -_v.z_val])
            q = torch.as_tensor([q.w_val, q.x_val, -q.y_val, -q.z_val])
            R[i] = quaternion_to_matrix(q)

            # 判断是否到达当前目标点：距离<1.5m判定到达
            if not done_flag[i] and torch.norm(p_target[i] - p) < 1.5:
                if waypoints:
                    # 还有后续航点，切换到下一个目标
                    x, y, z = waypoints.pop(0)
                    p_target[i] = torch.as_tensor([x, -y, -z])
                else:
                    # 所有航点跑完，任务完成
                    print(f"{agent_names[i]} arrived in {duration}s!")
                    done_flag[i] = True
                    client.moveToPositionAsync(*target_pos[i], 0.5, vehicle_name=agent_names[i])
                    # 全部无人机到达，提前结束仿真
                    if all(done_flag):
                        t_end = t_now + 0.5

        # 计算目标速度向量：从当前位置指向目标点，限制最大速度target_speed
        target_v = p_target - last_p
        target_v_norm = torch.norm(target_v, 2, -1, keepdim=True)
        target_v = target_v / target_v_norm * target_v_norm.clamp_max(args.target_speed)

        # 姿态矩阵预处理，投影到水平面（忽略高度，只保留水平前向）
        env_R = R.clone()
        fwd = R[:, :, 0].clone()
        up = torch.zeros_like(fwd)
        fwd[:, 2] = 0
        up[:, 2] = 1
        fwd = fwd / torch.norm(fwd, 2, -1, keepdim=True)
        R = torch.stack([fwd, torch.cross(up, fwd), up], -1)

        # 组装网络输入状态：机体速度、目标速度、旋转矩阵、安全margin
        state = [torch.squeeze(target_v[:, None] @ R, 1), R[:, 2], extra]
        local_v = torch.squeeze(v[:, None] @ R, 1)
        if not args.no_odom:
            state.insert(0, local_v)
        state = torch.cat(state, -1)

        # 深度图预处理与下采样
        depth = torch.as_tensor(depth, device=device)[:, None]
        x = 3 / depth.clamp_(0.3, 24) - 0.6
        x = F.max_pool2d(x, (args.sr, args.sr))

        # 神经网络前向推理，输出控制action
        state = state.to(device)
        action, _, hidden_state = model(x, state, hidden_state)

        # 将机体局部坐标系action转换为世界坐标系加速度指令
        v_setpoint, v_est = (R @ action.cpu().reshape(B, 3, -1)).unbind(-1)
        a_setpoint = v_setpoint - v_est + ctl_error
        a_setpoint[:, 2] += 9.80665 # 补偿重力

        # 加速度 -> 期望姿态角 + 油门
        throttle = torch.norm(a_setpoint, 2, -1)
        up_vec = a_setpoint / throttle[..., None]
        throttle = throttle + local_v[:, 2] * local_v[:, 2].abs() * 0.01

        # 计算期望roll/pitch/yaw
        forward_vec = env_R[..., 0] * 5 + p_target - last_p
        forward_vec[:, 2] = (forward_vec[:, 0] * up_vec[:, 0] + forward_vec[:, 1] * up_vec[:, 1]) / -up_vec[:, 2]
        forward_vec /= torch.norm(forward_vec, 2, -1, True)
        left_vec = torch.cross(up_vec, forward_vec)
        roll = torch.atan2(left_vec[:, 2], up_vec[:, 2])
        pitch = torch.asin(-forward_vec[:, 2])
        yaw = torch.atan2(forward_vec[:, 1], forward_vec[:, 0])

        # 下发roll/pitch/yaw/throttle指令到AirSim
        for i, (r, p, y, t) in enumerate(zip(roll.tolist(), pitch.tolist(), yaw.tolist(), throttle.tolist())):
            if done_flag[i]:
                continue
            t = t / 9.8 * hover_thr
            client.moveByRollPitchYawThrottleAsync(r, p, y, t, 0.5, agent_names[i])
            # 碰撞检测
            collision_info = client.simGetCollisionInfo(agent_names[i])
            if collision_info.has_collided:
                has_collided[i].add(collision_info.object_name)
                print(f"{agent_names[i]} collide with {collision_info.object_name}!")

        # 维持循环频率
        clockspeed = duration / (time() - t_begin_real)
        rate.hz = 15 * clockspeed
        rate.sleep()

    # 保存评估日志文本
    with open(f"{log_dir}/log", 'w') as f:
        f.write(f'{args}\n')
        for i, (x, t) in enumerate(zip(traveled_distance, traveled_time)):
            f.write(f'ours,{args.env},{args.target_speed},{agent_names[i]},{x:.2f},{t:.2f},0,{done_flag[i]},{"_".join(has_collided[i])}\n')

if __name__ == '__main__':
    import shutil
    # 备份本次eval.py到日志目录，方便复现
    shutil.copy(__file__, f"{log_dir}/eval.py")

    # ========== 全局画面录屏（x11grab，无桌面环境会报错，不影响评估 ==========
    # 如果你不需要录屏，可以直接注释掉下面这一段ffmpeg启动代码
    ffmpeg_p = subprocess.Popen([
        '/usr/bin/ffmpeg',
        '-f', 'x11grab',
        '-video_size', '896x504',
        '-i', ':0+512,340',
        '-c:v', 'h264_nvenc',
        '-vf', f'setpts={args.clockspeed}*PTS',
        '-loglevel', 'error',
        '-an',
        f'{log_dir}/{datetime_str}.mp4'
    ], stdin=subprocess.PIPE)

    def cleanup():
        # 程序退出时执行：保存轨迹json，关闭视频录制
        with open(f"{log_dir}/traj_history.json", 'w') as f:
            json.dump(traj_history, f)
        ffmpeg_p.stdin.close()
        ffmpeg_p.wait()
        depth_recorder.close()
        print("start recording")
    try:
        main()
        ffmpeg_p.send_signal(signal.SIGINT)
    finally:
        cleanup()


### 测试日志保存目录

测试视频按照时间记录在swarm下面的文件夹：

/home/yaozhuo/code/validation_code/swarm_v1/swarm/exps_2.5/