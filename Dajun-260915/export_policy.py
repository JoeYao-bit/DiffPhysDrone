# """Export a trained policy to a flat binary the on-board C runtime can load.

# Deliberately not TorchScript/ONNX: the flight application is plain C with no
# runtime to host either, and the network is small enough (about 0.5M weights)
# that a hand-written forward pass is less machinery than a converter would be.

# Layout: a fixed-size header, then every tensor back to back as little-endian
# float32 in the order listed by MANIFEST. The header carries the shapes so a
# mismatched pair of files fails at load instead of flying on garbage.

# Usage:
#     python export_policy.py checkpoint0004.pth policy.bin
#     python export_policy.py checkpoint0004.pth policy.bin --testvec test.bin
# """
# from __future__ import annotations

# import argparse
# import struct
# import sys

# import torch

# MAGIC = 0x504F4C31  # 'POL1'
# VERSION = 1

# # (name, expected_shape_or_None). Order is the wire order; the C side reads
# # sequentially and never looks tensors up by name, so this must not be
# # reordered without bumping VERSION.
# MANIFEST = [
#     ("tof_stem.0.weight", None),   # Conv2d(C_tof, 16, 3, s2, p1)
#     ("tof_stem.2.weight", None),   # Conv2d(16, 32, 3, s2, p1)
#     ("tof_stem.4.weight", None),   # Conv2d(32, 64, 3, s2, p1)
#     ("tof_stem.7.weight", None),   # Linear(64*f*f, 192)
#     ("v_proj.weight", None),       # Linear(dim_obs, 192)
#     ("v_proj.bias", None),
#     ("gru.weight_ih", None),       # (576, 192)
#     ("gru.weight_hh", None),
#     ("gru.bias_ih", None),
#     ("gru.bias_hh", None),
#     ("fc.weight", None),           # (dim_action, 192)
# ]

# # Optional flow stem. Exported only when present so a ToF-only policy does not
# # carry 290K weights it will never touch.
# FLOW_MANIFEST = [
#     ("stem.0.weight", None),
#     ("stem.2.weight", None),
#     ("stem.4.weight", None),
#     ("stem.7.weight", None),
# ]


# def load_state_dict(path):
#     obj = torch.load(path, map_location="cpu", weights_only=False)
#     for key in ("model", "state_dict", "net"):
#         if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
#             return obj[key]
#     if not isinstance(obj, dict):
#         raise SystemExit(f"{path}: unexpected checkpoint structure {type(obj)}")
#     return obj


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("checkpoint")
#     ap.add_argument("output")
#     ap.add_argument("--testvec", help="also write a reference input/output pair "
#                                       "for the C runtime to check itself against")
#     args = ap.parse_args()

#     sd = load_state_dict(args.checkpoint)
#     sd = {k[len("module."):] if k.startswith("module.") else k: v
#           for k, v in sd.items()}

#     has_flow = "stem.0.weight" in sd
#     has_tof = "tof_stem.0.weight" in sd
#     if not has_tof:
#         raise SystemExit(
#             "checkpoint has no tof_stem -- it was trained with use_tof=False. "
#             "The on-board runtime is ToF-driven; retrain with --use_tof.")

#     tof_channels = sd["tof_stem.0.weight"].shape[1]
#     dim_obs = sd["v_proj.weight"].shape[1]
#     dim_action = sd["fc.weight"].shape[0]
#     hidden = sd["gru.weight_hh"].shape[1]
#     # Linear(64 * f * f, 192) -> recover the post-conv spatial size.
#     flat = sd["tof_stem.7.weight"].shape[1]
#     feat = int(round((flat // 64) ** 0.5))
#     if feat * feat * 64 != flat:
#         raise SystemExit(f"tof_stem head input {flat} is not 64*f*f")

#     manifest = list(MANIFEST)
#     if has_flow:
#         manifest = FLOW_MANIFEST + manifest

#     missing = [n for n, _ in manifest if n not in sd]
#     if missing:
#         raise SystemExit("checkpoint is missing: " + ", ".join(missing))

#     blobs = []
#     for name, _ in manifest:
#         t = sd[name].detach().to(torch.float32).contiguous()
#         blobs.append((name, t))

#     header = struct.pack(
#         "<IIIIIIIII",
#         MAGIC, VERSION,
#         1 if has_flow else 0,
#         tof_channels,
#         dim_obs, dim_action, hidden, feat,
#         len(blobs),
#     )
#     with open(args.output, "wb") as f:
#         f.write(header)
#         # Per-tensor: rank, dims[4], so the C side can assert every shape
#         # instead of trusting the header alone.
#         for name, t in blobs:
#             dims = list(t.shape) + [0] * (4 - t.dim())
#             f.write(struct.pack("<I4I", t.dim(), *dims))
#         for name, t in blobs:
#             f.write(t.numpy().tobytes())

#     total = sum(t.numel() for _, t in blobs)
#     print(f"已导出 {args.output}")
#     print(f"  光流分支: {'有' if has_flow else '无'}   ToF 通道: {tof_channels}")
#     print(f"  dim_obs={dim_obs} dim_action={dim_action} hidden={hidden} tof_feat={feat}x{feat}")
#     print(f"  张量 {len(blobs)} 个, 参数 {total} 个, {total*4} 字节权重")
#     for name, t in blobs:
#         print(f"    {name:<22} {tuple(t.shape)}")

#     if args.testvec:
#         write_testvec(args.testvec, sd, has_flow, tof_channels, dim_obs, hidden)


# def write_testvec(path, sd, has_flow, tof_channels, dim_obs, hidden):
#     """Deterministic input + the PyTorch output for it.

#     The C runtime replays this at startup; a mismatch means the port drifted
#     from the model it was exported from, which is the failure this whole
#     hand-written-forward-pass approach is most exposed to.
#     """
#     sys.path.insert(0, ".")
#     from model import Model

#     tof_size = 30
#     net = Model(dim_obs=dim_obs, dim_action=sd["fc.weight"].shape[0],
#                 use_tof=True, tof_size=tof_size, tof_channels=tof_channels)
#     net.load_state_dict(sd, strict=False)
#     net.eval()

#     g = torch.Generator().manual_seed(20260828)
#     tof = torch.rand((1, tof_channels, tof_size, tof_size), generator=g) * 2 - 1
#     v = torch.rand((1, dim_obs), generator=g) * 2 - 1
#     hx = torch.rand((1, hidden), generator=g) * 2 - 1
#     x = torch.zeros((1, 4, 12, 16))
#     if has_flow:
#         x = torch.rand((1, 4, 12, 16), generator=g) * 2 - 1

#     with torch.no_grad():
#         act, _, hx_out = net(x, v, tof, hx)

#     with open(path, "wb") as f:
#         f.write(struct.pack("<II", MAGIC, VERSION))
#         for t in (x, v, tof, hx, act, hx_out):
#             f.write(t.detach().to(torch.float32).contiguous().numpy().tobytes())
#     print(f"已导出参考向量 {path}")
#     print(f"  act    = {act.flatten().tolist()}")
#     print(f"  hx_out[:4] = {hx_out.flatten()[:4].tolist()}")


# if __name__ == "__main__":
#     main()


"""Export a trained policy to a flat binary the on-board C runtime can load.
Deliberately not TorchScript/ONNX: the flight application is plain C with no
runtime to host either, and the network is small enough (about 0.5M weights)
that a hand-written forward pass is less machinery than a converter would be.
Layout: a fixed-size header, then every tensor back to back as little-endian
float32 in the order listed by MANIFEST. The header carries the shapes so a
mismatched pair of files fails at load instead of flying on garbage.
Usage:
    python export_policy.py checkpoint0004.pth policy.bin
    python export_policy.py checkpoint0004.pth policy.bin --testvec test.bin
"""
from __future__ import annotations
import argparse
import struct
import sys
import torch

# 文件魔数标识，固定字符串'POL1'，C端加载时首先校验这个值，判断是不是合法权重bin文件
MAGIC = 0x504F4C31  # 'POL1'
# 二进制文件格式版本号；一旦修改MANIFEST张量顺序，必须升级VERSION，防止C端解析错乱
VERSION = 1

# 权重清单MANIFEST：(权重名称, 期望形状或None)
# 【重要】顺序就是二进制写入顺序；C侧是按顺序串行读取，**不是按名字查找张量**
# 如果修改这个列表顺序，必须升级VERSION版本号，否则C程序读出来的张量全部错位
MANIFEST = [
    ("tof_stem.0.weight", None),   # Conv2d(C_tof, 16, 3, s2, p1) 深度图TOF分支第一层卷积权重
    ("tof_stem.2.weight", None),   # Conv2d(16, 32, 3, s2, p1) TOF分支第二层卷积权重
    ("tof_stem.4.weight", None),   # Conv2d(32, 64, 3, s2, p1) TOF分支第三层卷积权重
    ("tof_stem.7.weight", None),   # Linear(64*f*f, 192) TOF卷积展平后全连接层权重
    ("v_proj.weight", None),       # Linear(dim_obs, 192) 状态观测向量投影层权重
    ("v_proj.bias", None),         # v_proj对应的偏置项
    ("gru.weight_ih", None),       # GRU输入权重 (576, 192)
    ("gru.weight_hh", None),       # GRU隐状态循环权重
    ("gru.bias_ih", None),         # GRU输入部分偏置
    ("gru.bias_hh", None),         # GRU隐状态循环部分偏置
    ("fc.weight", None),           # (dim_action, 192) 输出动作全连接层，输出期望加速度指令
]

# 可选光流分支权重清单
# 只有checkpoint包含光流分支权重才导出；纯TOF策略不需要导出这部分，减少机载无用权重（约290K参数）
FLOW_MANIFEST = [
    ("stem.0.weight", None),
    ("stem.2.weight", None),
    ("stem.4.weight", None),
    ("stem.7.weight", None),
]


def load_state_dict(path):
    """加载pth checkpoint，自动提取网络state_dict
    兼容多种checkpoint保存格式：有些ckpt把网络放在model/state_dict/net键下
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)
    # 尝试从ckpt字典里找到网络权重的字典
    for key in ("model", "state_dict", "net"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            return obj[key]
    # 不是字典格式，直接报错
    if not isinstance(obj, dict):
        raise SystemExit(f"{path}: unexpected checkpoint structure {type(obj)}")
    # 本身就是纯state_dict字典，直接返回
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")   # 输入：训练得到的pth权重文件路径
    ap.add_argument("output")        # 输出：导出的policy.bin二进制权重文件，供机载C运行时加载
    ap.add_argument("--testvec", help="also write a reference input/output pair "
    "for the C runtime to check itself against")  # 可选：导出测试向量bin，用来校验C手写前向传播是否和PyTorch完全一致
    args = ap.parse_args()

    sd = load_state_dict(args.checkpoint)
    # 去除权重名前缀 module. （多卡DDP训练保存的ckpt会带module.前缀，C端模型没有这个前缀）
    sd = {k[len("module."):] if k.startswith("module.") else k: v
          for k, v in sd.items()}

    # 判断权重是否包含光流分支、TOF深度分支
    has_flow = "stem.0.weight" in sd
    has_tof = "tof_stem.0.weight" in sd
    # 本机载C运行时只支持TOF输入的策略，如果没有TOF分支直接退出
    if not has_tof:
        raise SystemExit(
            "checkpoint has no tof_stem -- it was trained with use_tof=False. "
            "The on-board runtime is ToF-driven; retrain with --use_tof.")
    
    # 从权重shape自动推断模型超参
    tof_channels = sd["tof_stem.0.weight"].shape[1]  # TOF深度图输入通道数
    dim_obs = sd["v_proj.weight"].shape[1]           # 观测向量维度
    dim_action = sd["fc.weight"].shape[0]            # 输出动作维度（三轴加速度）
    hidden = sd["gru.weight_hh"].shape[1]            # GRU隐状态维度

    # 反推卷积之后特征图空间尺寸feat
    # tof_stem.7.weight: Linear(64*f*f,192), 输入维度=64*feat*feat
    flat = sd["tof_stem.7.weight"].shape[1]
    feat = int(round((flat // 64) ** 0.5))
    if feat * feat * 64 != flat:
        raise SystemExit(f"tof_stem head input {flat} is not 64*f*f")

    # 合并权重清单：如果开启光流分支，光流权重排在TOF权重前面（C端按顺序读）
    manifest = list(MANIFEST)
    if has_flow:
        manifest = FLOW_MANIFEST + manifest
    
    # 检查：清单内所有权重必须在state_dict中存在，缺失直接报错
    missing = [n for n, _ in manifest if n not in sd]
    if missing:
        raise SystemExit("checkpoint is missing: " + ", ".join(missing))

    blobs = []
    for name, _ in manifest:
        # 取出张量，转为float32，内存连续，准备序列化
        t = sd[name].detach().to(torch.float32).contiguous()
        blobs.append((name, t))

    # ========== 写入文件头部Header（小端序） ==========
    # 打包固定头部：MAGIC, VERSION, has_flow, tof_channels, dim_obs, dim_action, hidden, feat, 张量总数量
    header = struct.pack(
        "<IIIIIIIII",
        MAGIC, VERSION,
        1 if has_flow else 0,
        tof_channels,
        dim_obs, dim_action, hidden, feat,
        len(blobs),
    )
    with open(args.output, "wb") as f:
        f.write(header)
        # 每个张量附加shape描述：维度数量 + 最多4个维度；C端读取时校验shape，防止加载错误权重
        for name, t in blobs:
            dims = list(t.shape) + [0] * (4 - t.dim())  # 不足4维补0占位
            f.write(struct.pack("<I4I", t.dim(), *dims))
        # 把所有张量的二进制字节依次写入bin文件（float32小端）
        for name, t in blobs:
            f.write(t.numpy().tobytes())

    # 打印导出统计信息
    total = sum(t.numel() for _, t in blobs)
    print(f"已导出 {args.output}")
    print(f"  光流分支: {'有' if has_flow else '无'}   ToF 通道: {tof_channels}")
    print(f"  dim_obs={dim_obs} dim_action={dim_action} hidden={hidden} tof_feat={feat}x{feat}")
    print(f"  张量 {len(blobs)} 个, 参数 {total} 个, {total*4} 字节权重")
    for name, t in blobs:
        print(f"    {name:<22} {tuple(t.shape)}")

    # 如果指定--testvec，生成参考测试向量，用来验证C手写前向传播
    if args.testvec:
        write_testvec(args.testvec, sd, has_flow, tof_channels, dim_obs, hidden)


def write_testvec(path, sd, has_flow, tof_channels, dim_obs, hidden):
    """Deterministic input + the PyTorch output for it.
    The C runtime replays this at startup; a mismatch means the port drifted
    from the model it was exported from, which is the failure this whole
    hand-written-forward-pass approach is most exposed to.
    功能：生成一组固定随机输入，运行PyTorch网络，保存输入+网络输出结果到test.bin
    机载C程序启动时，用这组输入跑一遍手写的前向代码；
    如果C输出和PyTorch保存的参考结果不一致：代表C端网络实现和PyTorch原版有bug（算子实现差异、精度错误）
    """
    sys.path.insert(0, ".")
    from model import Model
    tof_size = 30  # TOF深度图输入分辨率 30x30
    # 重建模型实例，加载权重
    net = Model(dim_obs=dim_obs, dim_action=sd["fc.weight"].shape[0],
                use_tof=True, tof_size=tof_size, tof_channels=tof_channels)
    net.load_state_dict(sd, strict=False)
    net.eval()

    # 固定随机种子，保证输入完全可复现
    g = torch.Generator().manual_seed(20260828)
    # 构造固定测试输入：TOF深度图、观测向量v、GRU初始隐状态hx
    tof = torch.rand((1, tof_channels, tof_size, tof_size), generator=g) * 2 - 1
    v = torch.rand((1, dim_obs), generator=g) * 2 - 1
    hx = torch.rand((1, hidden), generator=g) * 2 - 1
    x = torch.zeros((1, 4, 12, 16))
    if has_flow:
        x = torch.rand((1, 4, 12, 16), generator=g) * 2 - 1
    
    # 推理，拿到PyTorch标准输出
    with torch.no_grad():
        act, _, hx_out = net(x, v, tof, hx)
    
    # 写入testvec二进制文件，保存输入、参考输出，供C端自检
    with open(path, "wb") as f:
        f.write(struct.pack("<II", MAGIC, VERSION))
        for t in (x, v, tof, hx, act, hx_out):
            f.write(t.detach().to(torch.float32).contiguous().numpy().tobytes())
    print(f"已导出参考向量 {path}")
    print(f"  act    = {act.flatten().tolist()}")
    print(f"  hx_out[:4] = {hx_out.flatten()[:4].tolist()}")


if __name__ == "__main__":
    main()
