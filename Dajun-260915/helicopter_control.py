# """Runtime helpers for the RGB optical-flow helicopter policy.

# The policy uses the simulator convention FLU (forward, left, up), while
# ArduPilot uses FRD for body vectors and NED for local navigation vectors.
# This module keeps those conversions explicit and integrates the learned
# acceleration into the velocity target sent to ArduPilot.
# """

# from __future__ import annotations

# import torch
# from torch.nn import functional as F


# def prepare_dual_flow(
#     flow: torch.Tensor,
#     flow_clip: float = 16.0,
#     simulated_size: tuple[int, int] = (48, 64),
# ) -> torch.Tensor:
#     """Convert real-camera flow to the policy's 4x12x16 input.

#     Args:
#         flow: Bx2xHxW flow in source-image pixels/frame. Channel 0 is
#             horizontal displacement (right positive), channel 1 is vertical
#             displacement (down positive).
#         flow_clip: displacement in simulated 64x48 pixels mapped to magnitude 1.
#         simulated_size: resolution used by the training renderer.
#     """
#     if flow.ndim != 4 or flow.shape[1] != 2:
#         raise ValueError("flow must have shape Bx2xHxW")

#     sim_h, sim_w = simulated_size
#     src_h, src_w = flow.shape[-2:]
#     scaled = F.interpolate(flow, size=simulated_size, mode="area")
#     scaled = scaled.clone()
#     scaled[:, 0].mul_(sim_w / src_w)
#     scaled[:, 1].mul_(sim_h / src_h)

#     full = F.interpolate(scaled, size=(12, 16), mode="area")
#     central = scaled[:, :, sim_h // 4:3 * sim_h // 4,
#                      sim_w // 4:3 * sim_w // 4]
#     central = F.interpolate(central, size=(12, 16), mode="area")
#     dual = torch.cat([full, central], 1)
#     return dual.clamp(-flow_clip, flow_clip) / flow_clip


# TOF_DIRECTIONS = ('front', 'left', 'right')


# def prepare_tof(
#     range_m_front: torch.Tensor,
#     range_m_left: torch.Tensor,
#     range_m_right: torch.Tensor,
#     min_range: float = 0.15,
#     max_range: float = 19.8,
# ) -> torch.Tensor:
#     """Convert the three raw ToF range grids (meters) to the policy's Bx3x30x30 input.

#     Matches env_cuda.py's render_tof() + main_cuda.py's normalization: a
#     linear range encoding, +1 at min range, -1 at max range / no return.
#     Channel order is fixed (front, left, right) -- see TOF_DIRECTIONS --
#     matching training; don't reorder these.

#     Args:
#         range_m_front/left/right: Bx30x30 (or BxHxW matching each module's
#             native grid) range reading in meters. Pixels with no valid
#             return should already be set to max_range (the module's own
#             no-target behavior, or clamp it yourself before calling this).
#         min_range, max_range: the modules' shared datasheet range limits --
#             must match the values env_cuda.py was trained with (default
#             0.15m / 19.8m).
#     """
#     channels = []
#     for range_m in (range_m_front, range_m_left, range_m_right):
#         if range_m.ndim != 3:
#             raise ValueError("each range grid must have shape BxHxW")
#         range_m = range_m.clamp(min_range, max_range)
#         channels.append(1 - 2 * (range_m - min_range) / (max_range - min_range))
#     return torch.stack(channels, 1)


# def policy_acceleration(output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#     """Decode the six policy outputs into bounded FLU acceleration and velocity."""
#     if output.ndim != 2 or output.shape[1] != 6:
#         raise ValueError("policy output must have shape Bx6")
#     accel_raw, velocity_pred = output.reshape(-1, 3, 2).unbind(-1)
#     scale = output.new_tensor([1.0, 1.0, 1.0])
#     return torch.tanh(accel_raw) * scale, velocity_pred


# def yaw_only_rotation_frd_to_ned(yaw_ned: torch.Tensor) -> torch.Tensor:
#     """Build the level-flight (yaw-only) body-FRD -> NED rotation.

#     The training frame (see the ``R`` built in main_cuda.py's rollout loop)
#     always treats body 'up' as world-vertical and derives the local FLU axes
#     from the horizontal projection of heading only -- roll and pitch are
#     discarded. The action decoded by ``policy_acceleration`` and integrated
#     by ``AccelerationVelocityIntegrator`` was trained against that yaw-only
#     frame, so the deployment-side FRD<->NED transform must use the same
#     yaw-only rotation, not the vehicle's full 3-axis attitude. Passing a
#     full-attitude DCM here would inject roll/pitch-dependent errors into the
#     commanded acceleration whenever the airframe is banked or pitched --
#     harmless while the vertical/lateral authority was zero, but not once
#     those axes carry real commands.

#     Args:
#         yaw_ned: heading in radians, measured from North, positive towards
#             East (ArduPilot's yaw/heading convention), shape (...,).
#     """
#     cos_yaw = torch.cos(yaw_ned)
#     sin_yaw = torch.sin(yaw_ned)
#     zeros = torch.zeros_like(yaw_ned)
#     ones = torch.ones_like(yaw_ned)
#     return torch.stack([
#         cos_yaw, -sin_yaw, zeros,
#         sin_yaw,  cos_yaw, zeros,
#         zeros,    zeros,   ones,
#     ], -1).reshape(*yaw_ned.shape, 3, 3)


# class AccelerationVelocityIntegrator:
#     """Integrate policy acceleration into an ArduPilot NED velocity + yaw target."""

#     def __init__(self, tracking_gain: float = 0.0, yaw_inertia: float = 5.0):
#         self.tracking_gain = tracking_gain
#         # Matches the fixed `5` passed to quadsim_cuda.update_state_vec in
#         # main_cuda.py/env_cuda.py -- how strongly the target heading favors
#         # the current heading over the commanded velocity direction. This
#         # keeps the yaw target from jittering when the velocity command is
#         # near zero (hover), the same way training's yaw dynamics do.
#         self.yaw_inertia = yaw_inertia
#         self.velocity_ned: torch.Tensor | None = None

#     def reset(self, ekf_velocity_ned: torch.Tensor) -> None:
#         self.velocity_ned = ekf_velocity_ned.detach().clone()

#     def update(
#         self,
#         acceleration_flu: torch.Tensor,
#         ekf_velocity_ned: torch.Tensor,
#         yaw_ned: torch.Tensor,
#         dt: float,
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         """Return (NED velocity command, target heading in radians).

#         Call reset when autonomous mode starts.

#         Args:
#             yaw_ned: current heading in radians (North=0, East=+90deg), e.g.
#                 from ArduPilot's ATTITUDE.yaw. Deliberately a heading angle
#                 rather than a full attitude rotation -- see
#                 ``yaw_only_rotation_frd_to_ned`` for why roll/pitch must not
#                 be folded into this transform.

#         The returned heading follows env_cuda.py's update_state_vec: blend
#         the current measured heading with the commanded horizontal velocity
#         direction, weighted by ``yaw_inertia`` towards the current heading.
#         Send it to ArduPilot unsmoothed -- ArduPilot's own yaw-rate
#         controller supplies the physical turn-rate lag that
#         update_state_vec applies in simulation via the randomized
#         `yaw_ctl_delay` (mean time constant ~1/6 s). Pre-smoothing this
#         target in software would lag the heading twice and drift from what
#         the policy was trained against; instead tune ArduPilot's yaw
#         response to settle in about that time constant.
#         """
#         if self.velocity_ned is None:
#             self.reset(ekf_velocity_ned)
#         if dt <= 0 or dt > 0.25:
#             raise ValueError(f"invalid control interval: {dt}")

#         rotation_frd_to_ned = yaw_only_rotation_frd_to_ned(yaw_ned)

#         # Policy FLU -> ArduPilot body FRD -> navigation NED.
#         accel_frd = acceleration_flu * acceleration_flu.new_tensor([1., -1., -1.])
#         accel_ned = torch.squeeze(
#             rotation_frd_to_ned @ accel_frd[..., None], -1)
#         self.velocity_ned = (
#             self.velocity_ned
#             + accel_ned * dt
#             + self.tracking_gain * (ekf_velocity_ned - self.velocity_ned) * dt
#         )

#         # Enforce asymmetric limits in the current body frame.
#         velocity_frd = torch.squeeze(
#             rotation_frd_to_ned.transpose(-1, -2)
#             @ self.velocity_ned[..., None], -1)
#         velocity_flu = velocity_frd * velocity_frd.new_tensor([1., -1., -1.])
#         lower = velocity_flu.new_tensor([-1.0, -1.0, -1.0])
#         upper = velocity_flu.new_tensor([ 3.0,  1.0,  1.0])
#         velocity_flu = torch.maximum(torch.minimum(velocity_flu, upper), lower)
#         velocity_frd = velocity_flu * velocity_flu.new_tensor([1., -1., -1.])
#         self.velocity_ned = torch.squeeze(
#             rotation_frd_to_ned @ velocity_frd[..., None], -1)

#         # Target heading: current heading blended with the commanded
#         # horizontal velocity direction, weighted towards "keep current
#         # heading" by yaw_inertia -- mirrors update_state_vec's
#         # `forward * yaw_inertia + v_pred` before normalization, so the
#         # target holds still near hover instead of chasing velocity noise.
#         forward = torch.stack([torch.cos(yaw_ned), torch.sin(yaw_ned)], -1)
#         target_dir = forward * self.yaw_inertia + self.velocity_ned[..., :2]
#         yaw_target = torch.atan2(target_dir[..., 1], target_dir[..., 0])

#         return self.velocity_ned, yaw_target

"""Runtime helpers for the RGB optical-flow helicopter policy.
The policy uses the simulator convention FLU (forward, left, up), while
ArduPilot uses FRD for body vectors and NED for local navigation vectors.
This module keeps those conversions explicit and integrates the learned
acceleration into the velocity target sent to ArduPilot.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F


def prepare_dual_flow(
    flow: torch.Tensor,
    flow_clip: float = 16.0,
    simulated_size: tuple[int, int] = (48, 64),
) -> torch.Tensor:
    """Convert real-camera flow to the policy's 4x12x16 input.
    Args:
        flow: Bx2xHxW flow in source-image pixels/frame. Channel 0 is
            horizontal displacement (right positive), channel 1 is vertical
            displacement (down positive).
        flow_clip: displacement in simulated 64x48 pixels mapped to magnitude 1.
        simulated_size: resolution used by the training renderer.
    """
    # 校验光流张量维度：必须是B×2×H×W，2通道分别为水平、垂直光流
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError("flow must have shape Bx2xHxW")
    sim_h, sim_w = simulated_size
    src_h, src_w = flow.shape[-2:]

    # 第一步：把原始相机光流缩放到仿真训练时的分辨率(48,64)
    scaled = F.interpolate(flow, size=simulated_size, mode="area")
    scaled = scaled.clone()
    # 坐标缩放校正：插值后像素位移需要按分辨率比例换算
    scaled[:, 0].mul_(sim_w / src_w)
    scaled[:, 1].mul_(sim_h / src_h)

    # full分支：整张48×64光流图下采样到网络输入尺寸12×16
    full = F.interpolate(scaled, size=(12, 16), mode="area")
    # central分支：取图像中心区域（裁剪中间1/2区域），再缩放到12×16
    central = scaled[:, :, sim_h // 4:3 * sim_h // 4, sim_w // 4:3 * sim_w // 4]
    central = F.interpolate(central, size=(12, 16), mode="area")

    # 拼接full+central，得到4通道（2+2）的光流输入：B×4×12×16，和训练网络输入保持一致
    dual = torch.cat([full, central], 1)
    # 截断幅值，归一化到[-1,1]，和仿真训练预处理逻辑完全对齐
    return dual.clamp(-flow_clip, flow_clip) / flow_clip


TOF_DIRECTIONS = ('front', 'left', 'right')
def prepare_tof(
    range_m_front: torch.Tensor,
    range_m_left: torch.Tensor,
    range_m_right: torch.Tensor,
    min_range: float = 0.15,
    max_range: float = 19.8,
) -> torch.Tensor:
    """Convert the three raw ToF range grids (meters) to the policy's Bx3x30x30 input.
    Matches env_cuda.py's render_tof() + main_cuda.py's normalization: a
    linear range encoding, +1 at min range, -1 at max range / no return.
    Channel order is fixed (front, left, right) -- see TOF_DIRECTIONS --
    matching training; don't reorder these.
    Args:
        range_m_front/left/right: Bx30x30 (or BxHxW matching each module's
            native grid) range reading in meters. Pixels with no valid
            return should already be set to max_range (the module's own
            no-target behavior, or clamp it yourself before calling this).
        min_range, max_range: the modules' shared datasheet range limits --
            must match the values env_cuda.py was trained with (default
            0.15m / 19.8m).
    """
    # 把三路ToF深度图（前、左、右）转为网络输入B×3×30×30张量
    channels = []
    for range_m in (range_m_front, range_m_left, range_m_right):
        if range_m.ndim != 3:
            raise ValueError("each range grid must have shape BxHxW")
        # 限制测距在[0.15m,19.8m]，超量程/无回波统一置为max_range
        range_m = range_m.clamp(min_range, max_range)
        # 归一化公式：近距离=+1，远距离/无回波=-1，和仿真训练编码完全一致
        channels.append(1 - 2 * (range_m - min_range) / (max_range - min_range))
    # 堆叠三路ToF，通道顺序固定 front/left/right，严禁调换顺序
    return torch.stack(channels, 1)


def policy_acceleration(output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode the six policy outputs into bounded FLU acceleration and velocity pred."""
    # 网络输出是B×6向量，解码：前3维加速度，后3维预测速度
    if output.ndim != 2 or output.shape[1] != 6:
        raise ValueError("policy output must have shape Bx6")
    # reshape拆分成 [accel_raw(3), velocity_pred(3)]
    accel_raw, velocity_pred = output.reshape(-1, 3, 2).unbind(-1)
    scale = output.new_tensor([1.0, 1.0, 1.0])
    # tanh把加速度限制在[-1,1]，乘以scale得到FLU机体坐标系下期望加速度
    return torch.tanh(accel_raw) * scale, velocity_pred


def yaw_only_rotation_frd_to_ned(yaw_ned: torch.Tensor) -> torch.Tensor:
    """Build the level-flight (yaw-only) body-FRD -> NED rotation.
    The training frame (see the ``R`` built in main_cuda.py's rollout loop)
    always treats body 'up' as world-vertical and derives the local FLU axes
    from the horizontal projection of heading only -- roll and pitch are
    discarded. The action decoded by ``policy_acceleration`` and integrated
    by ``AccelerationVelocityIntegrator`` was trained against that yaw-only
    frame, so the deployment-side FRD<->NED transform must use the same
    yaw-only rotation, not the vehicle's full 3-axis attitude. Passing a
    full-attitude DCM here would inject roll/pitch-dependent errors into the
    commanded acceleration whenever the airframe is banked or pitched --
    harmless while the vertical/lateral authority was zero, but not once
    those axes carry real commands.
    Args:
        yaw_ned: heading in radians, measured from North, positive towards
            East (ArduPilot's yaw/heading convention), shape (...,).
    """
    # 【重点】仅用偏航角yaw构造旋转矩阵，忽略roll/pitch
    # 论文仿真里就是这么做的！不要用完整姿态DCM，否则倾斜时机体坐标转换会引入误差
    cos_yaw = torch.cos(yaw_ned)
    sin_yaw = torch.sin(yaw_ned)
    zeros = torch.zeros_like(yaw_ned)
    ones = torch.ones_like(yaw_ned)
    # FRD机体坐标系 → NED地面坐标系的旋转矩阵
    return torch.stack([
        cos_yaw, -sin_yaw, zeros,
        sin_yaw,  cos_yaw, zeros,
        zeros,    zeros,   ones,
    ], -1).reshape(*yaw_ned.shape, 3, 3)


class AccelerationVelocityIntegrator:
    """Integrate policy acceleration into an ArduPilot NED velocity + yaw target."""
    def __init__(self, tracking_gain: float = 0.0, yaw_inertia: float = 5.0):
        self.tracking_gain = tracking_gain
        # 和仿真env_cuda里yaw惯性参数保持一致；悬停时抑制yaw抖动
        self.yaw_inertia = yaw_inertia
        # 积分保存NED坐标系下当前速度指令
        self.velocity_ned: torch.Tensor | None = None

    def reset(self, ekf_velocity_ned: torch.Tensor) -> None:
        """自主模式启动时重置积分器，用EKF当前NED速度初始化"""
        self.velocity_ned = ekf_velocity_ned.detach().clone()

    def update(
        self,
        acceleration_flu: torch.Tensor,
        ekf_velocity_ned: torch.Tensor,
        yaw_ned: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (NED velocity command, target heading in radians).
        Call reset when autonomous mode starts.
        Args:
            yaw_ned: current heading in radians (North=0, East=+90deg), e.g.
                from ArduPilot's ATTITUDE.yaw. Deliberately a heading angle
                rather than a full attitude rotation -- see
                ``yaw_only_rotation_frd_to_ned`` for why roll/pitch must not
                be folded into this transform.
        The returned heading follows env_cuda.py's update_state_vec: blend
        the current measured heading with the commanded horizontal velocity
        direction, weighted by ``yaw_inertia`` towards the current heading.
        Send it to ArduPilot unsmoothed -- ArduPilot's own yaw-rate
        controller supplies the physical turn-rate lag that
        update_state_vec applies in simulation via the randomized
        `yaw_ctl_delay` (mean time constant ~1/6 s). Pre-smoothing this
        target in software would lag the heading twice and drift from what
        the policy was trained against; instead tune ArduPilot's yaw
        response to settle in about that time constant.
        """
        if self.velocity_ned is None:
            self.reset(ekf_velocity_ned)
        if dt <= 0 or dt > 0.25:
            raise ValueError(f"invalid control interval: {dt}")

        # 1. 获取FRD→NED旋转矩阵（只用yaw，忽略roll/pitch）
        rotation_frd_to_ned = yaw_only_rotation_frd_to_ned(yaw_ned)
        # 2. 坐标系转换：Policy输出 FLU → ArduPilot FRD
        # FLU:前、左、上；FRD:前、右、下；左右、上下符号取反
        accel_frd = acceleration_flu * acceleration_flu.new_tensor([1., -1., -1.])
        # FRD加速度转到世界NED坐标系
        accel_ned = torch.squeeze(rotation_frd_to_ned @ accel_frd[..., None], -1)

        # 3. 加速度积分得到NED速度指令，增加跟踪项
        self.velocity_ned = (
            self.velocity_ned
            + accel_ned * dt
            + self.tracking_gain * (ekf_velocity_ned - self.velocity_ned) * dt
        )

        # 4. 机体坐标系速度限幅（关键：在机体FLU空间做上下/左右/前后速度硬限幅）
        # NED速度转回机体FRD，再转FLU做限幅
        velocity_frd = torch.squeeze(rotation_frd_to_ned.transpose(-1, -2) @ self.velocity_ned[..., None], -1)
        velocity_flu = velocity_frd * velocity_frd.new_tensor([1., -1., -1.])
        # 速度边界：前向最大3m/s，左右/上下±1m/s，和仿真训练约束一致
        lower = velocity_flu.new_tensor([-1.0, -1.0, -1.0])
        upper = velocity_flu.new_tensor([ 3.0,  1.0,  1.0])
        velocity_flu = torch.maximum(torch.minimum(velocity_flu, upper), lower)
        # 限幅完成，转回FRD，再转回NED，下发给飞控
        velocity_frd = velocity_flu * velocity_flu.new_tensor([1., -1., -1.])
        self.velocity_ned = torch.squeeze(rotation_frd_to_ned @ velocity_frd[..., None], -1)

        # 5. 计算目标偏航角yaw_target：混合【当前机头朝向】+【期望水平速度方向】
        # yaw_inertia越大，机头越倾向维持原有朝向，抑制悬停时yaw抖动
        forward = torch.stack([torch.cos(yaw_ned), torch.sin(yaw_ned)], -1)
        target_dir = forward * self.yaw_inertia + self.velocity_ned[..., :2]
        yaw_target = torch.atan2(target_dir[..., 1], target_dir[..., 0])
        # 返回：NED速度指令 + 目标偏航角，下发给ArduPilot
        return self.velocity_ned, yaw_target
