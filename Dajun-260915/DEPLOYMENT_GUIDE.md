# 新无人机完整部署与验收手册

更新：2026-10-22。适用：本项目 AR 系列算力板、三路 ToF、相机 IMU、ArduPilot 飞控及地面图传板。本文整理软件安装流程，不表示已向当前连接的新飞机完成部署，也不表示已完成飞行验收。

## 1. 部署范围与当前版本边界

| 模块 | 电脑源码 | 板端位置 | 作用 |
|---|---|---|---|
| 采集及飞控桥接 | SDK/base/customer/apps/ar_lowdelay | /usrdata/uav/capture | ToF、相机、IMU 采集，MAVLink 收发 |
| ToF 定位 | tof_odometry/live | /usrdata/uav/tof | 点云匹配及位姿输出 |
| 定位服务入口 | 当前选定的 ToF 启动配置 | /usrdata/uav/vins | 服务名保留 vins，实际可以运行 tof_service |
| 建图与规划 | nano_explorer-0916 | /usrdata/uav/explorer | 建图、探索、悬停及单目标测试 |
| 共同协议与服务 | uav_ipc | 各模块及 /usrdata/uav | IPC、服务管理、记录目录索引 |
| 地面端 | xiaozhi-ground-station | /usrdata/uav/ground-relay | 状态、地图、控制与视频中继 |
| 遥控器应用 | xiaozhi-ground-station/android | Android APK | 显示地面网页 |

最近选定的定位启动配置是 `tof_odometry/deploy/direct-pose-20261021/run.sh`：直接 ToF 位姿、ESKF 关闭、测高替换开启、预测保持参数 600 ms。该文件是部署快照，使用前仍须与计划发布版本核对。

位姿输出频率和控制指令频率不同：约 20 Hz 控制循环不代表每 50 ms 完成一次 ToF 匹配。纯位姿模式不应要求飞控融合未发送的外部速度。最近实际系统使用位置目标；到目标约 20 cm 后保持。不要用历史速度控制版本说明覆盖当前实现。

**已知待验证问题：**此前有 Guided 切换瞬间转向、固定目标下漂移的问题；安装成功不能据此判断能安全自主飞行。新机必须重新检查航向、坐标方向、位置反馈和定位源切换连续性。

## 2. 电脑准备与目录约定

以下命令在电脑 Bash 中执行；新开终端需重新设置变量。

```bash
export UAV_HOME=/media/daotong/30201387201353683/UAV/xiaozhi
export SDK="$UAV_HOME/ars45-c402-evb-0.17.83-00-20250904_jh20260915"
export SCRIPTS="$SDK/base/customer/apps/ar_lowdelay/scripts"
export AARCH64_TOOLCHAIN=/home/daotong/.local/toolchains/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu
export UAV_IPC_ROOT="$UAV_HOME/uav_ipc"
export BOARD=root@192.168.3.100
```

需要：Git、Python 3、Make、CMake、Eigen 头文件、上述 ARM64 交叉工具链；地面中继构建需要 Go，APK 安装需要 adb。SDK 必须保留厂商依赖库；不是只复制一个可执行文件就能在任意固件运行。

保留各仓库提交号、未提交差异、构建时间、配置及二进制 SHA256。工作区有未提交修改时，提交号不能唯一代表部署内容。

## 3. 连接并确认设备身份

空中板和地面板的 USB 地址都可能是 192.168.3.100，不能仅凭 IP 判断设备。不要同时连接两个相同地址而不指定路由。

```bash
ip -br addr
ip route get 192.168.3.100
ssh -o BatchMode=yes -o ConnectTimeout=5 -o HostKeyAlgorithms=+ssh-rsa "$BOARD"  'uname -a; cat /proc/sys/kernel/random/boot_id; ls /usrdata/uav; df -h /mnt; ps'
```

若不可达，先检查 USB 网卡、网段和路由；不要先修改应用配置。SSH 身份变更应核实板子主机公钥；不要关闭主机校验。管理密钥与地面控制专用密钥分开保存，密码和私钥不写入本文件。

**安装、重启采集或修改飞控参数前，飞机必须落地且未解锁。**连接飞控 USB 后读取 autopilot/component 1 的心跳；相机等其他组件心跳不能当作解锁状态。状态冲突时停止部署，先确认。

## 4. 新板必须具备的文件和标定

先备份原板 `/usrdata/run.sh`、`/usrdata/uav`，并单独导出飞控完整参数。不要把备份放在将要覆盖的模块目录内。

必须准备：

- `capture/config/capture.json`、`sampling-profile.json`；
- `tof/geometry.conf`：三路 ToF 方向、外参及深度几何配置；
- 相机/IMU 标定、偏置配置及当前启动脚本实际引用的文件；
- `explorer/config/board.conf`；
- `vins/run.sh` 与每个模块的 `service.sh`、`module.name`；
- 地面控制密钥与可信主机公钥（新机需要重新核对）。

`geometry.conf` 不由 `live/build.sh` 自动生成，应从已核验的部署备份或该机标定结果取得。不要凭文件名创建空配置。即便安装位置相同，也要验证传感器序号对应前/左/右；曾出现前向 ToF 实际朝右的情况，必须修正安装或外参。

当前深度应按轴向距离解释，不能随意换回射线距离；以准备部署的 geometry 配置和实际代码为准。测高传感器 ID、安装方向、姿态补偿和量程需实测确认。

## 5. 编译

电脑执行：

```bash
bash "$SCRIPTS/build-capture.sh"
sh "$UAV_HOME/tof_odometry/live/build.sh"
bash "$UAV_HOME/nano_explorer-0916/scripts/build-live.sh"
```

产物：

| 模块 | 编译结果 |
|---|---|
| 采集完整包 | SDK/base/customer/apps/ar_lowdelay/build-live/package |
| ToF 二进制 | tof_odometry/live/tof_service |
| Explorer 完整包 | nano_explorer-0916/build-board/package |

采集包应含 `ar_lowdelay`、`lib/libjh_framework.so`、`config`、`run.sh`、`service.sh`、录制工具。ToF 的 `build.sh` **只构建二进制，不构建完整安装包**。

## 6. 制作完整暂存包

电脑执行。以下要求先把经过核验的本机几何配置保存为指定文件；不要直接使用未知旧机配置。

```bash
export RELEASE_ID=$(date +%Y%m%d-%H%M%S)
export STAGE="$HOME/UAV-deploy/$RELEASE_ID"
export GEOMETRY_FILE="$HOME/UAV-deploy/verified/geometry.conf"
test -s "$GEOMETRY_FILE" || { echo '缺少已核验的 geometry.conf'; exit 1; }
mkdir -p "$STAGE/uav/tof" "$STAGE/uav/vins"
cp -a "$SDK/base/customer/apps/ar_lowdelay/build-live/package" "$STAGE/uav/capture"
cp -a "$UAV_HOME/nano_explorer-0916/build-board/package" "$STAGE/uav/explorer"
cp "$UAV_HOME/tof_odometry/live/tof_service" "$STAGE/uav/tof/"
cp "$GEOMETRY_FILE" "$STAGE/uav/tof/geometry.conf"
cp "$UAV_HOME/tof_odometry/deploy/direct-pose-20261021/run.sh" "$STAGE/uav/vins/run.sh"
cp "$UAV_HOME/uav_ipc/scripts/service.sh" "$STAGE/uav/vins/service.sh"
printf 'vins\n' > "$STAGE/uav/vins/module.name"
cp "$SCRIPTS/boot-uav.sh" "$STAGE/uav/boot-all.sh"
cp "$UAV_HOME/uav_ipc/scripts/record-session.sh" "$STAGE/uav/"
cp "$UAV_HOME/uav_ipc/scripts/record-index.sh" "$STAGE/uav/"
find "$STAGE/uav" -name '*.sh' -exec chmod +x {} +
(cd "$STAGE" && find uav -type f -exec sha256sum {} + > SHA256SUMS)
```

如有额外标定文件或启动脚本依赖的库，必须放入暂存包并重新生成清单。用 `file` 检查三个主程序为 ARM64，用工具链 readelf 查看动态依赖，确认板端提供对应库。不要在电脑执行 ARM64 主程序。

上传暂存目录（不会启动程序）：

```bash
ssh "$BOARD" "test ! -e '/usrdata/uav-stage-$RELEASE_ID' && mkdir '/usrdata/uav-stage-$RELEASE_ID'"
tar -C "$STAGE" -cf - . | ssh "$BOARD" "tar -C '/usrdata/uav-stage-$RELEASE_ID' -xf -"
ssh "$BOARD" "cd '/usrdata/uav-stage-$RELEASE_ID' && sha256sum -c SHA256SUMS"
```

旧版 scp 不支持 `-O`，不要固定添加该参数。也可用 `uav_ipc/scripts/deploy_package.sh PACKAGE HOST NEW_ABSOLUTE_DIRECTORY` 上传单个完整模块包；它仅允许新目录，不负责覆盖或启动。

## 7. 激活、开机启动与回退

此阶段在确认未解锁后进行。建议先保存当前日志，再停止 Explorer、vins、capture；`stop` 是发送停止请求，应确认进程已退出，不能紧接着覆盖正在使用的库。

板端命令：

```sh
sh /usrdata/uav/explorer/service.sh stop
sh /usrdata/uav/vins/service.sh stop
sh /usrdata/uav/capture/service.sh stop
ps
```

新板无这些服务时不执行不存在的脚本。检查原厂 `ar_lowdelay` 是否仍由 `/usrdata/run.sh` 启动；不能让原厂和新服务同时占用相机/ToF。

激活步骤：

1. 将现有模块目录复制到 `/usrdata/uav-backups/发布前时间/`，包含 boot-all、record 脚本及 `/usrdata/run.sh`。
2. 将暂存包中的 capture、tof、vins、explorer 替换到 `/usrdata/uav/` 对应目录；保留与本次无关的地面控制文件。使用新目录改名替换，避免混入旧库。
3. 安装 boot-all.sh、record-session.sh、record-index.sh；校验最终文件哈希与暂存包一致。
4. 检查 `/usrdata/run.sh`：保留厂商系统/网络初始化，去除重复启动采集的入口；硬件初始化之后仅保留一次后台启动 `sh /usrdata/uav/boot-all.sh`。不要整份覆盖厂商启动文件，也不要重复追加。
5. 执行 `sh /usrdata/uav/boot-all.sh --check`；检查通过后，在未解锁状态执行 `/sbin/reboot`，重新连接验收。

新采集 SDK 软件重启若不能恢复相机，完全断电再上电；不要反复启动多个采集进程。

boot-all 检查设备节点、平台进程、挂载及文件，连续两次通过后启动 capture 和 vins，之后观察 IMU/三路 ToF 计数。它没有固定 25 秒等待；30/45 秒是等待超时上限。**收到数据不代表定位有效。Explorer 默认仍需手动启动。**

回退：未解锁状态停止服务，恢复同一次备份的二进制、库、配置、启动脚本；若修改过飞控参数也按参数备份回退，随后重启并重新验收。不要只退二进制留下新版协议或配置。

## 8. 飞控参数核对（不要盲目执行旧配置脚本）

连接飞控 USB 并确认 autopilot 心跳，保存完整参数。算力板 UART3 使用 115200，但 **UART3 不等于飞控 SERIAL3**。旧机曾接飞控 SERIAL2，新机需按接线核对；TX/RX 交叉并共地。

| 项目 | 当前纯位姿方案的核对方向 |
|---|---|
| 对应 SERIALn | MAVLink2、115200，保留经核验的选项，不能修改光流/遥控所在串口 |
| VISO_TYPE | MAVLink 外部定位后端；修改后按固件要求重启 |
| EK3_SRC2_POSXY | 6，ExternalNav |
| EK3_SRC2_VELXY | 0，None；纯位姿版本不提供外部速度 |
| EK3_SRC2_POSZ | 最近旧机为 2，RangeFinder；读取本机确认 |
| EK3_SRC2_VELZ | 最近旧机为 0，None |
| EK3_SRC2_YAW | 最近旧机为 6，ExternalNav；切换航向异常未完全闭环，必须验证 |
| Source 1 | 保留本机已经验证的手控定位配置，不直接覆盖 |
| WPNAV_SPEED | 旧机测试 10 cm/s，即 0.1 m/s；不是物理速度绝对上限 |
| WPNAV_ACCEL | 旧机最后测试 25 cm/s²，即 0.25 m/s²；不是新机通用推荐值 |

`configure-guided-external-nav.py` 当前仍把 `EK3_SRC2_VELXY` 写为 6，且首个心跳选择需要留意组件。**不要对纯位姿版本直接执行该脚本的 --apply。**先核对后按当前通信方式设置，保存、回读并记录差异。

不要关闭解锁检查来消除 `VisOdom not healthy` 等提示；应检查位姿实际接收、时间戳、后端与源配置。Source 2 的选择与 Guided 是不同概念，要确认桥接确实发送并完成源切换。

## 9. 地面板与遥控器

详见 `xiaozhi-ground-station/docs/DEPLOYMENT.md`。空中/地面消息版本应配套；增量地图协议升级先部署兼容中继，再部署空中 Explorer。

电脑构建：

```bash
sh "$UAV_HOME/xiaozhi-ground-station/relay/build.sh"
```

向地面 `/usrdata/uav/ground-relay/` 安装：`relay/build/uav-ground-relay`、`index.html`、`plan-map.js`、`plan-map.css`、`relay/service.sh`、`relay/video-service.sh`、`relay/go2rtc.yaml`；视频依赖按该部署文档准备。保留专用 `air_key` 和核验过的 `air_host.pub`。

空中安装 `relay/air-control.sh` 为 `/usrdata/uav/ground-control.sh`，专用公钥在 authorized_keys 使用受限命令前缀；保留原管理密钥。新飞机 SSH 主机公钥变化时同步地面可信公钥，不能用关闭校验来解决。

地面执行：

```sh
sh /usrdata/uav/ground-relay/service.sh stop
sh /usrdata/uav/ground-relay/service.sh start
/usrdata/uav/ground-relay/uav-ground-relay -check-control
tail -40 /tmp/uav-ground-relay.log
```

网页默认 `http://192.168.1.100:8088/`。仅网页变化通常不需换 APK，重新加载页面即可；修改 Android 容器时才安装匹配签名的 APK：

```bash
adb install -r "$UAV_HOME/xiaozhi-ground-station/dist/UAV-GroundStation-1.0.apk"
adb shell am start -n com.uav.groundstation/.MainActivity
```

退出 App 不等于停止探索。测试结束应明确停止测试/探索并确认飞控模式。

## 10. 部署验收顺序

### 10.1 静态只读检查

板端：

```sh
cat /tmp/uav-boot-ready.log
sh /usrdata/uav/capture/service.sh status
sh /usrdata/uav/vins/service.sh status
cat /tmp/uav-ipc/tof.status.json
cat /tmp/uav-ipc/flow-height.status.json
df -h /mnt
```

电脑采集测试：

```bash
python3 "$SCRIPTS/check_tof_capture.py" --host "$BOARD" --seconds 10 --output "$HOME/UAV-tof-check-$(date +%Y%m%d-%H%M%S)"
```

查看三路计数增量、有效点、深度和时间戳，而不是只看进程存在。拿平面物体分别放在三路前方已知距离，验证传感器方向、深度、遮挡。IMU 连续，静止时姿态正常；测高与尺量高度一致，倾斜后补偿方向正确。

### 10.2 未解锁手持检查

手持前后左右移动，比较 ToF、App、飞控本地位置变化；转动机头比较航向和坐标变换。检查 Guided/Source 2 切换时位置、航向、测高是否跳变。验证不发送外部速度时飞控源配置相符。未解锁不发送飞行动作，不代表桥接故障。

### 10.3 控制测试分层

先验证固定点保持，再验证小距离单目标，最后探索。分别记录最终目标、下发目标、飞控目标反馈、实际位置估计、航向、位姿年龄与拒绝原因。发包成功不等于飞控接收，也不等于执行成功。地图可用不代表定位持续有效。

最近存在切换转向问题，因此完成上述连续性验证之前不要将安装结果写成“飞行正常”。

## 11. 保存、下载与离线回放

默认记录模式 `sensors`：保存 IMU、ToF、640×480/15 Hz 灰度相机输入及配置快照；不等于保存全部算法内部状态。实际相机帧率还需实测。`full` 和 `off` 由 record-session.sh 支持，但需要在服务启动环境中配置，单独在临时终端 export 不会影响已有服务。

板端目录：

| 路径 | 内容 |
|---|---|
| /mnt/uav-recordings/<boot UUID>/ | 本次启动采集数据 |
| /mnt/uav-recordings/sessions.jsonl | 持久会话索引 |
| /mnt/uav-recordings/current.txt | 当前记录目录 |
| /mnt/uav-logs/ | 服务及桥接日志 |
| /tmp/uav-ipc/ | 实时状态、IPC、部分诊断，重启后可能丢失 |

电脑命令：

```bash
python3 "$SCRIPTS/download_replay.py" --list
python3 "$SCRIPTS/download_replay.py" --session previous --port 8770
# 指定 UUID：把示例中的 UUID 替换为 --list 输出的完整值
python3 "$SCRIPTS/download_replay.py" --session UUID --port 8771
```

默认下载至 `~/UAV-replays/时间-UUID前8位/`，生成 `start_replay.sh`；浏览器访问脚本显示的本地地址。脚本拒绝下载正在写入的当前会话。可用 `--local 已下载的UUID目录` 重新生成；`--no-serve` 只生成不启动页面。

当前下载脚本 `--height` 支持 flat/tof，默认 flat 用于平面对比，不要把它理解成已还原板载固定测高融合。`--clearance 0.25` 可改变离线净空半径，不会修改板端配置。

同时下载的 `current-board-logs` 是下载当时的日志，可能属于重启后的新会话，不能自动归到所选历史记录。离线重算与板载实时状态要区分：离线成功不证明板载当时没有超时或失锁。

## 12. 清理存储

先下载并核对，再清理。电脑执行：

```bash
bash "$SCRIPTS/clear_capture_data.sh" --preview "$BOARD"
bash "$SCRIPTS/clear_capture_data.sh" --apply "$BOARD"
```

该脚本只删除未被使用的历史 UUID 目录和轮转历史日志；保留当前会话、打开的文件、实时日志、索引、配置与程序。因此“执行成功”不代表磁盘所有数据已清空。不要用 `rm -rf /mnt/*`。

## 13. 常见问题

| 现象 | 优先检查 |
|---|---|
| SSH 超时 | USB 网卡、路由、供电、是否连错地面板 |
| 启动建图无反应 | 地面受限控制密钥、主机公钥、控制回执、Explorer 服务及有效位姿 |
| 三路计数不动 | 采集进程退出、SDK 初始化、硬件；旧记录不能当实时数据 |
| sensor_unavailable | 具体原因、各路时间戳、处理积压；不直接认定传感器坏 |
| no_clearance | 障碍点来源、外参、坐标、测高、占用更新；不要随意删除安全检查 |
| VisOdom not healthy | 外部定位后端、实际消息、持续性、源配置 |
| 切 Guided 突然转头 | 外部与飞控航向、源切换、目标 yaw/yaw_rate；暂停移动测试 |
| 上传成功但行为未变 | 实际运行路径、进程重启、最终文件哈希、配置与网页缓存 |
| 回放打不开 | 本地服务是否仍运行、端口、start_replay.sh；页面不是永久在线网站 |

## 14. 每次部署记录模板

- 设备身份、boot UUID、接线和固件：
- 日期、操作者、部署前未解锁状态：
- 各仓库提交号与未提交修改快照：
- 发布包路径与 SHA256 清单：
- 三路外参/IMU/测高配置来源：
- 飞控参数备份与修改后回读：
- 原版本备份目录与回退方法：
- 静态采集、手持坐标、源切换验收结果：
- 地面中继与 App 版本：
- 数据会话 UUID、下载目录、未解决问题：

完整部署应包含程序、库、配置、标定、服务、开机入口、地面控制链路和验证记录，而非仅替换某个二进制。
