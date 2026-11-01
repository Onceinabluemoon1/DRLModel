# 通道 RGB-D 回放停车

固定四段合成 RGB-D 上回放停车。输入是录好的序列，不是整车，也不是真车传感器。

人走进前方 2.5 m 停止区时，两个感知帧内给出 `STOP_PERSON`，速度为 0。车道里的周转箱或托盘用测到的米制距离，1.0 m 内给出 `STOP_DISTANCE`。距离来自深度反投影，不用检测框的像素高度。出速度的进程关掉之后，人的回放仍然是 `STOP_PERSON`。安全门不读速度进程的输出。

这一轮没有做：k3s、EKS、diffusion policy、共形区间、Astra、世界模型、机械臂、货架侧向间隙、TensorRT。

## 数据从哪来

画面、深度和标注都由 `aisle_replay/synth.py` 生成。

- 相机是针孔模型，光轴相对水平面向下俯仰 27°。`base_link` 的 x 向前、y 向左、z 向上。相机在 `(0.20, 0.00, 0.55)` m。
- 深度是沿相机光轴的 z，写成 uint16 毫米，再反投影回 `base_link`。
- 人、周转箱、托盘是纯色长方体。干扰物不上标签。
- 这不是仓库实采，不是 mcap，不能写成实车结果。

YOLO11-seg 用的是 Ultralytics `yolo11n-seg` 预训练权重（[v8.3.0 资源](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt)）在 64 张合成训练图、16 张合成验证图上微调 40 个 epoch。类别只有三个，且彼此独立：`person`、`tote`、`pallet`。`PERSON` 不会并进 `OTHER`。微调后的权重在 `aisle_replay/weights/yolo11n-seg-synth.pt`。

## 四段回放

| 片段 | 期望 |
| --- | --- |
| `empty` | 每帧 `OK`，速度是规则里的 0.6 m/s |
| `person` | 人在 2.5 m 外是 `OK`；前表面到 2.2 m 的那一帧起 `STOP_PERSON` |
| `tote` | 车道内、前表面 0.75 m 时 `STOP_DISTANCE`；车道外的箱子是 `OK` |
| `pallet` | 车道内、前表面 0.80 m 时 `STOP_DISTANCE`；车道外的托盘是 `OK` |

`tote_1m` 不是行为片段。周转箱前表面固定在 1.00 m，用来量距离误差。

阈值在 `aisle_replay/safety/rules.yaml`：人 2.5 m，周转箱和托盘 1.0 m，车道半宽 0.90 m。

## 怎么跑

在仓库根目录：

```bash
python3 -m unittest discover -s aisle_replay/tests -t .
python3 -m aisle_replay prove-kill
python3 -m aisle_replay replay --clip person --detector yolo \
  --weights aisle_replay/weights/yolo11n-seg-synth.pt
python3 -m aisle_replay replay --clip tote --detector yolo \
  --weights aisle_replay/weights/yolo11n-seg-synth.pt
python3 -m aisle_replay replay --clip pallet --detector yolo \
  --weights aisle_replay/weights/yolo11n-seg-synth.pt
python3 -m aisle_replay replay --clip empty --detector yolo \
  --weights aisle_replay/weights/yolo11n-seg-synth.pt
python3 -m aisle_replay measure
```

`prove-kill` 和单元测试用的是真值掩码（`--detector fixture`），不加载权重，用来钉死安全门。三个数走 `--detector yolo`。两条路径共用同一个安全门和同一套反投影。

重新微调（会覆盖上面的权重，指标要重测）：

```bash
python3 -m aisle_replay train --epochs 40
python3 -m aisle_replay measure
```

依赖见 `aisle_replay/requirements.txt`。安全门测试只需要 numpy 和 PyYAML。YOLO 回放需要 CPU 版 PyTorch 和 `ultralytics==8.3.40`。

速度占位进程可以单独开着看。它发送 `vx=0.8`。安全门不连接它。空通道时安全门给出的是规则里的 `0.6`，不是 `0.8`。

```bash
python3 -m aisle_replay velocity --port 18765 --vx 0.8
```

## 安全门和速度进程

`SafetyGate.evaluate` 只接收观测里的类别和米制距离。签名里没有速度。`replay.py` 不连接速度进程。

`prove-kill` 的顺序是：

1. 拉起速度进程，读到它发出的 `vx=0.8`。
2. 速度进程还活着时回放空通道。输出必须是 `OK` 且 `vx=0.6`。
3. 人的回放进程先停在原地，不读画面。
4. `SIGKILL` 杀掉速度进程，确认它已经死了。
5. 放行人的回放。进入停止区的帧仍然是 `STOP_PERSON`，速度为 0，并且落在两个感知帧内。

## 三个数

本次测量写在 `aisle_replay/results/metrics.json`。权重 sha256：`3e7b5d74cc93e7d0af5725a86cc87b618842abb632a2bfebd462198dfb684fc4`。置信度 0.25。图像 320×240。设备是这台机器的 CPU，不是车上处理器。

| 数 | 本次合成回放 | 怎么测的 |
| --- | --- | --- |
| 人的召回 | **100%**（24/24） | 种子 2026 的留出帧，和训练种子 7、8 不同，不写入训练集。正例是画面里 `PERSON` 掩码至少 40 像素。检测出一个 `PERSON`，和真值掩码的 IoU ≥ 0.5，算命中。召回 = 命中帧 / 正例帧。空通道帧不进分母。 |
| 1 m 处箱子的距离误差 | **0.000079 cm** | `tote_1m` 共 5 帧，前表面放在 `base_link` 前方 1.00 m。YOLO 掩码内的深度反投影后，取近表面前向 x 的中位数。误差是 `|估计 − 1.00| × 100` 的平均绝对误差，5 帧都检出。真值掩码走同一公式是 0.0016 cm。 |
| 检测前向 | **18.9 ms** | 预热 3 次后，对同一张留出帧再跑 15 次。取 Ultralytics 单张 `inference` 计时的中位数，不含预处理和 NMS。同一次调用的整段 `predict` 墙钟中位数是 20.0 ms。 |

人的回放：几何上第 2 帧（从 0 计）进入 2.5 m 停止区，YOLO 回放也在第 2 帧给出 `STOP_PERSON`，延迟 0，在两个感知帧内。四段回放的原因码和几何期望一致。

距离误差远小于 1 mm。深度是渲染器按毫米四舍五入写出来的，检测掩码又贴在这块前表面上，所以这个数只说明反投影在合成数据上是闭合的。它不是实传感器、不是标定残差，也不能写成车上测到的厘米级误差。

人的召回是同一套纯色长方体的留出帧，不是仓库里的人。100% 不能写成通道留出集上的召回。

## 还缺什么

- 没有仓库实采 RGB-D，也没有 mcap。
- 没有真人、真实周转箱、真实托盘的标注。方案里的约 800 张不在这里。
- 没有在真实帧上微调的权重。现有权重只认识这套合成形状。
- 没有整车，也没有车上计时。18.9 ms 只是这次 CPU 回放。
- 侧面的人这一轮不降速。货架侧向 0.30 m 没做。

有了真实对齐 RGB-D 和对应标注之后，换 `--weights`，再跑 `measure`，用新的三个数替换上面的表。在那之前不要把这些数写成实车。
