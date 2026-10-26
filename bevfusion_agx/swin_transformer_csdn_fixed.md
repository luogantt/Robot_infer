# Swin Transformer 完整架构图（CSDN 兼容版）

Swin Transformer 是一种分层式 Vision Transformer（视觉 Transformer）。核心思想是：**在局部窗口内计算 Self-Attention，再通过 Shifted Window（移位窗口）让相邻窗口之间交换信息，同时逐层降低特征图分辨率、增加通道数。**

在 BEVFusion 中，Swin Transformer 是 Camera 分支的图像特征提取 Backbone，后面通常接 FPN 和 LSS。

> **CSDN 兼容说明：** 本文架构图使用 Markdown 代码块中的文本流程图，不依赖 Mermaid 插件；公式使用 `$$...$$` 和 `$...$` 格式，避免复制到 CSDN 后图形或公式无法正常渲染。

## 一、Swin Transformer 整体网络架构

```text
输入 RGB 图像
[B, H, W, 3]
       |
       v
Patch Partition：切分为 4×4 图像块
       |
       v
Linear Embedding：每个 Patch 映射到 96 维
       |
       v
+------------------------------------------------+
| Stage 1                                        |
| Swin Block × 2                                 |
| W-MSA → SW-MSA                                 |
| 分辨率：H/4 × W/4；通道数：96                  |
+------------------------------------------------+
       |
       v
Patch Merging：空间分辨率减半，通道数加倍
       |
       v
+------------------------------------------------+
| Stage 2                                        |
| Swin Block × 2                                 |
| W-MSA → SW-MSA                                 |
| 分辨率：H/8 × W/8；通道数：192                 |
+------------------------------------------------+
       |
       v
Patch Merging：空间分辨率减半，通道数加倍
       |
       v
+------------------------------------------------+
| Stage 3                                        |
| Swin Block × 6                                 |
| W-MSA / SW-MSA 交替                            |
| 分辨率：H/16 × W/16；通道数：384               |
+------------------------------------------------+
       |
       v
Patch Merging：空间分辨率减半，通道数加倍
       |
       v
+------------------------------------------------+
| Stage 4                                        |
| Swin Block × 2                                 |
| W-MSA → SW-MSA                                 |
| 分辨率：H/32 × W/32；通道数：768               |
+------------------------------------------------+

多尺度特征输出：
    Stage 1 ──> F1：H/4  × W/4  × 96
    Stage 2 ──> F2：H/8  × W/8  × 192
    Stage 3 ──> F3：H/16 × W/16 × 384
    Stage 4 ──> F4：H/32 × W/32 × 768
                         |
                         v
                FPN 多尺度特征融合
                         |
                         v
             BEVFusion Camera 分支的 LSS
                         |
                         v
                   Camera BEV
```

以上是典型 Swin-T（Tiny）结构示意：Embedding Dimension 为 96，四个 Stage 的 Block 数为 `[2, 2, 6, 2]`，通道数依次为 96、192、384、768。实际模型的输入分辨率、输出层、归一化细节和 FPN 配置，应以所使用的代码配置为准。

## 二、一个 Swin Transformer Block 内部是什么？

每个 Swin Block 主要由 Attention、MLP、LayerNorm 和残差连接组成。

```text
输入特征 X
    |
    v
LayerNorm
    |
    v
Window MSA 或 Shifted Window MSA
    |
    v
残差相加：X + Attention 输出
    |
    v
LayerNorm
    |
    v
MLP：Linear → GELU → Linear
    |
    v
残差相加
    |
    v
输出特征 X'
```

数学形式如下。这里的 Attention 可以是 W-MSA，也可以是 SW-MSA，取决于当前 Block：

$$
\hat{X}=X+\operatorname{Attention}(\operatorname{LN}(X))
$$

$$
X'=\hat{X}+\operatorname{MLP}(\operatorname{LN}(\hat{X}))
$$

其中，`Attention` 在普通窗口 Block 中表示 W-MSA，在移位窗口 Block 中表示 SW-MSA。

以上公式省略了 Dropout、DropPath 等训练细节。

## 三、W-MSA 与 SW-MSA 的区别

### 1. W-MSA：窗口注意力

把特征图切成互不重叠的窗口，只在各自窗口内部计算 Self-Attention。

```text
特征图
+------+------+------+
| 窗口A | 窗口B | 窗口C |
+------+------+------+
| 窗口D | 窗口E | 窗口F |
+------+------+------+

每个窗口内部独立计算 Attention。
不同窗口之间在这一层不直接交换信息。
```

### 2. SW-MSA：移位窗口注意力

下一类 Block 将窗口划分整体平移，使上一层分属不同窗口的 Token 有机会进入同一个窗口，从而实现跨窗口信息交互。

```text
上一层窗口边界：
+------+------+------+
|   A  |   B  |   C  |
+------+------+------+
|   D  |   E  |   F  |
+------+------+------+

移位后重新划分窗口：
  +------+------+------+
  | A / B| B / C|      |
  +------+------+------+
  | D / E| E / F|      |
  +------+------+------+

示意图只用于表达“窗口划分发生位移”，不代表具体实现中的精确像素分区。
```

在典型 Swin Transformer 中，连续 Block 通常交替使用 W-MSA 和 SW-MSA，而不是在同一个 Block 内同时执行二者。

实现时通常还会使用循环移位、Attention Mask 和逆移位，避免不应该相互注意的区域发生错误交互。

### 3. 为什么窗口注意力计算更省？

设特征图大小为 $H\\times W$，通道数为 $C$，每个窗口边长为 $M$。

全局 Self-Attention 的主要计算量近似为：

$$
O((HW)^2C)
$$

窗口 Self-Attention 的主要计算量近似为：

$$
O(HW M^2 C)
$$

因为每个 Token 只与所在窗口中的有限数量 Token 计算 Attention，所以当特征图很大时，窗口注意力可以显著降低计算量。

## 四、Patch Merging：为什么分辨率越来越低？

Swin Transformer 是分层式网络，不同于始终保持相同 Token 分辨率的普通 ViT。

Patch Merging 会将相邻的 $2\\times2$ 个 Token 特征组合起来：
1. 将四个位置的特征拼接，通道从 $C$ 变成 $4C$；
2. 经过归一化与线性映射，将通道压到下一阶段的 $2C$；
3. 特征图的高、宽各缩小一半。

典型 Swin-T 的四个 Stage 如下：

| 阶段 | 特征图分辨率 | 通道数 | Block 数 | 注意力头数 |
|---|---|---:|---:|---:|
| Stage 1 | $H/4 \\times W/4$ | 96 | 2 | 3 |
| Stage 2 | $H/8 \\times W/8$ | 192 | 2 | 6 |
| Stage 3 | $H/16 \\times W/16$ | 384 | 6 | 12 |
| Stage 4 | $H/32 \\times W/32$ | 768 | 2 | 24 |

因此，网络越往后：
- 特征图的空间分辨率越低；
- 通道数越多；
- 感受野和高级语义表达能力逐渐增强。

与此同时，四个 Stage 的输出可以交给 FPN 做多尺度特征融合。

## 五、Swin Transformer 在 BEVFusion 中的位置

```text
多路 Camera RGB 图像
          |
          v
+---------------------------+
| Swin Transformer Backbone |
| 提取多尺度图像特征         |
+---------------------------+
          |
          v
+---------------------------+
| FPN                       |
| 融合不同分辨率的特征       |
+---------------------------+
          |
          v
+---------------------------+
| LSS Depth / Context       |
| 预测深度分布和上下文特征   |
+---------------------------+
          |
          v
+---------------------------+
| Lift + Splat / BEV Pooling|
| 利用标定把特征映射到 BEV   |
+---------------------------+
          |
          v
Camera BEV Feature
          |
          v
与 LiDAR BEV Feature 融合
          |
          v
BEVFusion 下游检测 / 分割任务
```

Swin Transformer 本身**不负责输出三维车辆位置，也不直接生成 BEV**，它输出的是二维图像特征。

后续模块的作用是：
- **Swin Transformer：** 从图像中提取局部纹理和高级语义；
- **FPN：** 融合不同分辨率的图像特征；
- **LSS：** 预测深度分布，并结合相机标定进行视角转换；
- **BEVFusion：** 将 Camera BEV 与 LiDAR BEV 融合，供下游检测头使用。

因此，在 BEVFusion 中替换 Swin Transformer 时，需要同时检查 FPN 的输入通道、输出尺度和 LSS 的特征接口。

## 六、总结

Swin Transformer 的三个核心设计是：

1. **窗口注意力（W-MSA）：** 将 Attention 限制在局部窗口，控制计算量；
2. **移位窗口（SW-MSA）：** 让不同窗口之间逐层交换信息；
3. **Patch Merging：** 构建分层特征金字塔，兼顾局部细节和高级语义。

放在 BEVFusion 中看，Swin Transformer 负责“从图像提取什么特征”；FPN 负责“如何融合不同尺度”；LSS 负责“如何把二维图像特征映射到 BEV”。三者职责不同，不能把 Swin Transformer 本身理解为三维感知或 BEV 变换模块。
