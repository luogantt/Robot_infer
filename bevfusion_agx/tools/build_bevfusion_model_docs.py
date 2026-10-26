#!/usr/bin/env python3
"""Build the self-contained BEVFusion architecture/source-map HTML.

The visual shell is intentionally inherited from the BEVFormer architecture page
supplied with this project task.  All model data and source excerpts below come
from this repository's deployment implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = ROOT / "tools" / "templates" / "bevformer-model-architecture.html"
DEFAULT_OUTPUT = ROOT / "docs" / "bevfusion-model-architecture.html"
GITHUB_ROOT = "https://github.com/luogantt/Robot_infer/blob/master/bevfusion_agx"


def source(
    source_id: str,
    relpath: str,
    start: int,
    end: int,
    owner: str,
    symbol: str,
    note: str = "",
) -> tuple[str, dict]:
    path = ROOT / relpath
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if not (1 <= start <= end <= len(lines)):
        raise ValueError(f"Invalid source range {relpath}:{start}-{end} (file has {len(lines)} lines)")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    encoded = quote(relpath.replace("\\", "/"), safe="/")
    item = {
        "file": relpath.replace("\\", "/"),
        "owner": owner,
        "symbol": symbol,
        "start": start,
        "end": end,
        "sha256": digest,
        "code": "\n".join(lines[start - 1 : end]),
        "localUrl": "../" + relpath.replace("\\", "/"),
        "localPath": relpath.replace("\\", "/"),
        "url": f"{GITHUB_ROOT}/{encoded}#L{start}-L{end}",
        "note": note,
    }
    return source_id, item


def node(
    title: str,
    subtitle: str,
    kind: str,
    summary: str,
    input_: str,
    action: str,
    output: str,
    sources: list[str],
    *,
    notes: list[str] | None = None,
    related: list[str] | None = None,
    view: str | None = None,
    formula: str | None = None,
) -> dict:
    return {
        "title": title,
        "subtitle": subtitle,
        "kind": kind,
        "summary": summary,
        "input": input_,
        "action": action,
        "output": output,
        "sources": sources,
        "notes": notes or [],
        "related": related or [],
        "view": view,
        "formula": formula,
    }


def build_catalog() -> dict:
    sources = dict(
        [
            source("core-init", "CUDA-BEVFusion/src/bevfusion/bevfusion.cpp", 40, 100, "BEVFusion Core", "CoreImplement::init", "创建相机、点云、融合和检测模块。"),
            source("core-forward", "CUDA-BEVFusion/src/bevfusion/bevfusion.cpp", 104, 133, "BEVFusion Core", "CoreImplement::forward_only", "部署时真实执行顺序。"),
            source("core-api", "CUDA-BEVFusion/src/bevfusion/bevfusion.hpp", 39, 72, "BEVFusion API", "CoreParameter / Core", "公开输入、标定更新和推理接口。"),
            source("runtime-config", "CUDA-BEVFusion/src/main.cpp", 167, 238, "Runtime configuration", "create_core", "输入尺寸、体素范围、BEV 网格、模型文件和解码阈值。"),
            source("normalize", "CUDA-BEVFusion/src/bevfusion/camera-normalization.cu", 93, 165, "CUDA preprocessing", "load_pixel / normalize_value / normalize_to_planar_kernel", "双线性采样、ImageNet mean/std 与 NCHW 平面写入。"),
            source("depth-project", "CUDA-BEVFusion/src/bevfusion/camera-depth.cu", 43, 109, "CUDA preprocessing", "compute_depth_kernel / DepthImplement::forward", "把 LiDAR 点投影为六路稀疏深度图。"),
            source("camera-model", "CUDA-BEVFusion/qat/export-camera.py", 50, 80, "Camera export", "SubclassCameraModule::forward", "ResNet/neck 与 depthnet 的导出边界。"),
            source("camera-export", "CUDA-BEVFusion/qat/export-camera.py", 95, 135, "Camera export", "torch.onnx.export", "相机 backbone 与 vtransform 的 ONNX binding。"),
            source("camera-runtime", "CUDA-BEVFusion/src/bevfusion/camera-backbone.cu", 36, 80, "TensorRT runtime", "BackboneImplement", "camera.backbone.plan 的输入输出绑定。"),
            source("geometry-kernel", "CUDA-BEVFusion/src/bevfusion/camera-geometry.cu", 111, 156, "Camera geometry", "create_frustum_kernel / compute_geometry_kernel", "把相机 frustum 样本映射到 LiDAR/BEV 坐标。"),
            source("geometry-setup", "CUDA-BEVFusion/src/bevfusion/camera-geometry.cu", 178, 213, "Camera geometry", "GeometryImplement::init", "创建 118 个深度 bin 的 frustum 与索引缓冲区。"),
            source("geometry-update", "CUDA-BEVFusion/src/bevfusion/camera-geometry.cu", 218, 265, "Camera geometry", "GeometryImplement::update", "标定更新后排序 geometry ranks 并构建 BEVPool intervals。"),
            source("bevpool", "CUDA-BEVFusion/src/bevfusion/camera-bevpool.cu", 42, 121, "CUDA BEV pooling", "bevpool_half_pack10_kernel / BEVPoolImplement", "按 geometry interval 聚合 feature × depth weight。"),
            source("vtransform", "CUDA-BEVFusion/src/bevfusion/camera-vtransform.cu", 36, 65, "TensorRT runtime", "VTransformImplement", "camera.vtransform.plan 的 feat_in/feat_out 绑定。"),
            source("voxelize-kernels", "CUDA-BEVFusion/src/bevfusion/lidar-voxelization.cu", 77, 145, "CUDA voxelization", "build_hash_table_kernel / voxelization_kernel", "散列活跃体素并写入体素点。"),
            source("voxelize", "CUDA-BEVFusion/src/bevfusion/lidar-voxelization.cu", 209, 269, "CUDA voxelization", "VoxelizationImplement::forward", "生成 FP16 voxel features 与 [V,4] 坐标。"),
            source("lidar-scn", "CUDA-BEVFusion/src/bevfusion/lidar-scn.cpp", 33, 67, "LiDAR wrapper", "SCNImplement", "体素化后把 sparse tensor 交给源码 SCN 引擎。"),
            source("onnx-parser", "CUDA-BEVFusion/src/bevfusion/lidar-scn-onnx-parser.cpp", 107, 198, "Open sparse parser", "load_engine_from_onnx", "解析 SparseConvolution 等节点并构建执行图；开放后端当前仅支持 FP16。"),
            source("sparse-build", "libraries/New3DSparseConvolution/CMakeLists.txt", 28, 58, "Open sparse build", "open_spconv / newscn targets", "生成 libscn.so 并链接源码构建的 libspconv.so。"),
            source("sparse-doc", "docs/BEVFUSION_OPEN_SPARSE_BACKEND.md", 34, 66, "Architecture note", "libscn.so / libspconv.so roles", "仓库内两个动态库的职责与依赖关系。"),
            source("sparse-conv", "libraries/New3DSparseConvolution/libspconv/src/sparseConvImplicit.cu", 356, 457, "libscn operator wrapper", "SparseConvImplicit::forward", "libscn.so 中的算子封装：准备 sparse tensors，并调用 libspconv.so 的 indice-pair 与 Mask Implicit GEMM API。"),
            source("sparse-output", "libraries/New3DSparseConvolution/libspconv/src/sparseConvImplicit.cu", 459, 474, "libscn operator wrapper", "SparseConvImplicit output", "SubM 复用输入坐标；普通卷积输出新坐标与网格。"),
            source("scatter-dense", "libraries/New3DSparseConvolution/libspconv/src/sparseToDense.cu", 146, 198, "libscn CUDA operator", "SparseToDense::forward", "libscn.so 中的末端算子：清零并 scatter 成稠密 BEV。"),
            source("fusion-runtime", "CUDA-BEVFusion/src/bevfusion/transfusion.cu", 37, 79, "TensorRT runtime", "TransfusionImplement", "fuser.plan 的 camera/lidar/middle 绑定。"),
            source("fusion-export", "CUDA-BEVFusion/qat/export-transfuser.py", 225, 240, "Fusion export", "SubclassFuser::forward", "ConvFuser 后继续执行 decoder backbone 与 neck。"),
            source("proposal-head", "CUDA-BEVFusion/qat/export-transfuser.py", 49, 182, "TransFusion export", "SubclassHeadBBox::head_forward", "热力图局部极大值、Top-K query 与 Transformer decoder。"),
            source("raw-head", "CUDA-BEVFusion/qat/export-transfuser.py", 184, 220, "TransFusion export", "SubclassHeadBBox::get_bboxes / forward", "组合 query heatmap score 并导出六组原始预测。"),
            source("fusion-bindings", "CUDA-BEVFusion/qat/export-transfuser.py", 285, 310, "TensorRT export", "fuser/head ONNX export", "固定 camera、LiDAR、middle 和检测头 binding。"),
            source("decode", "CUDA-BEVFusion/src/bevfusion/head-transbbox.cu", 40, 98, "CUDA postprocess", "decode_kernel", "恢复中心、尺寸、航向、速度、置信度与类别。"),
            source("head-runtime", "CUDA-BEVFusion/src/bevfusion/head-transbbox.cu", 100, 191, "TensorRT + CUDA postprocess", "TransBBoxImplement", "运行 head.bbox.plan、CUDA 过滤、回传并按分数排序。"),
            source("python-output", "CUDA-BEVFusion/src/python.cpp", 173, 195, "Python binding", "Core::forward_with_normalization", "把 C++ BoundingBox 打包为 [K,11] float32 数组。"),
            source("manifest-camera", "artifacts/vendor-source/results/engine-build-manifest.json", 35, 90, "Verified engine manifest", "camera.backbone bindings", "本仓库构建产物记录的 FP16 binding 形状。"),
            source("manifest-fusion", "artifacts/vendor-source/results/engine-build-manifest.json", 98, 177, "Verified engine manifest", "fuser / camera.vtransform bindings", "本仓库构建产物记录的 FP16 binding 形状。"),
            source("manifest-head", "artifacts/vendor-source/results/engine-build-manifest.json", 181, 261, "Verified engine manifest", "head.bbox bindings", "本仓库构建产物记录的 200-query 输出形状。"),
            source("convfuser-config", "CUDA-BEVFusion/configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50/convfuser.yaml", 1, 5, "Model config", "ConvFuser", "相机 80 通道与 LiDAR 256 通道融合为 256 通道。"),
        ]
    )

    nodes = {
        "input": node(
            "同步传感器帧", "6×Camera + LiDAR + calibration", "input",
            "一次推理接收同一时刻的六路环视图像、LiDAR 点云和相机标定。主流程在同一 CUDA stream 上依次调度。",
            "6 路 RGB 原图；点云 [N,5]；camera2lidar、intrinsics、lidar2image、img_aug_matrix。",
            "Core 保存/更新标定，把图像和点云送入相机与 LiDAR 两条分支。",
            "相机图像、点云和几何查找表进入各自模块。",
            ["core-api", "core-forward"], related=["images", "points", "calibration", "core"]
        ),
        "images": node(
            "六路相机图像", "[6,900,1600,3] · uint8", "input",
            "nuScenes 六个方向的同步 RGB 图像。部署端先在 CUDA 中完成缩放、裁剪、归一化和 HWC→NCHW。",
            "每个相机一张 1600×900 RGB 图。",
            "按 resize_lim=0.48 做双线性采样，并裁成 704×256。",
            "img: [1,6,3,256,704] FP16。",
            ["runtime-config", "normalize"], related=["normalize", "camera-backbone"]
        ),
        "points": node(
            "LiDAR 点云", "[N,5] · FP16 · N≤300000", "input",
            "点云的五列为 x、y、z、intensity、time lag。它既进入稀疏 3D 主干，也投影到六路图像生成稀疏深度。",
            "points: [N,5]，超出 300000 点时裁剪。",
            "复制到 GPU 后分给 voxelization 与 depth projection。",
            "稀疏体素输入；depth: [1,6,1,256,704]。",
            ["core-forward", "runtime-config"], related=["depth", "voxelize"]
        ),
        "calibration": node(
            "相机标定与增强矩阵", "camera↔LiDAR / intrinsics / img_aug", "state",
            "同一组标定服务两处：把 LiDAR 点投影到图像深度图，以及把相机 frustum 映射到 BEV 网格。",
            "每个相机的 4×4 camera2lidar、intrinsics、lidar2image、img_aug_matrix。",
            "Depth 模块复制 lidar2image/img_aug；Geometry 求逆、生成 ranks、排序并构建 intervals。",
            "深度投影矩阵；BEVPool indices 与 intervals。",
            ["core-api", "depth-project", "geometry-update"], related=["depth", "geometry"]
        ),
        "normalize": node(
            "CUDA 图像预处理", "resize + crop + mean/std + HWC→NCHW", "compute",
            "把六张 uint8 原图直接转换成相机 TensorRT 引擎需要的 FP16 平面张量。",
            "[6,900,1600,3] uint8。",
            "双线性插值到 704×256；乘 1/255 后按 ImageNet mean/std 归一化；按 NCHW 写出。",
            "img: [1,6,3,256,704] FP16。",
            ["runtime-config", "normalize"], related=["images", "camera-backbone"]
        ),
        "depth": node(
            "LiDAR→Camera 稀疏深度", "CUDA projection", "compute",
            "将每个 LiDAR 点投到六路增强后的图像平面；命中的像素写入距离，其他位置保持 0。",
            "points [N,5]；lidar2image 与 img_aug_matrix。",
            "计算齐次投影和深度，做图像边界判断并写入对应相机平面。",
            "depth: [1,6,1,256,704] FP16。",
            ["depth-project", "core-forward", "manifest-camera"], related=["points", "calibration", "camera-backbone"]
        ),
        "camera-backbone": node(
            "Camera Backbone + DepthNet", "camera.backbone.plan · TensorRT FP16", "compute",
            "ResNet-50/neck 提取图像特征；dtransform 与 depthnet 融合稀疏深度，预测每个像素的深度分布和 80 维相机特征。",
            "img [1,6,3,256,704]；depth [1,6,1,256,704]。",
            "共享图像 backbone/neck 后，把 dtransform(depth) 与图像特征拼接；depthnet 输出 118 个深度权重和 80 通道 feature。",
            "camera_feature [6,32,88,80]；camera_depth_weights [6,118,32,88]。",
            ["camera-model", "camera-runtime", "manifest-camera"], related=["depth", "bevpool", "engine-files"]
        ),
        "geometry": node(
            "Geometry 预计算", "frustum → LiDAR → BEV ranks/intervals", "state",
            "为每个相机、深度 bin 和特征像素计算对应的 BEV 网格位置，并把落在同一格的样本整理为连续 interval。",
            "6 相机标定；feature grid 32×88；depth bins D=118；BEV 360×360×1。",
            "创建 frustum，反投影到相机 3D，再转换到 LiDAR 坐标；过滤越界项、排序 ranks、建立 indices/intervals。",
            "BEVPool indices；intervals(start,end,output_offset)。",
            ["runtime-config", "geometry-kernel", "geometry-setup", "geometry-update"], related=["calibration", "bevpool"]
        ),
        "bevpool": node(
            "CUDA BEVPool", "Lift-Splat 聚合", "compute",
            "把 80 维相机特征沿 118 个深度候选提升到 3D，并依照 geometry lookup 累加到 BEV 平面。",
            "feature [6,32,88,80]；depth weights [6,118,32,88]；indices/intervals。",
            "对每个 interval 累加 feature × depth_weight；累加用 FP32，结果写回 FP16。",
            "camera BEV high-res: [1,80,360,360]，0.3 m 网格。",
            ["bevpool", "geometry-update", "camera-runtime"], related=["camera-backbone", "geometry", "vtransform"],
            formula="BEV[c,y,x] = Σᵢ depth_weight[i] × camera_feature[i,c]"
        ),
        "vtransform": node(
            "Camera VTransform", "camera.vtransform.plan · TensorRT FP16", "compute",
            "对 360×360 相机 BEV 做导出模型中的 downsample，使它和 LiDAR BEV 的空间分辨率对齐。",
            "feat_in: [1,80,360,360]。",
            "运行 TensorRT downsample 网络。",
            "feat_out / camera BEV: [1,80,180,180]，0.6 m 网格。",
            ["camera-export", "vtransform", "manifest-fusion"], related=["bevpool", "camera-bev"]
        ),
        "camera-bev": node(
            "相机 BEV 特征", "[1,80,180,180] · FP16", "output",
            "相机分支最终输出，已从透视图特征变换到与 LiDAR 对齐的鸟瞰网格。",
            "VTransform 输出。",
            "作为 fuser.plan 的 camera binding。",
            "80 通道、180×180 的 BEV feature。",
            ["manifest-fusion", "fusion-runtime"], related=["vtransform", "fuser"]
        ),
        "camera-pipeline": node(
            "Camera Encoder", "CUDA + 2×TensorRT + BEVPool", "compute",
            "相机分支同时使用六路图像、由 LiDAR 投影出的稀疏深度及标定，把多视角特征提升并聚合到 BEV。",
            "images；points；calibration。",
            "Normalize → Camera Backbone/DepthNet → Geometry-aware BEVPool → VTransform。",
            "camera BEV [1,80,180,180]。",
            ["core-forward", "runtime-config", "manifest-camera", "manifest-fusion"], related=["camera-backbone", "camera-bev"], view="camera"
        ),
        "voxelize": node(
            "CUDA 动态体素化", "hash active voxels + reduce mean", "compute",
            "只为有点的体素建立条目，用哈希表收集体素内点，并把每个体素最多 10 个点的五维特征取均值。",
            "points [N,5]；范围 [-54,54]×[-54,54]×[-5,3] m。",
            "以 [0.075,0.075,0.2] m 划分 1440×1440×40 bins；活动体素最多 160000。",
            "features [V,5] FP16；indices [V,4] int32；SCN padded grid [1440,1440,41]。",
            ["runtime-config", "voxelize-kernels", "voxelize"], related=["points", "sparse-tensor"],
            notes=["物理离散体素数的 z 维是 40；代码为 SparseEncoder 的输出 grid 在 z 维额外 +1，因此稀疏 spatial shape 为 41。"]
        ),
        "sparse-tensor": node(
            "稀疏张量", "coordinates [V,4] + features [V,5]", "input",
            "稀疏主干不创建完整 1440×1440×41 的稠密三维特征，只保存活跃坐标和对应特征。",
            "体素化输出。",
            "将 features、indices 和 grid_size 引用到 SCN engine input。",
            "SparseDTensor，V 随当前帧点云变化。",
            ["lidar-scn", "sparse-doc"], related=["voxelize", "libscn"]
        ),
        "onnx-parser": node(
            "Sparse ONNX Parser", "lidar.backbone.xyz.onnx", "config",
            "该 LiDAR 模型不是 TensorRT plan。解析器逐节点读取 SparseConvolution、Add/ReLU 和 ScatterDense，构建开放稀疏执行图。",
            "lidar.backbone.xyz.onnx；FP16 precision。",
            "把 ONNX 属性、权重、bias、kernel/stride/padding/dilation 推入 EngineBuilder。",
            "libscn.so 可执行的稀疏图。",
            ["onnx-parser", "runtime-config"], related=["libscn", "engine-files"],
            notes=["开放后端当前明确限制为 FP16；未实现 NVIDIA 专有 sortmask、Blackwell 和 auxiliary-stream 选项。"]
        ),
        "libscn": node(
            "libscn.so", "稀疏执行图与兼容层 · 本仓库源码", "compute",
            "管理 SparseDTensor 和节点执行顺序，调度 SparseConvolution，并实现 Add、ReLU、融合 Add+ReLU 与 SparseToDense。",
            "解析后的稀疏图；当前帧 SparseDTensor。",
            "按 ONNX 拓扑运行各稀疏节点，并调用底层 open_spconv kernel 库。",
            "最终稠密 LiDAR BEV。",
            ["sparse-build", "sparse-doc", "lidar-scn", "sparse-conv", "scatter-dense"], related=["onnx-parser", "libspconv", "sparse-dense"]
        ),
        "libspconv": node(
            "libspconv.so", "spconv 2.3.8 + cumm 0.7.11 · 源码构建", "compute",
            "提供活跃坐标索引生成、Mask Implicit GEMM 与 CUDA/cuBLASLt 底层实现。本仓库生成的同名库与 NVIDIA 原库 API/ABI 不兼容，需配套 libscn.so。",
            "features/indices、卷积权重与卷积参数。",
            "为每层生成邻接索引和 mask，选择 implicit GEMM kernel 完成稀疏卷积。",
            "新的活跃特征和坐标。",
            ["sparse-build", "sparse-doc", "sparse-conv"], related=["libscn", "indice-pairs", "implicit-gemm"]
        ),
        "indice-pairs": node(
            "Indice Pair / Mask 生成", "active input ↔ active output", "compute",
            "按 kernel、stride、padding、dilation 建立输入活跃行与输出活跃行的邻接关系，并为有效邻居准备 mask 与 argsort。",
            "indices [V,4]；input spatial shape；卷积参数。",
            "调用 get_indice_pairs_implicit_gemm；SubM 保持网格，普通卷积重新计算输出网格。",
            "pair_fwd、pair_mask、mask_argsort、output indices。",
            ["sparse-conv", "sparse-output"], related=["libspconv", "implicit-gemm"]
        ),
        "implicit-gemm": node(
            "Mask Implicit GEMM", "sparse convolution CUDA kernels", "compute",
            "只依据索引关系读取有效邻居，在 kernel 内隐式组织矩阵乘法；不会展开完整稠密 3D im2col。",
            "active features、weights、pair、mask、argsort、可选 bias/activation。",
            "ConvGemmOps::implicit_gemm 选择与 GPU 架构匹配的卷积 kernel。",
            "output active features [Vout,Cout]。",
            ["sparse-conv", "sparse-doc"], related=["indice-pairs", "sparse-stages"]
        ),
        "sparse-stages": node(
            "3D Sparse Backbone", "21×SparseConv · residual Add/ReLU", "compute",
            "多个 SubM block 在同一活动坐标上提取特征，stride convolution 逐级降低 XY 分辨率并扩大通道；垂直卷积把 z 维压到 2。",
            "[V,5] @ 1440×1440×41。",
            "5→16；stride2→32 @720×720×21；→64 @360×360×11；→128 @180×180×5；vertical conv→180×180×2。",
            "稀疏 128 通道、z=2 的活动特征。",
            ["onnx-parser", "sparse-conv", "sparse-output"], related=["implicit-gemm", "sparse-dense"],
            notes=["同一套 indice-pair 与 implicit-GEMM 步骤会在每个稀疏卷积层执行；这里将 21 个卷积和 8 个残差组合并为一个可读模块。"]
        ),
        "sparse-dense": node(
            "ScatterDense + reshape", "稀疏尾部才转稠密", "compute",
            "将最后一层的活动特征 scatter 到 180×180×2 网格，并把 z×channel 合并成 256 个 BEV 通道。",
            "active features/indices，128 channels，z=2。",
            "先清零输出，再按 XYZ 坐标 scatter；按导出图的 transpose/reshape 得到 NCHW。",
            "LiDAR BEV [1,256,180,180] FP16。",
            ["scatter-dense", "sparse-doc"], related=["sparse-stages", "lidar-bev"]
        ),
        "lidar-bev": node(
            "LiDAR BEV 特征", "[1,256,180,180] · FP16", "output",
            "LiDAR 稀疏 3D 主干的稠密鸟瞰输出，与相机 BEV 共享 180×180 空间网格。",
            "SparseToDense/reshape 输出。",
            "作为 fuser.plan 的 lidar binding。",
            "256 通道、180×180 的 BEV feature。",
            ["fusion-bindings", "fusion-runtime", "scatter-dense"], related=["sparse-dense", "fuser"]
        ),
        "lidar-pipeline": node(
            "LiDAR Sparse Encoder", "CUDA voxelize + libscn.so + libspconv.so", "compute",
            "点云只在活动体素上执行 3D 稀疏卷积；项目内源码后端取代原先依赖特定 GPU 架构的 NVIDIA 动态库。",
            "points [N,5] FP16。",
            "Voxelize → ONNX parser/libscn → indice generation/Mask Implicit GEMM → ScatterDense。",
            "LiDAR BEV [1,256,180,180]。",
            ["lidar-scn", "sparse-build", "sparse-doc"], related=["voxelize", "libscn", "libspconv", "lidar-bev"], view="lidar"
        ),
        "fuser": node(
            "BEV Feature Fuser", "fuser.plan · TensorRT FP16", "compute",
            "相机与 LiDAR BEV 先沿通道维拼接，ConvFuser 用 3×3 卷积完成局部跨模态混合；SECOND 提取两个尺度，SECONDFPN 对齐空间尺寸后拼成检测头输入。",
            "camera [1,80,180,180]；lidar [1,256,180,180]。",
            "Cat(C): 80+256→336；Conv3×3 s1 p1: 336→256 + BN + ReLU；SECOND stage0: 256→128、6×Conv3×3 @180²；stage1: 128→256、首层 s2 后共 6×Conv3×3 @90²；SECONDFPN: 1×1 Conv(128→256) 与 2×2 Deconv s2(256→256)；最后 Cat(C)。",
            "middle: [1,512,180,180] FP16。",
            ["fusion-runtime", "fusion-export", "fusion-bindings", "manifest-fusion", "convfuser-config"],
            notes=[
                "SECOND 的 layer_nums=[5,5] 表示每个 stage 在入口卷积之后再接 5 个卷积，因此每个 stage 共 6 个 Conv-BN-ReLU。",
                "SECONDFPN 的两路输出都是 [1,256,180,180]；512 通道来自这两路按通道拼接，不是 ConvFuser 直接输出 512。",
                "这一段已经是稠密 2D BEV 卷积；3D 稀疏卷积在 LiDAR 分支生成 [1,256,180,180] 之前已经结束。",
                "TensorRT 构建时通常会折叠 BN 并融合 Conv/BN/ReLU；图中按训练框架的逻辑层展示。",
            ],
            related=["camera-bev", "lidar-bev", "fused-bev"], view="fusion",
            formula=(
                "X = Cat(camera, lidar, dim=1) ∈ R^{1×336×180×180}\n"
                "F = ReLU(BN(Conv3×3(X))) ∈ R^{1×256×180×180}\n"
                "B0 = SECOND₀(F) ∈ R^{1×128×180×180}\n"
                "B1 = SECOND₁(B0) ∈ R^{1×256×90×90}\n"
                "U0 = ReLU(BN(Conv1×1(B0))) ∈ R^{1×256×180×180}\n"
                "U1 = ReLU(BN(Deconv2×2,s2(B1))) ∈ R^{1×256×180×180}\n"
                "middle = Cat(U0, U1, dim=1) ∈ R^{1×512×180×180}"
            )
        ),
        "fused-bev": node(
            "融合 BEV / middle", "[1,512,180,180] · FP16", "output",
            "相机语义与 LiDAR 几何经过 ConvFuser 和 2D decoder 后形成检测头的统一特征。",
            "camera BEV + LiDAR BEV。",
            "由 fuser.plan 输出，直接绑定到 head.bbox.plan。",
            "middle [1,512,180,180]。",
            ["fusion-runtime", "fusion-bindings", "manifest-fusion", "manifest-head"], related=["fuser", "proposal"]
        ),
        "proposal": node(
            "Heatmap + Top-200 Queries", "heatmap-guided query initialization", "compute",
            "共享卷积产生 dense heatmap；局部极大值抑制后，在类别×BEV 位置中选出 200 个 proposal。",
            "middle [1,512,180,180]。",
            "shared_conv → heatmap_head → sigmoid/local max → Top-K；从展平 BEV gather query feature 与位置，并加入类别编码。",
            "query feature / query position：P=200。",
            ["proposal-head", "manifest-head"], related=["fused-bev", "transformer-head"]
        ),
        "transformer-head": node(
            "TransFusion Decoder", "BEV query ↔ fused BEV", "compute",
            "让 200 个候选 Query 以融合 BEV 特征为 key/value 做 Transformer decoding，并逐层修正中心位置。",
            "query_feat [B,C,200]；flattened BEV [B,C,32400]；query/bev positions。",
            "decoder layers 进行 cross-attention；prediction heads 输出 center/height/dim/rot/vel/heatmap，并更新下一层 query_pos。",
            "每个 query 的分类与 3D 框参数。",
            ["proposal-head"], related=["proposal", "raw-head"], view="fusion"
        ),
        "raw-head": node(
            "检测头原始输出", "head.bbox.plan · TensorRT FP16", "output",
            "TensorRT 检测头输出 200 个 proposal 的类别分数、旋转、尺寸、平面中心、高度和速度。",
            "middle [1,512,180,180]。",
            "计算 query heatmap score 与分类分数，并导出六组固定 binding。",
            "score [1,10,200]；rot [1,2,200]；dim [1,3,200]；reg [1,2,200]；height [1,1,200]；vel [1,2,200]。",
            ["raw-head", "fusion-bindings", "manifest-head", "head-runtime"], related=["transformer-head", "decode"]
        ),
        "decode": node(
            "CUDA 框解码与过滤", "threshold=0.12 · range filter · score sort", "compute",
            "把 200 个原始候选恢复为物理坐标中的 3D 框，按置信度和空间范围过滤，并按分数降序排序。",
            "reg/height/dim/rot/vel/score；out_size_factor=8；voxel_size=0.075。",
            "x/y 映射到点云范围；dim 取 exp；yaw=atan2(rot_sin,rot_cos)；z 转到底部中心；取最高类别分数。",
            "最多 1024 个 BoundingBox（当前 head 仅有 200 个 query）。",
            ["runtime-config", "decode", "head-runtime"], related=["raw-head", "output"],
            formula="x = regₓ × 8 × 0.075 − 54\ny = regᵧ × 8 × 0.075 − 54\nsize = exp(dim),  yaw = atan2(rot₀, rot₁)"
        ),
        "output": node(
            "3D 目标检测结果", "Python: [x,y,z,w,l,h,yaw,vx,vy,class_id,score]", "output",
            "当前部署路径输出 nuScenes 三维目标框、速度、置信度和类别；没有导出语义分割结果。",
            "通过阈值与范围过滤的候选。",
            "从 GPU 回传 C++ BoundingBox；Python binding 按固定列顺序打包为 float32 数组。",
            "C++：K 个 BoundingBox；Python：[K,11]，列为 x,y,z,w,l,h,yaw,vx,vy,class_id,score。K≤200（缓冲容量 1024）。",
            ["decode", "head-runtime", "core-forward", "python-output"], related=["decode"]
        ),
        "core": node(
            "BEVFusion Core", "C++ orchestration · one CUDA stream", "config",
            "负责创建所有模块、持有模型路径与缓冲区、更新标定，并按部署拓扑串联相机、LiDAR、融合和检测后处理。",
            "CoreParameter 与同步传感器输入。",
            "init 创建 9 个子模块；forward_only 在同一 stream 上顺序提交 CUDA/TensorRT/SCN 操作。",
            "std::vector<BoundingBox>。",
            ["core-init", "core-forward", "core-api"], related=["input", "engine-files"]
        ),
        "engine-files": node(
            "部署模型边界", "3×TRT branch engines + 1×TRT head + sparse ONNX", "config",
            "相机、VTransform、Fuser 与 bbox head 使用 TensorRT plan；LiDAR 稀疏主干保留 ONNX，由 libscn.so 直接解析执行。",
            "camera.backbone.plan；camera.vtransform.plan；fuser.plan；head.bbox.plan；lidar.backbone.xyz.onnx。",
            "Core 启动时加载固定 shape FP16 engine，并为每个 binding 分配 GPU 缓冲。",
            "四个 TensorRT 执行段 + 一个开放稀疏执行段。",
            ["runtime-config", "core-init", "onnx-parser", "manifest-camera", "manifest-fusion", "manifest-head"], related=["camera-backbone", "onnx-parser", "fuser", "raw-head"]
        ),
    }

    views = [
        {
            "id": "overview",
            "title": "完整模型",
            "subtitle": "从同步相机/点云输入到 3D 检测框。两条感知分支在 180×180 BEV 空间对齐并融合。",
            "rows": 9,
            "nodes": [
                {"id": "input", "row": 0, "col": 2, "caption": "6×Camera · LiDAR [N,5] · Calibration"},
                {"id": "images", "row": 1, "col": 1, "caption": "6×1600×900 RGB"},
                {"id": "calibration", "row": 1, "col": 2, "caption": "投影 + BEV geometry"},
                {"id": "points", "row": 1, "col": 3, "caption": "N≤300000 · 还参与 depth projection"},
                {"id": "camera-pipeline", "row": 2, "col": 1, "caption": "CUDA + TensorRT + BEVPool"},
                {"id": "lidar-pipeline", "row": 2, "col": 3, "caption": "开源 libscn.so + libspconv.so"},
                {"id": "camera-bev", "row": 3, "col": 1},
                {"id": "lidar-bev", "row": 3, "col": 3},
                {"id": "fuser", "row": 4, "col": 2, "caption": "fuser.plan · camera + lidar"},
                {"id": "fused-bev", "row": 5, "col": 2},
                {"id": "transformer-head", "row": 6, "col": 2, "caption": "Top-200 object queries"},
                {"id": "decode", "row": 7, "col": 2},
                {"id": "output", "row": 8, "col": 2, "caption": "3D boxes · score · class · velocity"},
            ],
            "edges": [
                {"from": "input", "to": "images", "kind": "data"},
                {"from": "input", "to": "points", "kind": "data"},
                {"from": "input", "to": "calibration", "kind": "state"},
                {"from": "images", "to": "camera-pipeline", "kind": "data"},
                {"from": "points", "to": "camera-pipeline", "kind": "data", "label": "depth"},
                {"from": "calibration", "to": "camera-pipeline", "kind": "state"},
                {"from": "points", "to": "lidar-pipeline", "kind": "data"},
                {"from": "camera-pipeline", "to": "camera-bev", "kind": "data"},
                {"from": "lidar-pipeline", "to": "lidar-bev", "kind": "data"},
                {"from": "camera-bev", "to": "fuser", "kind": "data"},
                {"from": "lidar-bev", "to": "fuser", "kind": "data"},
                {"from": "fuser", "to": "fused-bev", "kind": "data"},
                {"from": "fused-bev", "to": "transformer-head", "kind": "data"},
                {"from": "transformer-head", "to": "decode", "kind": "data"},
                {"from": "decode", "to": "output", "kind": "data"},
            ],
            "footnote": "实际 Core::forward 在同一 CUDA stream 上先计算 LiDAR sparse backbone，再计算相机分支，然后执行 fuser 与检测头；本图按数据依赖并列展示两条分支。LiDAR 点云还会进入相机深度投影，这是两分支间的交叉输入。",
        },
        {
            "id": "camera",
            "title": "展开相机分支",
            "subtitle": "图像特征与 LiDAR 稀疏深度共同估计 118 个深度权重，再按标定聚合为 BEV。",
            "rows": 6,
            "nodes": [
                {"id": "images", "row": 0, "col": 1},
                {"id": "calibration", "row": 0, "col": 2},
                {"id": "points", "row": 0, "col": 3},
                {"id": "normalize", "row": 1, "col": 1, "caption": "[1,6,3,256,704]"},
                {"id": "geometry", "row": 1, "col": 2, "caption": "indices + intervals"},
                {"id": "depth", "row": 1, "col": 3, "caption": "[1,6,1,256,704]"},
                {"id": "camera-backbone", "row": 2, "col": 2, "caption": "feature [6,32,88,80]\ndepth weights [6,118,32,88]"},
                {"id": "bevpool", "row": 3, "col": 2, "caption": "[1,80,360,360] · 0.3 m"},
                {"id": "vtransform", "row": 4, "col": 2, "caption": "TensorRT downsample"},
                {"id": "camera-bev", "row": 5, "col": 2, "caption": "[1,80,180,180] · 0.6 m"},
            ],
            "edges": [
                {"from": "images", "to": "normalize", "kind": "data"},
                {"from": "points", "to": "depth", "kind": "data"},
                {"from": "calibration", "to": "depth", "kind": "state", "label": "lidar2image"},
                {"from": "calibration", "to": "geometry", "kind": "state"},
                {"from": "normalize", "to": "camera-backbone", "kind": "data", "label": "img"},
                {"from": "depth", "to": "camera-backbone", "kind": "data", "label": "depth"},
                {"from": "camera-backbone", "to": "bevpool", "kind": "data", "label": "feature + weights"},
                {"from": "geometry", "to": "bevpool", "kind": "state", "label": "lookup"},
                {"from": "bevpool", "to": "vtransform", "kind": "data"},
                {"from": "vtransform", "to": "camera-bev", "kind": "data"},
            ],
            "footnote": "相机特征采用 NHWC [6,32,88,80]，深度权重为 [6,118,32,88]。BEVPool 先生成 360×360 高分辨率相机 BEV，再由 VTransform 下采样到与 LiDAR 相同的 180×180。",
        },
        {
            "id": "lidar",
            "title": "展开 LiDAR / 开源稀疏卷积分支",
            "subtitle": "展示从动态体素到源码动态库、稀疏卷积 kernel 和末端 ScatterDense 的完整调用关系。",
            "rows": 9,
            "nodes": [
                {"id": "points", "row": 0, "col": 2},
                {"id": "voxelize", "row": 1, "col": 2, "caption": "bins 1440×1440×40\nsparse grid 1440×1440×41"},
                {"id": "sparse-tensor", "row": 2, "col": 2, "caption": "features [V,5] + indices [V,4]"},
                {"id": "onnx-parser", "row": 2, "col": 1, "caption": "lidar.backbone.xyz.onnx"},
                {"id": "libscn", "row": 3, "col": 2, "caption": "graph / Add / ReLU / Scatter"},
                {"id": "libspconv", "row": 4, "col": 2, "caption": "spconv + cumm · CUDA/cuBLASLt"},
                {"id": "indice-pairs", "row": 5, "col": 1, "caption": "pair · mask · argsort"},
                {"id": "implicit-gemm", "row": 5, "col": 3, "caption": "active features × weights"},
                {"id": "sparse-stages", "row": 6, "col": 2, "caption": "21 conv · 8 residual\n1440→720→360→180"},
                {"id": "sparse-dense", "row": 7, "col": 2, "caption": "scatter + z/channel reshape"},
                {"id": "lidar-bev", "row": 8, "col": 2, "caption": "[1,256,180,180]"},
            ],
            "edges": [
                {"from": "points", "to": "voxelize", "kind": "data"},
                {"from": "voxelize", "to": "sparse-tensor", "kind": "data"},
                {"from": "sparse-tensor", "to": "libscn", "kind": "data"},
                {"from": "onnx-parser", "to": "libscn", "kind": "control", "label": "build graph"},
                {"from": "libscn", "to": "libspconv", "kind": "control", "label": "dispatch"},
                {"from": "libspconv", "to": "indice-pairs", "kind": "data"},
                {"from": "indice-pairs", "to": "implicit-gemm", "kind": "data"},
                {"from": "implicit-gemm", "to": "sparse-stages", "kind": "data"},
                {"from": "sparse-stages", "to": "sparse-dense", "kind": "data"},
                {"from": "sparse-dense", "to": "lidar-bev", "kind": "data"},
            ],
            "footnote": "libscn.so 与 libspconv.so 都由本仓库源码构建。libscn.so 管执行图和稀疏算子；libspconv.so 提供 indice pair 与 Mask Implicit GEMM kernel。索引生成和 GEMM 在每个稀疏卷积层重复，图中为避免交叉线只展开一次。",
        },
        {
            "id": "fusion",
            "title": "展开 BEV 融合与检测头",
            "subtitle": "两种传感器特征在同一 180×180 网格融合，TransFusion 用 200 个 object query 预测并解码 3D 框。",
            "rows": 8,
            "nodes": [
                {"id": "camera-bev", "row": 0, "col": 1, "caption": "[1,80,180,180]"},
                {"id": "lidar-bev", "row": 0, "col": 3, "caption": "[1,256,180,180]"},
                {"id": "fuser", "row": 1, "col": 2, "caption": "Cat 336 → ConvFuser 256 → SECOND → FPN 256+256"},
                {"id": "fused-bev", "row": 2, "col": 2, "caption": "middle [1,512,180,180]"},
                {"id": "proposal", "row": 3, "col": 2, "caption": "dense heatmap → local max → Top-200"},
                {"id": "transformer-head", "row": 4, "col": 2, "caption": "query feature + BEV key/value"},
                {"id": "raw-head", "row": 5, "col": 2, "caption": "score / rot / dim / reg / height / vel"},
                {"id": "decode", "row": 6, "col": 2, "caption": "threshold 0.12 + range filter + sort"},
                {"id": "output", "row": 7, "col": 2, "caption": "K≤200 3D boxes"},
            ],
            "edges": [
                {"from": "camera-bev", "to": "fuser", "kind": "data"},
                {"from": "lidar-bev", "to": "fuser", "kind": "data"},
                {"from": "fuser", "to": "fused-bev", "kind": "data"},
                {"from": "fused-bev", "to": "proposal", "kind": "data"},
                {"from": "proposal", "to": "transformer-head", "kind": "data"},
                {"from": "transformer-head", "to": "raw-head", "kind": "data"},
                {"from": "raw-head", "to": "decode", "kind": "data"},
                {"from": "decode", "to": "output", "kind": "data"},
            ],
            "footnote": "fuser.plan 已包含 ConvFuser、decoder backbone 与 neck；head.bbox.plan 包含 proposal 初始化、Transformer decoder 和六组预测 head。C++ 端只负责 CUDA 解码、阈值/范围过滤及排序。当前导出路径只支持 object 3D detection。",
        },
    ]

    return {
        "title": "BEVFusion · TensorRT 推理架构与源码",
        "sourceRoot": "Robot_infer/bevfusion_agx/ · master",
        "generatedNote": "2026-10-25 · 依据当前 master 的 C++/CUDA 推理路径、TensorRT 导出脚本、已验证 engine manifest 与项目内开放稀疏后端生成。",
        "sources": sources,
        "nodes": nodes,
        "views": views,
        "quickNodes": ["core", "engine-files", "camera-backbone", "libscn", "libspconv", "fuser", "raw-head"],
    }


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise ValueError(f"Template fragment not found: {old[:100]!r}")
    return text.replace(old, new, 1)


def validate_catalog(catalog: dict) -> None:
    node_ids = set(catalog["nodes"])
    source_ids = set(catalog["sources"])
    view_ids = {view["id"] for view in catalog["views"]}
    if len(view_ids) != len(catalog["views"]):
        raise ValueError("Duplicate view id")
    for node_id, item in catalog["nodes"].items():
        missing_sources = set(item["sources"]) - source_ids
        missing_related = set(item["related"]) - node_ids
        if missing_sources or missing_related:
            raise ValueError(f"Broken references in node {node_id}: sources={missing_sources}, related={missing_related}")
        if item["view"] is not None and item["view"] not in view_ids:
            raise ValueError(f"Node {node_id} links to missing view {item['view']}")
    for view in catalog["views"]:
        visible = {item["id"] for item in view["nodes"]}
        if not visible <= node_ids:
            raise ValueError(f"View {view['id']} uses missing nodes: {visible - node_ids}")
        for edge in view["edges"]:
            if edge["from"] not in visible or edge["to"] not in visible:
                raise ValueError(f"View {view['id']} has an edge outside its visible nodes: {edge}")
    if set(catalog["quickNodes"]) - node_ids:
        raise ValueError("quickNodes contains an unknown node")


def build(template_path: Path, output_path: Path) -> None:
    html = template_path.read_text(encoding="utf-8")
    replacements = [
        (
            '<meta name="description" content="BEVFormer PyTorch 模型结构：点击模块查看作用、张量形状与对应的真实源码。">',
            '<meta name="description" content="BEVFusion TensorRT 推理架构：相机、LiDAR 开源稀疏卷积、BEV 融合、TransFusion 检测头与真实源码。">',
        ),
        ("<title>BEVFormer · 模型结构与源码</title>", "<title>BEVFusion · TensorRT 推理架构与源码</title>"),
        (
            '<div class="topline"><a class="back-link" href="index.html">← 架构图目录</a><span class="edition">PYTORCH MODEL / SOURCE MAP</span></div>',
            '<div class="topline"><a class="back-link" href="../README.md">← 项目主页</a><span class="edition">TENSORRT / CUDA / OPEN SPARSE SOURCE MAP</span></div>',
        ),
        ('<div class="eyebrow">BEVFORMER · 模型结构</div>', '<div class="eyebrow">BEVFUSION · 部署架构</div>'),
        ('<h1 id="page-title">BEVFormer 模型结构与源码</h1>', '<h1 id="page-title">BEVFusion TensorRT 推理架构与源码</h1>'),
        (
            '<p>从 <strong>6 路相机图像</strong>，到 <strong>BEV 空间表示</strong>，再到 <strong>3D 检测结果</strong>。点击图中任一模块，查看它做什么、张量如何变化，以及对应的 PyTorch 实现。</p>',
            '<p>从 <strong>6 路相机 + LiDAR 点云</strong>，到 <strong>双分支 BEV 特征</strong>，再到 <strong>TransFusion 3D 检测结果</strong>。点击任一模块，查看真实张量形状、TensorRT/CUDA 边界，以及项目内 <strong>libscn.so + libspconv.so</strong> 的源码调用链。</p>',
        ),
        (
            '<span><code>B</code> batch</span><span><code>N = 6</code> 相机数</span><span><code>C = 256</code> BEV 特征维度</span><span>示例输入 <code>1 × 6 × 3 × 928 × 1600</code></span>',
            '<span><code>B = 1</code> batch</span><span><code>Ncam = 6</code> 相机数</span><span><code>FP16</code> 部署精度</span><span>融合网格 <code>180 × 180</code></span><span>检测 Query <code>200</code></span>',
        ),
        ('<span class="object-nav-label">对象层级 · 点击查看</span>', '<span class="object-nav-label">关键执行对象 · 点击查看</span>'),
        (
            '<div class="legend" aria-label="图例"><span class="legend-data"><i class="legend-line" aria-hidden="true"></i>数据流</span><span class="legend-state"><i class="legend-line state" aria-hidden="true"></i>跨帧状态</span><span><i class="legend-dot" aria-hidden="true"></i>当前选择</span><span class="click-hint">点击模块查看说明与源码 ↗</span></div>',
            '<div class="legend" aria-label="图例"><span class="legend-data"><i class="legend-line" aria-hidden="true"></i>数据流</span><span class="legend-state"><i class="legend-line state" aria-hidden="true"></i>标定 / Geometry</span><span><i class="legend-line control" aria-hidden="true"></i>模型 / 调度</span><span><i class="legend-dot" aria-hidden="true"></i>当前选择</span><span class="click-hint">点击模块查看说明与源码 ↗</span></div>',
        ),
        (
            '<footer><span>范围：BEVFormer 的 PyTorch 模型结构与本地源码快照。框架片段 © OpenMMLab，Apache-2.0；对应上游版本链接见源码详情。</span><span>形状用于解释数据流；本页不执行模型，也不代表实测推理性能。</span></footer>',
            '<footer><span>范围：BEVFusion C++/CUDA/TensorRT 推理与仓库内开放稀疏后端。源码许可证与上游版本见项目清单及各源码详情。</span><span>形状来自当前导出脚本和 engine manifest；本页展示 object 3D detection，不包含 segmentation，也不代表实测性能。</span></footer>',
        ),
        (
            "const kinds = {input:'输入张量',compute:'计算模块',output:'输出张量',state:'跨帧状态',config:'模型配置'};",
            "const kinds = {input:'输入张量',compute:'计算模块',output:'输出张量',state:'标定 / 几何',config:'模型 / 调度'};",
        ),
        ("addLink(links,source.url,'上游源码链接 ↗');", "addLink(links,source.url,'GitHub 源文件 ↗');"),
        (
            "const connectedName = (edge,key) => (catalog.nodes[edge[key]]?.title || edge[key]) + (edge.kind === 'state' ? '（跨帧状态）' : '');",
            "const connectedName = (edge,key) => (catalog.nodes[edge[key]]?.title || edge[key]) + (edge.kind === 'state' ? '（标定 / 几何）' : edge.kind === 'control' ? '（模型 / 调度）' : '');",
        ),
        ("catalog.title || 'BEVFormer 模型结构与源码'", "catalog.title || 'BEVFusion TensorRT 推理架构与源码'"),
        ("catalog.title || 'BEVFormer · 模型结构与源码'", "catalog.title || 'BEVFusion · TensorRT 推理架构与源码'"),
    ]
    for old, new in replacements:
        html = replace_once(html, old, new)

    html = replace_once(
        html,
        ".legend-line.state:after{border-left-color:var(--purple)}",
        ".legend-line.state:after{border-left-color:var(--purple)}.legend-line.control{border-top-color:var(--orange);border-top-style:dotted}.legend-line.control:after{border-left-color:var(--orange)}",
    )

    catalog = build_catalog()
    validate_catalog(catalog)
    catalog_json = json.dumps(catalog, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    pattern = re.compile(r'<script id="catalog-data" type="application/json">.*?</script>', re.DOTALL)
    replacement = f'<script id="catalog-data" type="application/json">{catalog_json}</script>'
    html, count = pattern.subn(lambda _: replacement, html, count=1)
    if count != 1:
        raise ValueError("Could not replace catalog-data in template")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8", newline="\n")
    if re.search(r"<script[^>]+src=", html, re.IGNORECASE) or re.search(r"<link[^>]+stylesheet", html, re.IGNORECASE):
        raise ValueError("Generated page unexpectedly depends on an external script or stylesheet")
    if html.count('<script id="catalog-data" type="application/json">') != 1:
        raise ValueError("Generated page does not contain exactly one catalog-data block")
    print(f"Wrote {output_path}")
    print(f"Nodes: {len(catalog['nodes'])}; sources: {len(catalog['sources'])}; views: {len(catalog['views'])}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.template.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
