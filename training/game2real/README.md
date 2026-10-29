# Game2Real: content-preserving Transformer U-Net

独立、可训练的 **GTA5 → SeSekai2 实拍街景外观转换**。
两套数据独立采样，不要求同场景配对。不下载、转换或假定服务器数据集的专用格式。
支持普通 RGB 图片目录和 JSONL 清单。

这是一套新的 RGB → RGB 训练实现，不是 NVIDIA NR 权重的微调器。
生成器是多尺度 Transformer U-Net；没有扩散模型、采样器或去噪训练目标。
预训练 VGG、DINOv2、SegFormer 仅在训练/评估时提供监督，导出的生成器不依赖它们。

**目标是提升摄影真实性，同时约束原场景内容。结构保持是优化目标，不是数学保证。**
无配对真实图像不能作为游戏图像的逐像素真值。完整画质结论必须来自独立数据评估和人工对比。
本仓库提供可运行代码与初始研究配置，不声称默认超参数已经达到最佳结果。

## 1. 服务器环境

建议 Python 3.11、Linux、与驱动匹配的 CUDA PyTorch。先安装**匹配版本**的 PyTorch 与 torchvision：

```bash
# 以下组合在本地 CUDA 检查中实际使用；服务器驱动必须支持对应运行时。
python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128

cd open-dlss/training/game2real
python -m pip install -e '.[test]'
python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0))'
```

已有兼容 PyTorch 环境时，保留服务器的安装，不必重装上面的组合。
CUDA Toolkit 不必与 wheel 运行时完全相同；以 PyTorch 官方安装说明和驱动要求为准。
H800 使用 `bf16`。多卡通过 `torchrun` / NCCL 运行；默认每 GPU 2 张 512² 图块、累积 2 次。
有效 batch = `batch_size × GPU 数量 × accumulation`。
该默认值尚未在 H800 上做容量或吞吐标定。

## 2. 数据准备

只把 RGB 图像目录传给扫描器。GTA5 的标签目录不能混在 RGB 目录里当作图像扫描。

```text
/datasets/GTA5/images/...
/datasets/SeSekai2/images/...
```

生成独立的训练/验证清单：

```bash
python -m gta2real.prepare_data \
  --source-root /datasets/GTA5/images \
  --target-root /datasets/SeSekai2/images \
  --out manifests --val-fraction 0.05
```

输出：`source_train.jsonl`、`source_val.jsonl`、`target_train.jsonl`、`target_val.jsonl`、`split_summary.json`。
先检查 summary 中的分组依据与警告。相邻帧不得随机散落到 train/val。
分组优先级为显式 `sequence_id`、`--group-regex`、子目录、文件名序列前缀。
例如 `clip_a_0001.png` 与 `clip_a_0002.png` 属于同组。
共享当前帧/前一帧引用的组也会合并。只有一组时验证集保持为空，不伪造独立验证。
无法识别序列的图片按独立图片处理；这不能证明它们在时序上独立。

已有清单可以使用 `--source-manifest` 和 `--target-manifest` 代替 roots。
每个 JSONL 记录只要求 `image`：

```json
{"image":"../GTA5/images/drive01/000123.png","sequence_id":"drive01","frame_index":123}
```

所有相对路径相对于清单位置解析。未知元数据保留在 `tags`。
真实域的记录完全独立；相同序号不表示与源域配对。
两套数据最好在视角、物体类型、尺度、天气和曝光范围上有足够重叠。
第三人称游戏与车载前视实拍存在视角差异时，重点检查是否改动物体或布局。

### 可选标签与时序数据

```json
{"image":"rgb/000123.png","label":"labels/000123.png","sequence_id":"drive01","frame_index":123,"previous":"rgb/000122.png","flow":"flow/000123.npy","mask":"masks/000123.png"}
```

- `label`：单通道类别图，编号必须对应 Cityscapes **train IDs 0–18**，`255` 为 ignore。
  如果是 GTA5 原始类别 ID，在 YAML 的 `data.label_map` 中明确映射；未映射 ID 忽略。
- `previous`：与当前 RGB 同尺寸的前一帧。
- `flow`：`.npy`，`H×W×2` 或 `2×H×W`，**当前像素 → 前一帧对应位置**的像素位移。
  它是 backward sampling flow，不是通常的 previous → current forward flow。
- `mask`：有效区域，1 为有效、0 为排除；可用 `[0,1]` 浮点或整数图像范围。
- RGB、标签、前一帧、光流、遮罩采用相同源域几何变换。
  resize 会缩放 flow 分量；水平翻转会反转 x 位移。

没有这些字段也可以运行 RGB 单帧训练。代码不会伪造零光流或深度。
混合 batch 中缺少的可选字段会被省略，并记录 `data/provided_*` 和 `data/temporal_batch`。
没有有效 previous/flow 时，时序损失不生效，不能据此宣称训练过视频一致性。
要求每个 batch 都有时序监督时，设置 `train.require_temporal: true`；缺失会明确失败。

## 3. 预训练监督模型

在线机器运行一次：

```bash
python -m gta2real.prepare_teachers --out weights
```

它准备 ImageNet VGG16、DINOv2-base 和 Cityscapes SegFormer-b2，并生成 `weights/teachers.local.yaml`。
下载失败会退出，**不会静默替换成随机网络**。
可用 `--dino-revision`、`--semantic-revision` 固定 Hugging Face 修订号。
导出的目录仅含公开模型配置与权重，不含账号凭证。
内网服务器可复制 weights 目录，再调整 YAML 的本地路径。
`--local-files-only` 仅使用已有缓存；它不代表为缺失模型生成替代权重。

## 4. 可选的内容候选检索

为减轻两个数据集的内容分布差异，可先在**训练集内部**建立候选索引：

```bash
python -m gta2real.match_data \
  --source manifests/source_train.jsonl \
  --target manifests/target_train.jsonl \
  --model weights/dinov2 --local-files-only \
  --output manifests/matches.json --topk 8
```

检索使用冻结 DINO 图像特征的余弦相似度，并缓存描述符。
这是**整图级候选检索**，不是 EPE 的完整 patch matching，也不是像素对齐伪标签。
训练仍对两域分别随机裁剪；`matches.json` 从未作为像素级目标使用。
索引只能包含训练域目标，不能把验证实拍图混入候选。
没有索引时可以直接训练；比较检索有无效果属于后续实测。

## 5. H800 启动

```bash
# 单卡：NPROC_PER_NODE=1；多卡替换成实际 GPU 数量。
NPROC_PER_NODE=1 bash scripts/launch_h800.sh \
  --source manifests/source_train.jsonl \
  --target manifests/target_train.jsonl \
  --val-source manifests/source_val.jsonl \
  --val-target manifests/target_val.jsonl \
  --teachers-config weights/teachers.local.yaml \
  --match-index manifests/matches.json \
  --output runs/gta5_sesekai2
```

未生成候选索引时去掉 `--match-index`。验证清单必须非空且独立。
若暂时没有独立验证，两个 `--val-*` 一起省略；此时只保存 latest，不生成验证选出的 selected。
禁止将训练清单原样传作验证清单，代码会检查图像、历史引用与显式序列 ID 的重叠。

显存调整无需改代码：

```bash
NPROC_PER_NODE=8 bash scripts/launch_h800.sh \
  --source manifests/source_train.jsonl --target manifests/target_train.jsonl \
  --val-source manifests/source_val.jsonl --val-target manifests/target_val.jsonl \
  --teachers-config weights/teachers.local.yaml \
  --batch-size 2 --accumulation 2 --crop-size 512
```

推荐先用 `--steps 100` 检查真实数据读取、loss 有效覆盖率与比较图，再开始完整实验。
从短试跑继续到完整计划时，可以提高 `--steps`；这会改变后续 cosine 调度，不等同于不中断的原计划。

断点恢复：

```bash
NPROC_PER_NODE=1 bash scripts/launch_h800.sh \
  --source manifests/source_train.jsonl --target manifests/target_train.jsonl \
  --val-source manifests/source_val.jsonl --val-target manifests/target_val.jsonl \
  --teachers-config weights/teachers.local.yaml \
  --resume runs/gta5_sesekai2/checkpoints/latest.pt
```

恢复时使用原来的匹配索引参数、GPU 数量、数据、seed、batch、workers、crop 和累积设置。
检查点保存 G/D、EMA、两个优化器、AMP scaler、每 rank RNG 和数据流位置。
DataLoader 重建同一 epoch 并重放已消费 batch，因此恢复大 epoch 时需要读取之前的 batch。
数据记录指纹用于检测清单变化，不等同于对全部图像内容做密码学校验。
检查点通过 `torch.load` 读取，只加载自己生成的可信检查点。

## 6. 训练目标与结构约束

生成器：四尺度 shifted-window Transformer U-Net，带相对位置偏置、局部 depthwise FFN、跨尺度跳连。
初始残差头为零，所以初始化输出严格等于输入。
输出是 `clamp(input + max_residual * tanh(head), 0, 1)`，默认逐通道变化上限为 0.35。
它不包含显式空间形变、光流预测或重新布局模块；这些设计仍不保证语义绝对不变。

| 目标 | 监督与作用 |
|---|---|
| 真实性 | 冻结 VGG 多尺度特征上的谱归一化判别器、hinge GAN |
| 内容与局部对应 | 源图/输出的 VGG 特征、同坐标 PatchNCE |
| 结构 | 源图/输出的冻结 DINO patch 特征余弦距离 |
| 语义 | 真实标签 CE；无标签时为高置信源图教师预测的一致性 KL |
| 边缘与大范围色调 | Sobel 边缘方向/梯度、低频图像约束 |
| 实拍恒等 | `G(real)` 接近 real，减少已有实拍外观的无谓改动 |
| 时序（有数据才生效） | 对齐前一帧输出的有效区域一致性 |
| 数值稳定 | 身份初始化、对抗权重渐增、梯度裁剪、EMA、惰性 feature-space R1 |

DINO 与语义教师处理相同位置的源图和输出，不拿任意实拍照片做内容目标。
语义教师不是人工真值，尤其需要检查它在游戏域的可信覆盖率。
所有权重、教师分辨率和选择阈值可在 `configs/quality_h800.yaml` / 配置默认值中调整。
H800 使用 BF16、SDPA 和 activation checkpoint；FP16 也提供 scaler，CPU 使用 FP32。

## 7. 输出与评估

```text
runs/gta5_sesekai2/
  config.json                 # 解析后的实际配置
  environment.json            # PyTorch、GPU、effective batch
  train.jsonl                 # 每项 loss、有效 guide 计数、耗时
  checkpoints/latest.pt       # 最近完整训练状态
  checkpoints/selected.pt     # 通过内容阈值且 style proxy 最低的验证选择
  validation/00002000.json    # 固定验证 crop 指标
  validation/00002000/         # 左原图、右输出的比较图
```

`selected.pt` 仅表示验证代理指标选择：固定 VGG 特征的均值/标准差距离，带边缘、DINO 和语义阈值。
它不是“最佳真实感”的证明。阈值必须结合 held-out 对比校准，且不存在源/真实配对 PSNR。
日志中的输入/输出差值也不是画质分数。

原尺寸推理，默认 EMA：

```bash
python -m gta2real.infer \
  --checkpoint runs/gta5_sesekai2/checkpoints/selected.pt \
  --input /datasets/GTA5/held_out_images --output runs/predictions \
  --tile 1024 --overlap 128 --precision bf16
```

大图使用重叠 Hann 权重融合，无 resize。图块推理与整图推理的上下文不同，并非严格逐位一致。
`--strength 0.7` 可以降低输出强度。没有 selected 时明确改用 latest 路径。
目录输出保留相对路径，扩展名统一 PNG；重名冲突或覆盖输入会拒绝执行。

独立评估（原图与输出按相对路径对应，实拍集合仍不配对）：

```bash
python -m pip install -e '.[metrics]'
python -m gta2real.evaluate \
  --source /datasets/GTA5/held_out_images --generated runs/predictions \
  --real /datasets/SeSekai2/held_out_images \
  --teachers-config weights/teachers.local.yaml --semantic --fid \
  --output runs/evaluation.json
```

报告包含 DINO 内容距离、边缘、语义一致率与教师覆盖率；`--fid` 额外计算 FID/KID。
分布指标受视角和数据采样影响，不能替代内容核对或人工画质比较。
默认独立评估没有光流时序分数。视频稳定性须另用保留的完整片段检查。

## 8. 本地检查

```bash
python -m pytest -q
```

测试包含 odd-size/window mask、初始恒等、真实梯度、光流方向、配对 guide 几何、无配对采样、分组隔离、
worker/跨 epoch 数据恢复、真实优化器更新、检查点恢复和原尺寸推理。
`configs/smoke.yaml` 只检查接线：随机 VGG 特征没有有效的摄影外观先验，不可用于正式质量训练。
测试数据为程序生成的图案，不代表 GTA5/SeSekai2 的训练效果。
本地检查范围与未验证项记录在 [VALIDATION.json](VALIDATION.json)。

## 9. 方法参考

- [Uformer](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Uformer_A_General_U-Shaped_Transformer_for_Image_Restoration_CVPR_2022_paper.html)：多尺度窗口 Transformer。
- [Enhancing Photorealism Enhancement](https://isl-org.github.io/PhotorealismEnhancement/)：无配对游戏到实拍外观及内容分布匹配。
- [CUT](https://taesung.me/ContrastiveUnpairedTranslation/)：空间对应的对比约束。
- [DINOv2](https://github.com/facebookresearch/dinov2)、[SegFormer](https://huggingface.co/nvidia/segformer-b2-finetuned-cityscapes-1024-1024)：冻结监督网络。

这是自定义组合的训练实现，不是上述论文训练设置的逐项复刻，也未用真实数据验证 SOTA 性能。
