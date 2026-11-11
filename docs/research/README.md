# DLSS 架构与产品调研

资料核查日期：2026 年 10 月 28 日。源码范围：本仓库的 DLSS 5 NR 310.8.0。

建议从 [第 2.0 节的课堂式讲解](dlss-survey-2026-10-28.html#classroom-walkthrough) 开始。它按一张图像的完整流动路线解释尺寸、底部 ViT 与解码过程。三张架构图的自然语言标注统一为中文。

| 文件 | 用途 |
| --- | --- |
| [完整报告 HTML](dlss-survey-2026-10-28.html) | 推荐阅读。带目录，内嵌公式与图片，正文可以离线打开 |
| [完整报告 PDF](dlss-survey-2026-10-28.pdf) | 打印、归档与分享 |
| [报告 Markdown](dlss-survey-2026-10-28.md) | 可编辑正文与引用 |
| [整体网络 SVG](assets/nr-architecture.svg) | 精确单元编号、通道、张量尺寸与跳连 |
| [单元结构 SVG](assets/nr-block.svg) | FFN、cosine attention、窗口相位与数值发布 |
| [产品管线 SVG](assets/dlss-product-pipeline.svg) | SR、RR、NR 与 FG/MFG 的职责关系 |
| [验证记录 JSON](validation.json) | 文件、公式、PDF 与保存工件的检查范围 |

报告区分源码事实、NVIDIA 官方说明、分析推导与仓库既有实验。它不把静态重复性或 CPU 数值检查当成 GPU 全网原生一致性证明。

重跑已有工件的只读核查：

```powershell
python docs/research/verify_survey_evidence.py
```

重建图和 HTML：

```powershell
python docs/research/assets/generate_nr_diagrams.py
python docs/research/render_report.py
```

HTML 生成脚本使用 Python-Markdown。它将 MathJax 3.2.2 缓存到系统临时目录，再嵌入输出。首次重建需要网络；生成后的 HTML 正文与图无需网络。

PDF 使用 Chrome headless 从该 HTML 打印。源码和参考数据均未修改。重跑全网 parity 仍依赖匹配版本的模型与 native fixtures。
