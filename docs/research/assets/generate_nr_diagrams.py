"""Generate source-grounded DLSS-NR diagrams using only the Python standard library."""

from html import escape
from pathlib import Path
import xml.etree.ElementTree as ET


DEST = Path(__file__).resolve().parent
INK = "#17324d"
MUTED = "#52677b"
BLUE = "#2d6eac"
TEAL = "#18847c"
ORANGE = "#c87916"


class Figure:
    def __init__(self, width, height, title, description):
        self.parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>
<defs>
<marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L9 4.5 L0 9 Z" fill="{BLUE}"/></marker>
<marker id="skip-arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L9 4.5 L0 9 Z" fill="{TEAL}"/></marker>
</defs>
<rect width="100%" height="100%" fill="#ffffff"/>
<style>text{{font-family:'Microsoft YaHei','微软雅黑',Arial,sans-serif}}.mono{{font-family:Consolas,'Microsoft YaHei','微软雅黑',monospace}}</style>''']

    def text(self, x, y, label, size=17, weight="normal", fill=INK, anchor="start", mono=False):
        cls = ' class="mono"' if mono else ""
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{cls}>{escape(str(label))}</text>')

    def rect(self, x, y, width, height, fill="#f3f7fb", stroke="#bfd0df", radius=12):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')

    def path(self, path, skip=False, width=2):
        color = TEAL if skip else BLUE
        dash = ' stroke-dasharray="7 6"' if skip else ""
        marker = "skip-arrow" if skip else "arrow"
        self.parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{width}"{dash} marker-end="url(#{marker})"/>')

    def node(self, x, y, width, height, title, subtitle, note=None, fill="#f1f6fb", stroke="#adc8e1"):
        self.rect(x, y, width, height, fill, stroke)
        if note:
            lines = [(title, 23, 18, "bold"), (subtitle, 46, 16, "normal"), (note, 65, 14, "normal")]
        else:
            lines = [(title, 30, 19, "bold"), (subtitle, 55, 16, "normal")]
        for text, dy, size, weight in lines:
            self.text(x + width/2, y+dy, text, size, weight, anchor="middle")

    def save(self, filename):
        content = "\n".join(self.parts + ["</svg>"])
        ET.fromstring(content)
        path = DEST / filename
        path.write_text(content, encoding="utf-8")
        print(f"XML OK: {path.name} ({len(content):,} characters)")


def architecture():
    f = Figure(1520, 1100, "DLSS 5 神经渲染：一帧数据如何穿过网络", "全部尺寸按高×宽×通道书写。依据版本310.8.0源码绘制。先构造16通道输入，再逐级下采样并执行局部Transformer。第六次下采样到8×12×1024后，执行8个全局ViT单元。B39先投影为512通道，再按16×20目标网格上采样。解码器融合编码侧细节，最后输出与输入同分辨率的图像。")
    f.text(76, 49, "DLSS 5 神经渲染：一帧数据如何穿过网络", 32, "bold")
    f.text(76, 79, "① 构造输入 → ② 局部编码 → ③ 全局交互 → ④ 解码融合 → ⑤ 同分辨率输出", 19, fill=MUTED)
    f.text(76, 108, "尺寸顺序：高×宽×通道。示例输入：512×512×3。局部窗口：8×8；每头 32 通道。尺寸包含源码对齐。版本：310.8.0。", 16, fill=MUTED)

    left, right, width = 80, 960, 480
    center_l, center_r = left+width/2, right+width/2
    rows = [242, 334, 426, 518, 610, 702, 794]
    h = 74
    f.node(left, 132, width, 74, "① 构造 16 通道输入，再投影 16 → 32", "原图 512×512×3 → 填充特征 512×576×16", "拼接当前图、噪声、历史图和控制条件；FP16 投影", fill="#f5f1fc", stroke="#c8b5e0")
    f.node(right, 132, width, 74, "⑤ 输出映射：32 → 4 通道（FP16）", "3 个 RGB 残差 + 1 个时间混合对数几率（FP32）", "与当前图和历史图合成，输出仍为 512×512", fill="#f5f1fc", stroke="#c8b5e0")
    f.text(760, 157, "沿左侧向下读，", 16, "bold", anchor="middle")
    f.text(760, 181, "底部全局交互后，转右侧向上读。", 15, fill=MUTED, anchor="middle")
    f.path(f"M{center_l} 206 V242")
    f.path(f"M{center_r} 242 V206")

    enc = [
        ("B0｜原分辨率局部 Transformer", "512×576×32｜1 个注意力头", "保留浅层细节，供最后一次融合"),
        ("B1–4｜局部 Transformer ×4", "L0：256×288×32｜1 个注意力头", "第一次下采样后，继续提取局部特征"),
        ("B5–8｜局部 Transformer ×4", "L1：128×144×64｜2 个注意力头", "第二次下采样后，继续提取局部特征"),
        ("B9–14｜局部 Transformer ×6", "L2：64×72×128｜4 个注意力头", "第三次下采样后，继续提取局部特征"),
        ("B15–22｜局部 Transformer ×8", "L3：32×36×256｜8 个注意力头", "第四次下采样后，继续提取局部特征"),
        ("B23–30｜局部 Transformer ×8", "L4：16×20×512｜16 个注意力头", "第五次下采样后，继续提取局部特征"),
    ]
    dec = [
        ("B70｜原分辨率局部 Transformer", "512×576×32｜1 个注意力头", "融合最浅层细节，预测逐像素输出"),
        ("B66–69｜局部 Transformer ×4", "L0：256×288×32｜1 个注意力头", "上采样并融合编码侧细节后，继续计算"),
        ("B62–65｜局部 Transformer ×4", "L1：128×144×64｜2 个注意力头", "上采样并融合编码侧细节后，继续计算"),
        ("B56–61｜局部 Transformer ×6", "L2：64×72×128｜4 个注意力头", "上采样并融合编码侧细节后，继续计算"),
        ("B48–55｜局部 Transformer ×8", "L3：32×36×256｜8 个注意力头", "上采样并融合编码侧细节后，继续计算"),
        ("B40–47｜局部 Transformer ×8", "L4：16×20×512｜16 个注意力头", "B39 完成投影、上采样与融合后，才进入此层"),
    ]
    for i, (a,b,c) in enumerate(enc):
        f.node(left, rows[i], width, h, a, b, c)
        f.node(right, rows[i], width, h, *dec[i], fill="#eff8f5", stroke="#acd1c6")
    f.node(left, rows[6], width, 90, "B31–38｜③ 全局 ViT ×8", "L5：8×12×1024｜32 个注意力头", "第六次下采样后，96 个空间位置全局交互", fill="#edf1fd", stroke="#a9b7dd")
    f.text(center_l, rows[6]+82, "96 个有效空间标记；存储补齐至 128 个", 13, fill=MUTED, anchor="middle")
    f.node(right, rows[6], width, 90, "B39｜全局结果转回解码器", "先投影：8×12×1024 → 8×12×512", "再按目标网格上采样：16×20×512", fill="#fff6e8", stroke="#e3bd80")
    f.text(center_r, rows[6]+82, "与编码器同尺度特征相加，再送入 B40–47", 13, fill=MUTED, anchor="middle")

    down = ["2×2 池化；通道不变", "池化；通道 32→64", "池化；通道 64→128", "池化；通道 128→256", "池化；通道 256→512", "池化；通道 512→1024"]
    for i in range(6):
        bottom = rows[i]+h
        f.path(f"M{center_l} {bottom} V{rows[i+1]}")
        f.text(center_l+16, bottom+13, down[i], 11, fill=MUTED)
    up = ["上采样；两路缩放后融合", "投影 64→32；上采样", "投影 128→64；上采样", "投影 256→128；上采样", "投影 512→256；上采样", "B39 融合结果向上送入"]
    for i in range(6):
        bottom = rows[i]+h
        f.path(f"M{center_r} {rows[i+1]} V{bottom}")
        f.text(center_r+16, bottom+13, up[i], 11, fill=MUTED)

    # The first five skips merge before the decoder block; B30 merges inside B39.
    for i in range(5):
        yy=rows[i]+h/2
        f.path(f"M560 {yy} H960", skip=True)
        f.text(760, yy-9, "编码侧细节：缩放后相加", 15, fill=TEAL, anchor="middle")
    f.path("M560 739 H725 V813 H960", skip=True)
    f.text(760, 729, "编码侧细节进入 B39 融合", 15, fill=TEAL, anchor="middle")
    f.path("M560 852 H960", width=2.5)
    f.text(760, 876, "先全局交互，再投影和上采样", 15, fill=BLUE, anchor="middle")
    f.text(320, 917, "② 编码：长宽变小，通道增多", 17, "bold", fill=BLUE, anchor="middle")
    f.text(1200, 917, "④ 解码：长宽变大，融合编码侧细节", 17, "bold", fill=TEAL, anchor="middle")

    f.rect(80, 943, 1360, 110, "#f7f9fc", "#d2dce6")
    f.text(102, 973, "71 个编号单元 = 70 个 Transformer 单元 + B39 过渡单元", 20, "bold")
    f.text(102, 1002, "主要矩阵与中间发布值使用 E4M3；累加使用 FP16。输入适配和输出映射使用 FP16。", 17, fill=MUTED)
    f.text(102, 1029, "六次池化后，每层长宽向上对齐至 4 的倍数。解码器按明确的目标网格采样。网络内部恢复尺寸，输出不放大。", 16, fill=MUTED)
    f.text(80, 1080, "源码证据：src/nr_graph.cpp:670–1000；docs/network.md；ports/browser-webgpu/src/graph.js。本图仅说明本仓库的神经渲染模型。", 13, fill=MUTED)
    f.save("nr-architecture.svg")


def block():
    f=Figure(1520, 1100, "神经渲染模型：一个 Transformer 单元如何计算", "先计算前馈网络，再融合逐通道学习缩放的残差。随后投影QKV，归一化Q和K，执行余弦注意力，再次融合缩放残差。橙色量化标记表示FP16和E4M3的舍入发布。下方列出各通道宽度的前馈变体及四种窗口相位。")
    f.text(60, 48, "一个 Transformer 单元如何计算", 32, "bold")
    f.text(60, 79, "① 前馈网络 → ② 缩放残差相加 → ③ QKV 投影与余弦注意力 → ④ 输出投影与残差相加", 18, fill=MUTED)

    y=207; cy=y+39
    f.node(60,y,125,78,"输入 x", "E4M3",fill="#f5f1fc",stroke="#c8b5e0")
    f.node(245,y,150,78,"前馈网络 FFN", "具体变体见下方")
    f.node(555,y,150,78,"QKV 投影", "通道 C → 3C")
    f.node(765,y,210,78,"余弦注意力", "每头维度 = 32")
    f.node(1035,y,145,78,"输出投影", "通道 C → C")
    f.node(1340,y,125,78,"输出", "E4M3",fill="#eff8f5",stroke="#acd1c6")

    for x in (475,1260):
        f.parts.append(f'<circle cx="{x}" cy="{cy}" r="21" fill="#ffffff" stroke="{BLUE}" stroke-width="2"/>')
        f.text(x,cy+7,"+",26,"bold",anchor="middle")
    for a,b in [(185,245),(395,454),(496,555),(705,765),(975,1035),(1180,1239),(1281,1340)]:
        f.path(f"M{a} {cy} H{b}")
    f.path(f"M210 {cy} V146 H475 V225")
    f.text(336,132,"x * s_ffn",17,"bold",anchor="middle")
    f.text(336,166,"逐通道学习缩放",13,fill=MUTED,anchor="middle")
    f.path(f"M510 {cy} V342 H1260 V267")
    f.text(875,329,"y × s_attn：C32 使用原始 FP16；更宽单元使用 E4M3",16,fill=MUTED,anchor="middle")
    f.text(475,302,"y",17,"bold",anchor="middle")
    for x in (525,1005):
        f.rect(x-21, cy-13, 42, 26, "#fff2d9", ORANGE, 8)
        f.text(x,cy+5,"量化",12,"bold",fill=ORANGE,anchor="middle")
    f.text(60,380,"量化：先舍入至 FP16，再舍入至 E4M3；两步均用最近偶数舍入。FFN 各层之间也发布量化结果。",16,fill=MUTED)

    # Detailed panels leave the residual path clear above them.
    f.rect(60,410,900,245,"#f3f7fb","#c0d2e3")
    f.text(82,442,"余弦注意力：先比较方向，再聚合内容",22,"bold")
    equations=[
        (475,"q_hat = tau_h * q / ||q||;   k_hat = k / ||k||",18),
        (505,"S = q_hat k_hat^T + B_h;   P = 近似Softmax(S);   A = P V",17),
    ]
    for yy,label,size in equations: f.text(82,yy,label,size,mono=True)
    f.text(82,538,"局部单元：窗口含 8×8 个位置；学习偏置 B_h[64,64]；窗口外零值仍影响归一化分母。",15)
    f.text(82,566,"全局单元：底部所有位置交互；没有 B_h；Q 额外乘 √32；聚合 V 后再除以分母。",15)
    f.text(82,594,"实现不减去最大分数。源码使用截断与位运算近似指数函数。",15)
    f.text(82,622,"公式说明计算结构。精确输出还取决于 FP16 与 E4M3 的舍入顺序。",15,fill=MUTED)

    f.rect(985,410,480,245,"#f0f8f5","#b7d6cb")
    f.text(1007,442,"四种窗口相位：逐块循环",22,"bold")
    offsets=["(0, 0)","(-4,-4)","(-4, 0)","(0,-4)"]
    for i,offset in enumerate(offsets):
        x=1007+i*111
        f.rect(x,466,98,78,"#ffffff","#b3d3c5",8)
        f.text(x+49,491,f"相位 {i}",14,"bold",anchor="middle")
        f.text(x+49,522,offset,15,anchor="middle",mono=True)
    f.text(1007,577,"括号内数值表示窗口原点偏移。",16)
    f.text(1007,604,"解码侧延续编码侧的相位计数。",16)
    f.text(1007,632,"每个分辨率层级单独计数。",15,fill=MUTED)

    f.rect(60,684,1405,277,"#fafbfd","#ced9e4")
    f.text(82,719,"FFN 随通道宽度改变结构：每条路径计算每个位置，没有稀疏专家路由",22,"bold")
    cards=[
        ("C32：稠密前馈",["32 → 128 → 32","扩展至 128 通道后计算 SiLU","2 个学习矩阵","全网共 10 个此类单元"]),
        ("C64 / C128 / C256",["并行路径数 E = C / 32","每条：C → 128 → 32","拼接后再投影：C → C","每条路径在 128 通道处计算 SiLU"]),
        ("C512：八条分支",["先投影 512 → 512，再拆成 8×64","每条：64 → 256 → 64","拼接后再投影：512 → 512","仅在 64 → 256 后计算 SiLU"]),
        ("C1024：全局 ViT",["1024 → 4096 → 1024","扩展至 4096 通道后计算 SiLU","稠密 FFN；32 个注意力头","底部共 8 个全局单元"]),
    ]
    for i,(title,lines) in enumerate(cards):
        x=82+i*344
        f.rect(x,741,323,191,"#ffffff","#c9d7e4",10)
        f.text(x+16,771,title,18,"bold")
        for j,line in enumerate(lines):
            f.text(x+16,805+j*31,line,14,mono=(j==0))

    f.text(60,1000,"计算精度属于模型定义",20,"bold")
    f.text(60,1028,"缩放残差先写入矩阵乘法累加器。若改成矩阵乘法结束后再相加，舍入结果会改变。",16,fill=MUTED)
    f.text(60,1054,"实现近似计算 SiLU 和指数函数。记录的前向图没有常规特征层归一化；Q、K 在注意力内部归一化。",16,fill=MUTED)
    f.text(60,1081,"源码证据：docs/network.md:86–174；docs/numerics.md；src/nr_graph.cpp:295–668；src/reference.cpp:84–116、215–290。",13,fill=MUTED)
    f.save("nr-block.svg")


if __name__ == "__main__":
    architecture()
    block()
