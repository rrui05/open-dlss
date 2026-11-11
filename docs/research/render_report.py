"""Render the local DLSS survey as a self-contained, offline-readable HTML file.

Requires Python-Markdown. MathJax 3.2.2 is cached under the OS temporary folder.
The report and its diagrams remain editable source artifacts.
"""

from __future__ import annotations

import base64
import html
import json
import mimetypes
from pathlib import Path
import re
import tempfile
import urllib.request
import xml.etree.ElementTree as ET

import markdown


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
STEM = "dlss-survey-2026-10-28"
PIN = "5cef97658735ef49daf048949e203d3f547699b8"


def make_pipeline() -> Path:
    """Draw task responsibilities; exact pass ordering is engine-dependent."""
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1480" height="820" '
        'viewBox="0 0 1480 820" role="img" aria-labelledby="title desc">',
        '<title id="title">DLSS 各功能的职责与接入关系</title>',
        '<desc id="desc">超分辨率或光线重建之后可以加入神经渲染。帧生成另需运动、深度、界面与显示同步。'
        '此图说明功能职责，具体执行顺序由引擎与插件契约决定。</desc>',
        '<defs><marker id="a" markerWidth="9" markerHeight="9" refX="8" refY="4.5" '
        'orient="auto"><path d="M0 0 L9 4.5 L0 9 Z" fill="#326e9b"/></marker></defs>',
        '<rect width="1480" height="820" fill="white"/>',
        '<style>text{font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif} '
        '.line{fill:none;stroke:#326e9b;stroke-width:2;marker-end:url(#a)}</style>',
        '<text x="60" y="58" font-size="31" fill="#17324d" font-weight="bold">'
        'DLSS 是一组职责不同的渲染技术</text>',
        '<text x="60" y="91" font-size="17" fill="#52677b">'
        '先区分各功能做什么，再按目标引擎的接口安排执行顺序</text>',
    ]

    def box(x, y, w, h, title, lines, fill="#edf5fc", border="#a6c3de"):
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" '
                     f'fill="{fill}" stroke="{border}" stroke-width="1.5"/>')
        parts.append(f'<text x="{x+20}" y="{y+32}" font-size="21" font-weight="bold" '
                     f'fill="#17324d">{html.escape(title)}</text>')
        for n, line in enumerate(lines):
            parts.append(f'<text x="{x+20}" y="{y+60+n*25}" font-size="17" '
                         f'fill="#435f78">{html.escape(line)}</text>')

    box(60, 145, 300, 148, "游戏引擎渲染", ["光栅化、光追或路径追踪", "输出颜色、运动与深度", "按功能提供材质引导"])
    box(450, 132, 365, 126, "超分辨率或抗锯齿", ["SR：低分辨率重建为高分辨率", "DLAA：原生分辨率抗锯齿"])
    box(450, 302, 365, 142, "光线重建 RR", ["读取噪声光追颜色与材质引导", "替代降噪并承担相关升采样", "此路径替代 SR，不能盲目重复叠加"])
    box(915, 212, 490, 154, "按需加入神经渲染 NR", ["保持图像尺寸，增强材质与光照外观", "读取当前颜色、历史状态与艺术控制", "输出图像，不修改底层场景几何"], "#f3effb", "#c7b4df")
    box(450, 542, 365, 130, "按需加入帧生成 FG / MFG", ["推断额外的显示帧", "使用后备缓冲、运动与深度", "按需提供无界面颜色与界面缓冲"], "#edf8f4", "#a2cebf")
    box(915, 542, 490, 130, "组合界面并送往显示", ["按功能接口组合界面", "管理交换链、显示节奏与 Reflex", "显示帧率与游戏模拟频率分开衡量"], "#edf8f4", "#a2cebf")
    parts += [
        '<path class="line" d="M360 192 H450"/>',
        '<path class="line" d="M360 235 H400 V365 H450"/>',
        '<path class="line" d="M815 195 H865 V260 H915"/>',
        '<path class="line" d="M815 365 H865 V315 H915"/>',
        '<path class="line" d="M1160 366 V480 H630 V542"/>',
        '<path class="line" d="M815 607 H915"/>',
        '<path class="line" d="M360 265 H390 V607 H450" stroke-dasharray="7 5"/>',
        '<text x="75" y="523" font-size="16" fill="#52677b">辅助输入：</text>',
        '<text x="75" y="548" font-size="16" fill="#52677b">运动、深度、界面</text>',
        '<text x="75" y="573" font-size="16" fill="#52677b">与逐帧相机常量</text>',
        '<rect x="60" y="720" width="1345" height="58" rx="10" fill="#fff6e8"/>',
        '<text x="80" y="754" font-size="18" fill="#805719">'
        '本仓库实现的是 NR 310.8.0。超分辨率、抗锯齿、光线重建与帧生成分别属于其他产品功能。</text>',
        '</svg>',
    ]
    target = ROOT / "assets" / "dlss-product-pipeline.svg"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(parts), encoding="utf-8")
    ET.parse(target)
    return target


def stash_math(text: str) -> tuple[str, list[str]]:
    fragments = []

    def replace(match):
        value = match.group(0)
        display = value.startswith("$$")
        inner = value[2:-2] if display else value[1:-1]
        wrapper = ('<div class="math-display">\\[' + html.escape(inner) + '\\]</div>') if display else (
            '<span class="math-inline">\\(' + html.escape(inner) + '\\)</span>')
        token = f"MATHSURVEYPLACEHOLDER{len(fragments)}END"
        fragments.append(wrapper)
        return "\n\n" + token + "\n\n" if display else token

    return re.sub(r"\$\$[\s\S]+?\$\$|(?<!\$)\$(?!\$)[^\n$]+?\$(?!\$)", replace, text), fragments


def data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def render():
    make_pipeline()
    source = (ROOT / f"{STEM}.md").read_text(encoding="utf-8")
    source, formulas = stash_math(source)
    converter = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "sane_lists"],
                                  extension_configs={"toc": {"toc_depth": "2-3"}})
    body = converter.convert(source)
    for index, fragment in enumerate(formulas):
        token = f"MATHSURVEYPLACEHOLDER{index}END"
        body = body.replace(f"<p>{token}</p>", fragment).replace(token, fragment)

    embedded = []

    def image_replace(match):
        original = html.unescape(match.group(1))
        path = (ROOT / original).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        embedded.append(str(path.relative_to(REPO)))
        return 'src="' + data_uri(path) + '"'

    body = re.sub(r'src="([^"]+)"', image_replace, body)

    def link_replace(match):
        value = html.unescape(match.group(1))
        if value.startswith(("http:", "https:", "#", "data:")):
            return match.group(0)
        pathpart, _, anchor = value.partition("#")
        path = (ROOT / pathpart).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_relative_to(ROOT):
            # Newly generated attachments are not in the fixed source commit.
            return 'href="' + data_uri(path) + '" download="' + html.escape(path.name) + '"'
        remote = f"https://github.com/rrui05/open-dlss/blob/{PIN}/{path.relative_to(REPO).as_posix()}"
        if anchor:
            remote += "#" + anchor
        return 'href="' + html.escape(remote) + '"'

    body = re.sub(r'href="([^"]+)"', link_replace, body)
    cache = Path(tempfile.gettempdir()) / "codex-dlss-survey-20261028"
    cache.mkdir(exist_ok=True)
    bundle = cache / "mathjax-3.2.2-tex-svg-full.js"
    if not bundle.exists():
        bundle.write_bytes(urllib.request.urlopen(
            "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg-full.js", timeout=60).read())
    mathjax = bundle.read_text(encoding="utf-8").replace("</script", "<\\/script")
    style = """
    :root{--ink:#1a3147;--muted:#5d7284;--blue:#236ea3;--line:#dce5ec}
    *{box-sizing:border-box}html{scroll-behavior:smooth}
    body{margin:0;background:#f1f5f8;color:var(--ink);font:16px/1.85 'Microsoft YaHei','Noto Sans CJK SC',sans-serif}
    nav{position:fixed;left:0;top:0;bottom:0;width:265px;background:#fff;border-right:1px solid var(--line);padding:28px 20px;overflow:auto;font-size:12px;line-height:1.6}
    nav .brand{font-size:19px;font-weight:700;margin:0 0 8px}nav .meta{color:var(--muted);margin-bottom:24px}
    nav ul{list-style:none;padding:0}nav li{margin:7px 0}nav ul ul{padding-left:11px;font-size:11px}
    main{margin:34px 32px 50px 297px;max-width:1120px;background:white;padding:52px 58px;box-shadow:0 8px 35px #16324d0b;border:1px solid var(--line);border-radius:12px}
    a{color:var(--blue);text-decoration:none}a:hover{text-decoration:underline}
    h1{font-size:34px;line-height:1.4;margin:0 0 24px;color:#123451}
    h2{font-size:25px;margin-top:52px;padding-top:18px;border-top:2px solid #d4e6f3;line-height:1.5}
    h3{font-size:19px;margin-top:32px;line-height:1.6}p{margin:14px 0}
    blockquote{margin:22px 0;padding:12px 22px;background:#f0f6fb;border-left:4px solid #3a8dbb;color:#4c657c;font-size:14px}
    table{border-collapse:collapse;width:100%;margin:22px 0;font-size:13px;line-height:1.65}
    th,td{border:1px solid #dce5ec;padding:9px 11px;text-align:left;vertical-align:top;overflow-wrap:anywhere}
    th{background:#eaf2f8;font-weight:700}tr:nth-child(even) td{background:#fafcfd}
    code{font:0.88em/1.6 Consolas,'Courier New',monospace;background:#f0f3f6;padding:2px 5px;border-radius:4px;overflow-wrap:anywhere}
    pre{padding:18px 20px;background:#f3f6f9;border:1px solid #dce5ec;overflow:auto;border-radius:8px;font-size:13px;line-height:1.6}
    pre code{background:none;padding:0}img{display:block;width:100%;height:auto;margin:25px auto;border:1px solid #e2e9ee;border-radius:6px}
    li{margin:5px 0}.math-display{overflow-x:auto;padding:8px 0;font-size:100%}
    .footer{margin-top:50px;padding-top:20px;border-top:1px solid var(--line);color:var(--muted);font-size:12px}
    @media(max-width:1000px){nav{position:static;width:auto;max-height:260px}main{margin:15px;padding:26px}h1{font-size:28px}}
    @page{size:A4;margin:16mm 14mm 18mm}
    @media print{
      html{scroll-behavior:auto}body{background:white;font-size:10pt;line-height:1.65;color:#182d40}
      nav{display:none}main{margin:0;padding:0;border:0;box-shadow:none;max-width:none;border-radius:0}
      h1{font-size:24pt}h2{font-size:17pt;margin-top:25pt;padding-top:10pt;break-after:avoid}
      h3{font-size:12pt;margin-top:17pt;break-after:avoid}p{orphans:3;widows:3}
      table{font-size:8.1pt;line-height:1.55;margin:12pt 0}th,td{padding:5pt 6pt}
      thead{display:table-header-group}tr{break-inside:avoid}img{break-inside:avoid;margin:14pt auto;max-height:225mm;object-fit:contain}
      pre{white-space:pre-wrap;font-size:8pt;line-height:1.5;break-inside:avoid;padding:10pt}
      a{color:#23638e}blockquote{font-size:9pt;break-inside:avoid}.math-display{overflow:visible;break-inside:avoid;font-size:9pt}
      *{-webkit-print-color-adjust:exact;print-color-adjust:exact}
    }
    """
    configuration = """
    window.MathJax={tex:{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']]},
      svg:{fontCache:'local'},options:{enableMenu:false},startup:{ready:()=>{
        MathJax.startup.defaultReady();MathJax.startup.promise.then(()=>{
          document.documentElement.dataset.mathReady='true';
          document.documentElement.dataset.mathErrors=document.querySelectorAll('mjx-merror').length;
        });}}};
    """
    document = f"""<!doctype html>
    <html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>DLSS 模型架构与产品调研报告 · 2026-10-28</title>
    <style>{style}</style><script>{configuration}</script><script>{mathjax}</script></head>
    <body><nav><div class="brand">DLSS 研究报告</div><div class="meta">源码架构 · 产品接入 · 证据边界<br>2026-10-28 · 310.8.0 / 当前产品</div>{converter.toc}</nav>
    <main>{body}<div class="footer">基于 open-dlss 固定提交与 NVIDIA 官方资料。图与公式已嵌入，正文可离线阅读。外部引用链接联网打开。</div></main>
    </body></html>"""
    output = ROOT / f"{STEM}.html"
    output.write_text(document, encoding="utf-8")
    result = {"output":str(output),"html_bytes":output.stat().st_size,
              "math_fragments":len(formulas),"embedded_images":embedded,
              "external_scripts":len(re.findall(r'<script[^>]+src=', document)),
              "source_characters":len((ROOT/f'{STEM}.md').read_text(encoding='utf-8'))}
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    render()
