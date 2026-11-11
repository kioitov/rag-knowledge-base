"""生成魔搭创空间封面图。

要求（页面提示）：jpg/png，高度大于 220px 或宽度大于 400px。
这里做 1200×800，能适配各种展示位。
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 800
OUT = Path(r"E:\deepseek work\rag-knowledge-base\deploy_pkg\cover.png")

# 深色科技感背景
img = Image.new("RGB", (W, H), (13, 17, 23))
draw = ImageDraw.Draw(img)

# 渐变背景（从深蓝到近黑）
for y in range(H):
    t = y / H
    r = int(13 + 12 * (1 - t))
    g = int(22 + 28 * (1 - t))
    b = int(38 + 52 * (1 - t))
    draw.line([(0, y), (W, y)], fill=(r, g, b))

# 装饰：右下角半透明圆弧
overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
od = ImageDraw.Draw(overlay)
od.ellipse([W - 420, H - 380, W + 180, H + 220], fill=(56, 132, 255, 26))
od.ellipse([W - 260, H - 260, W + 60, H + 60], fill=(110, 86, 207, 30))
img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
draw = ImageDraw.Draw(img)

# 字体
FONT_DIR = "C:/Windows/Fonts/"
f_title = ImageFont.truetype(FONT_DIR + "msyhbd.ttc", 76)
f_sub = ImageFont.truetype(FONT_DIR + "msyh.ttc", 34)
f_small = ImageFont.truetype(FONT_DIR + "msyh.ttc", 26)
f_tag = ImageFont.truetype(FONT_DIR + "msyh.ttc", 24)

# 主标题
draw.text((88, 200), "知识库问答", font=f_title, fill=(255, 255, 255))
draw.text((88, 300), "RAG 检索增强生成演示", font=f_sub, fill=(150, 190, 255))

# 分隔线
draw.line([(88, 372), (420, 372)], fill=(56, 132, 255), width=4)

# 说明文字
lines = [
    "语料：Kubernetes 中文文档 63 篇 · 54.9 万字",
    "切块 1290 段 · BM25 + 向量混合检索",
    "回答带来源定位，资料不足时明确拒答",
]
y = 410
for line in lines:
    draw.text((88, y), line, font=f_small, fill=(185, 195, 210))
    y += 48

# 底部标签
tags = ["Streamlit", "Chroma", "qwen-plus", "CC BY 4.0"]
x = 88
ty = 640
for tag in tags:
    bbox = draw.textbbox((0, 0), tag, font=f_tag)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    pad_x, pad_y = 20, 12
    draw.rounded_rectangle(
        [x, ty, x + tw + pad_x * 2, ty + th + pad_y * 2],
        radius=8, fill=(30, 41, 59), outline=(70, 90, 120), width=1,
    )
    draw.text((x + pad_x, ty + pad_y - 2), tag, font=f_tag, fill=(160, 180, 205))
    x += tw + pad_x * 2 + 16

# 右上角小标识
draw.text((W - 300, 60), "RAG · 可溯源问答", font=f_tag, fill=(100, 120, 145))

OUT.parent.mkdir(parents=True, exist_ok=True)
img.save(OUT, "PNG", optimize=True)

print(f"已生成 {OUT}")
print(f"尺寸 {W}x{H}，{OUT.stat().st_size / 1024:.0f} KB")
