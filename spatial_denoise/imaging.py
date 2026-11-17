"""Đọc ảnh, tạo ảnh mẫu và chuẩn bị ảnh Sobel để hiển thị."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

# Đọc ảnh 8 bit, chỉnh hướng EXIF và ghép vùng trong suốt lên nền trắng.
# Giữ độ phân giải; trả về bản sao mảng uint8 xám hoặc RGB, từ chối ảnh 16 bit/HDR.
def load_image(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode in ("I", "F") or image.mode.startswith("I;16"):
            raise ValueError("Hãy chuyển ảnh 16 bit/HDR sang ảnh 8 bit trước khi sử dụng.")
        if image.mode in ("RGBA", "LA") or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", image.size, "white")
            image = Image.alpha_composite(background, rgba)
        mode = "L" if image.mode in ("1", "L", "LA") else "RGB"
        return np.asarray(image.convert(mode), dtype=np.uint8).copy()


# Tạo ảnh xám mẫu có nền chuyển sắc, sóng hoặc ô bàn cờ theo variant.
# Vẽ hình học, đường mảnh và chữ để quan sát mức giảm nhiễu và mất chi tiết khi lọc.
def demo_image(size: int = 512, variant: int = 0) -> np.ndarray:
    y, x = np.mgrid[:size, :size]
    if variant == 1:
        values = 100 + 45 * np.sin(x / 5) + 30 * np.cos(y / 11)
    elif variant == 2:
        values = 45 + 160 * ((x // max(4, size // 24) + y // max(4, size // 24)) % 2)
    else:
        values = 35 + 150 * x / max(1, size - 1) + 15 * np.sin(y / 18)
    image = Image.fromarray(np.clip(values, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    s = size / 512
    box = lambda coords: tuple(round(v * s) for v in coords)
    draw.rectangle(box((40, 55, 220, 220)), fill=225, outline=10, width=max(1, round(3 * s)))
    draw.ellipse(box((285, 60, 465, 240)), fill=55, outline=245, width=max(1, round(4 * s)))
    draw.polygon([tuple(round(c * s) for c in p) for p in ((75, 420), (175, 265), (250, 420))], fill=245)
    for i in range(12):
        yy = 290 + i * 10
        draw.line(box((290, yy, 465, yy)), fill=235 if i % 2 else 15, width=max(1, round(2 * s)))
    try:
        font = ImageFont.truetype("arial.ttf", max(12, round(25 * s)))
    except OSError:
        font = ImageFont.load_default()
    draw.text((round(40 * s), round(458 * s)), "SPATIAL FILTERS  012345", fill=5, font=font)
    return np.asarray(image).copy()


# Nhân độ lớn gradient với 2, giới hạn về 0–255 và đổi sang uint8 để hiển thị.
# Phép chuyển đổi này không thay đổi gradient gốc dùng để tính chỉ số.
def gradient_display(gradient: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(gradient * 2, 0, 255)).astype(np.uint8)
