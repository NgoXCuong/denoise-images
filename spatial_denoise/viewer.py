"""Khung xem ảnh với mức zoom và vùng quan sát dùng chung."""

import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk


# Tính tỷ lệ và độ lệch từ tọa độ ảnh sang khung xem.
# Căn giữa ảnh vừa khung và giới hạn vị trí kéo để không lộ nền khi ảnh lớn hơn khung.
def image_transform(image_size, viewport_size, scale, center):
    iw, ih = image_size
    vw, vh = viewport_size
    zoom = min(vw / iw, vh / ih) if scale is None else scale
    offsets = []
    for extent, available, position in zip((iw, ih), (vw, vh), center):
        length = extent * zoom
        offset = ((available - length) / 2 if length <= available else
                  min(0, max(available - length, available / 2 - position * length)))
        offsets.append(offset)
    return zoom, *offsets


class SharedView:
    # Khởi tạo tỷ lệ, tâm quan sát và danh sách khung ảnh dùng chung.
    def __init__(self, master):
        self.scale = None
        self.center = (0.5, 0.5)
        self.cards = []

    # Vẽ lại các khung ảnh đang hiển thị theo tỷ lệ và tâm quan sát dùng chung.
    def redraw(self):
        for card in self.cards:
            if card.winfo_viewable():
                card.draw()

    # Đặt ảnh vừa khung, đưa tâm về giữa; có thể hoãn vẽ khi đang thay dữ liệu ảnh.
    def fit(self, *, redraw=True):
        self.scale, self.center = None, (0.5, 0.5)
        if redraw:
            self.redraw()

    # Đặt tỷ lệ 100%, mỗi điểm ảnh tương ứng một điểm trên canvas.
    def actual_size(self):
        self.scale = 1.0
        self.redraw()

    # Thay đổi mức zoom trong khoảng 5%–800% và cập nhật các khung ảnh đồng bộ.
    # Khi có vị trí con trỏ, giữ điểm ảnh dưới con trỏ ổn định, kể cả từ chế độ vừa khung.
    def zoom(self, factor, card=None, point=None):
        if card is None:
            card = next((c for c in self.cards if c.array is not None and c.winfo_viewable()), None)
        if card is None or card.array is None:
            return
        scale, ox, oy = card.transform()
        new_scale = min(8.0, max(0.05, scale * factor))
        if point is not None:
            vw, vh = card.viewport_size()
            iw, ih = card.image_size()
            px, py = (point[0] - ox) / scale, (point[1] - oy) / scale
            self.center = (min(1, max(0, (px + (vw / 2 - point[0]) / new_scale) / iw)),
                           min(1, max(0, (py + (vh / 2 - point[1]) / new_scale) / ih)))
        self.scale = new_scale
        self.redraw()


class ImageCard(ttk.Frame):
    # Tạo khung ảnh, tiêu đề và mô tả; đăng ký sự kiện zoom, kéo và đổi kích thước.
    def __init__(self, master, title, view):
        super().__init__(master, style="Card.TFrame", padding=6)
        self.array = None
        self.photo = None
        self.pending = None
        self.empty_text = "Chưa có kết quả\nChọn ảnh rồi bấm xử lý ở bên trái."
        self.view = view
        self.view.cards.append(self)
        self.heading = ttk.Label(self, text=title, style="CardTitle.TLabel", wraplength=200)
        self.heading.pack(anchor="w", pady=(0, 4))
        self.canvas = tk.Canvas(self, width=1, height=1, bg="#e9eeec",
                                bd=0, highlightthickness=0, cursor="fleur")
        self.canvas.pack(fill="both", expand=True)
        self.detail = ttk.Label(self, text=" ", style="CardMuted.TLabel", wraplength=200)
        self.detail.pack(anchor="w", pady=(4, 0))
        self.canvas.bind("<Configure>", self._resize)
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Button-4>", lambda event: self._wheel(event, 1.25))
        self.canvas.bind("<Button-5>", lambda event: self._wheel(event, 0.8))
        self.canvas.bind("<ButtonPress-1>", self._pan_start)
        self.canvas.bind("<B1-Motion>", self._pan)
        self.canvas.bind("<Double-Button-1>", lambda _event: self.view.actual_size())
        self.bind("<Configure>", self._wrap_labels)

    # Điều chỉnh độ rộng xuống dòng của tiêu đề và mô tả theo kích thước khung.
    def _wrap_labels(self, event):
        width = max(80, event.width - 24)
        self.heading.configure(wraplength=width)
        self.detail.configure(wraplength=width)

    # Cập nhật tiêu đề của khung ảnh.
    def set_title(self, title):
        self.heading.configure(text=title)

    # Gán dữ liệu ảnh và nội dung khi trống, sau đó vẽ lại khung.
    def set_array(self, array, empty_text="Chưa có kết quả"):
        self.array, self.empty_text = array, empty_text
        self.draw()

    # Lấy kích thước canvas hiện tại, bảo đảm mỗi chiều tối thiểu một điểm.
    def viewport_size(self):
        return max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())

    # Lấy chiều rộng và chiều cao của mảng ảnh đang hiển thị.
    def image_size(self):
        return self.array.shape[1], self.array.shape[0]

    # Tính tỷ lệ và độ lệch hiện tại của ảnh từ trạng thái vùng xem dùng chung.
    def transform(self):
        return image_transform(self.image_size(), self.viewport_size(), self.view.scale, self.view.center)

    # Hủy lịch vẽ cũ và hẹn vẽ lại sau 40 ms để giảm số lần vẽ khi đổi kích thước.
    def _resize(self, _event=None):
        if self.pending:
            self.after_cancel(self.pending)
        self.pending = self.after(40, self._scheduled_draw)

    # Xóa trạng thái lịch hẹn và thực hiện lần vẽ đã lên lịch.
    def _scheduled_draw(self):
        self.pending = None
        self.draw()

    # Vẽ ảnh hoặc thông báo trống trong canvas theo mức zoom và vị trí quan sát.
    # Chỉ dựng phần ảnh đang nhìn thấy để hạn chế bộ nhớ khi phóng to ảnh lớn.
    def draw(self):
        if self.pending:
            self.after_cancel(self.pending)
            self.pending = None
        self.canvas.delete("all")
        vw, vh = self.viewport_size()
        if self.array is None:
            self.photo = None
            self.canvas.create_text(vw / 2, vh / 2, text=self.empty_text,
                                    width=max(1, vw - 24), justify="center", fill="#63758a")
            return
        if vw < 2 or vh < 2:
            return
        scale, ox, oy = self.transform()
        iw, ih = self.image_size()
        left, top = max(0.0, -ox / scale), max(0.0, -oy / scale)
        right, bottom = min(iw, (vw - ox) / scale), min(ih, (vh - oy) / scale)
        size = (max(1, round((right - left) * scale)), max(1, round((bottom - top) * scale)))
        picture = Image.fromarray(self.array).resize(
            size, Image.Resampling.NEAREST if scale >= 1 else Image.Resampling.LANCZOS,
            box=(left, top, right, bottom))
        self.photo = ImageTk.PhotoImage(picture)
        self.canvas.create_image(max(0, ox), max(0, oy), anchor="nw", image=self.photo)

    # Đổi sự kiện lăn chuột thành zoom quanh con trỏ và chặn cuộn tiếp ra vùng ngoài.
    def _wheel(self, event, factor=None):
        self.view.zoom(factor or (1.25 if event.delta > 0 else 0.8), self, (event.x, event.y))
        return "break"

    # Ghi lại vị trí chuột và phép biến đổi ảnh khi bắt đầu kéo.
    def _pan_start(self, event):
        if self.array is None:
            return
        scale, ox, oy = self.transform()
        self.drag_origin = (event.x, event.y, scale, ox, oy)

    # Cập nhật tâm quan sát theo độ dịch chuyển chuột và vẽ lại các ảnh đồng bộ.
    # Khi kéo từ chế độ vừa khung, cố định tỷ lệ hiện tại để các khung dùng chung mức zoom.
    def _pan(self, event):
        if self.array is None or not hasattr(self, "drag_origin"):
            return
        x, y, scale, ox, oy = self.drag_origin
        vw, vh = self.viewport_size()
        iw, ih = self.image_size()
        self.view.scale = scale
        self.view.center = (min(1, max(0, (vw / 2 - ox - event.x + x) / (iw * scale))),
                            min(1, max(0, (vh / 2 - oy - event.y + y) / (ih * scale))))
        self.view.redraw()
