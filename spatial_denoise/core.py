from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# Kiểm tra ảnh đầu vào là mảng uint8 xám hoặc RGB
def check_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if (image.ndim != 2 and not (image.ndim == 3 and image.shape[2] == 3)) or not image.size or image.dtype != np.uint8:
        raise ValueError("Ảnh phải là mảng uint8 xám hoặc RGB, không rỗng.")
    return image


# Kiểm tra cửa sổ lọc là số nguyên lẻ từ 3 đến 31; trả về kích thước hợp lệ.
def check_kernel(size: int) -> int:
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)):
        raise ValueError("Kích thước cửa sổ phải là số nguyên.")
    if size < 3 or size > 31 or size % 2 == 0:
        raise ValueError("Kích thước cửa sổ phải lẻ, từ 3 đến 31.")
    return int(size)


# Chuyển giá trị sang số thực và kiểm tra hữu hạn, nằm trong khoảng cho phép.
def _finite(value: float, name: str, lower: float, upper: float) -> float:
    value = float(value)
    if not np.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{name} phải nằm trong [{lower:g}, {upper:g}].")
    return value


@dataclass(frozen=True)
class RunConfig:
    noise: str = "gaussian"
    sigma: float = 20.0
    amount: float = 0.05
    kernel: int = 3
    seed: int = 42
    edge_threshold: float = 20.0
    input_mode: str = "simulate"

    # Kiểm tra chế độ xử lý, tham số nhiễu, cửa sổ, seed và ngưỡng Sobel trước khi chạy.
    def validate(self) -> None:
        if self.input_mode not in ("simulate", "existing"):
            raise ValueError("Chế độ xử lý không hợp lệ.")
        if self.noise not in ("gaussian", "salt_pepper"):
            raise ValueError("Loại nhiễu không hợp lệ.")
        _finite(self.sigma, "Sigma", 0, 100)
        _finite(self.amount, "Tỷ lệ nhiễu", 0, 1)
        _finite(self.edge_threshold, "Ngưỡng biên", 0.01, 255)
        check_kernel(self.kernel)
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)):
            raise ValueError("Seed phải là số nguyên không âm.")
        if not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("Seed phải nằm trong [0, 4294967295].")


# Thêm nhiễu Gaussian hoặc muối tiêu vào bản sao ảnh sạch.
def add_noise(image: np.ndarray, config: RunConfig) -> np.ndarray:
    image = check_image(image)
    config.validate()
    if config.input_mode != "simulate":
        raise ValueError("Chế độ ảnh đã nhiễu không bổ sung nhiễu.")
    rng = np.random.default_rng(config.seed)
    if config.noise == "gaussian":
        values = image.astype(np.float64) + rng.normal(0, config.sigma, image.shape)
        return np.rint(np.clip(values, 0, 255)).astype(np.uint8)
    uniform = rng.random(image.shape[:2])
    result = image.copy()
    result[uniform < config.amount / 2] = 0
    result[uniform >= 1 - config.amount / 2] = 255
    return result


# Lọc trung bình bằng ảnh tích phân với cửa sổ size × size và đệm biên lặp.
# Ảnh RGB được lọc riêng từng kênh; kết quả giữ kích thước và kiểu uint8.
def mean_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
    image = check_image(image)
    size = check_kernel(size)
    if image.ndim == 3:
        return np.stack([mean_filter(image[..., channel], size) for channel in range(3)], axis=-1)
    padded = np.pad(image, size // 2, mode="edge").astype(np.float64)
    integral = np.pad(padded.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
    total = (integral[size:, size:] - integral[:-size, size:]
             - integral[size:, :-size] + integral[:-size, :-size])
    return np.rint(total / (size * size)).clip(0, 255).astype(np.uint8)


# Lọc trung vị trên cửa sổ size × size với đệm biên lặp, xử lý riêng từng kênh RGB.
# Chia cửa sổ trượt thành các dải hàng để hạn chế bộ nhớ tạm; trả về ảnh uint8.
def median_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
    image = check_image(image)
    size = check_kernel(size)
    if image.ndim == 3:
        return np.stack([median_filter(image[..., channel], size) for channel in range(3)], axis=-1)
    padded = np.pad(image, size // 2, mode="edge")
    windows = sliding_window_view(padded, (size, size))
    result = np.empty_like(image)
    rows = max(1, min(64, 8_000_000 // (image.shape[1] * size * size)))
    for start in range(0, image.shape[0], rows):
        result[start:start + rows] = np.median(
            windows[start:start + rows], axis=(-2, -1)
        ).astype(np.uint8)
    return result


# Tính độ lớn gradient Sobel 3×3; ảnh RGB được đổi sang độ sáng để tính biên.
# Đệm biên lặp, chia Gx và Gy cho 8; trả về hypot(Gx, Gy) kiểu float64 chưa ngưỡng hóa.
def sobel(image: np.ndarray) -> np.ndarray:
    image = check_image(image)
    if image.ndim == 3:
        image = image.astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    p = np.pad(image.astype(np.float64), 1, mode="edge")
    gx = (-p[:-2, :-2] + p[:-2, 2:] - 2 * p[1:-1, :-2]
          + 2 * p[1:-1, 2:] - p[2:, :-2] + p[2:, 2:]) / 8.0
    gy = (-p[:-2, :-2] - 2 * p[:-2, 1:-1] - p[:-2, 2:]
          + p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) / 8.0
    return np.hypot(gx, gy)


# Tính sai số bình phương trung bình sai lệch giữa hai mảng cùng kích thước, hữu hạn.
def mse(reference: np.ndarray, image: np.ndarray) -> float:
    a, b = np.asarray(reference), np.asarray(image)
    if a.shape != b.shape or not a.size:
        raise ValueError("Hai ảnh phải cùng kích thước và không rỗng.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Ảnh không được chứa NaN hoặc vô cực.")
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


# Tính tỷ số tín hiệu đỉnh trên nhiễu PSNR từ MSE theo miền giá trị ảnh.
# Trả về vô cực khi MSE bằng 0; PSNR càng cao càng gần ảnh chuẩn theo sai số điểm ảnh.
def psnr(reference: np.ndarray, image: np.ndarray, data_range: float = 255) -> float:
    if not np.isfinite(data_range) or data_range <= 0:
        raise ValueError("Miền giá trị ảnh phải dương và hữu hạn.")
    error = mse(reference, image)
    return _psnr_from_mse(error, data_range)


# Tính PSNR từ MSE đã có để tránh tính lại sai số trên toàn bộ ảnh.
def _psnr_from_mse(error: float, data_range: float = 255) -> float:
    return float("inf") if error == 0 else float(10 * np.log10(data_range**2 / error))


# Dùng cùng ngưỡng để tạo hai bản đồ biên và tính F1 bằng đối sánh đúng tọa độ.
# Nếu cả hai bản đồ không có biên, quy ước F1 bằng 1.
def edge_f1(reference_gradient: np.ndarray, gradient: np.ndarray, threshold: float) -> float:
    threshold = _finite(threshold, "Ngưỡng biên", 0.01, 255)
    if reference_gradient.shape != gradient.shape:
        raise ValueError("Hai bản đồ gradient phải cùng kích thước.")
    truth, predicted = reference_gradient >= threshold, gradient >= threshold
    denominator = int(truth.sum()) + int(predicted.sum())
    if denominator == 0:
        return 1.0
    return float(2 * np.count_nonzero(truth & predicted) / denominator)


@dataclass
class RunResult:
    config: RunConfig
    images: dict[str, np.ndarray]
    gradients: dict[str, np.ndarray]
    metrics: list[dict]
    reference_source: str | None = None

    # Cho biết kết quả có ảnh sạch làm chuẩn để tính các chỉ số tham chiếu hay không.
    @property
    def has_reference(self) -> bool:
        return "original" in self.images

def run_pipeline(original: np.ndarray, config: RunConfig, *,
                 reference: np.ndarray | None = None, reference_source: str | None = None) -> RunResult:
    original = check_image(original)
    config.validate()
    if config.input_mode == "simulate":
        if reference is not None:
            raise ValueError("Ảnh đối chiếu riêng chỉ dùng trong chế độ ảnh đã nhiễu.")
        reference = original
        noisy = add_noise(original, config)
    else:
        noisy = original.copy()
        if reference is not None:
            reference = check_image(reference)
            if reference.shape != noisy.shape:
                raise ValueError("Ảnh sạch đối chiếu phải cùng kích thước với ảnh nhiễu.")
    images = {"noisy": noisy}
    if reference is not None:
        images = {"original": reference.copy(), **images}
    elapsed = {"noisy": 0.0}
    for name, function in (("mean", mean_filter), ("median", median_filter)):
        start = perf_counter()
        images[name] = function(noisy, config.kernel)
        elapsed[name] = (perf_counter() - start) * 1000
    gradients = {name: sobel(img) for name, img in images.items()}
    rows = []
    for name in ("noisy", "mean", "median"):
        error = mse(reference, images[name]) if reference is not None else None
        rows.append({
            "method": name, "kernel": 0 if name == "noisy" else config.kernel,
            "mse": error,
            "psnr_db": _psnr_from_mse(error) if error is not None else None,
            "sobel_mse": mse(gradients["original"], gradients[name]) if reference is not None else None,
            "edge_f1": (edge_f1(gradients["original"], gradients[name], config.edge_threshold)
                        if reference is not None else None),
            "filter_ms": elapsed[name],
        })
    return RunResult(config, images, gradients, rows, reference_source if reference is not None else None)
