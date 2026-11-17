# Khử nhiễu ảnh bằng bộ lọc trong miền không gian

Ứng dụng Python có giao diện tiếng Việt: tự cài đặt bộ lọc trung bình, trung vị và Sobel; bổ sung nhiễu Gaussian/muối tiêu; đánh giá MSE, PSNR, MSE Sobel và F1 biên. Thử nghiệm từng cấu hình trực tiếp trên giao diện.

## Khởi tạo và chạy trên Windows

Yêu cầu: **Python 3.10 trở lên**, có Tcl/Tk (bản Python Windows thông thường đã có). Mở PowerShell tại thư mục dự án:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Không cần activate môi trường ảo, không cần thay đổi ExecutionPolicy. Khi đã cài thư viện, có thể nhấp đúp **run.bat** để mở chương trình.

Nếu máy đang có sẵn NumPy và Pillow:

```powershell
python app.py
```

Kiểm tra Tkinter bằng `python -m tkinter`. Nếu không mở được cửa sổ, bổ sung thành phần Tcl/Tk trong trình cài Python. Chương trình không cần OpenCV, SciPy, kết nối mạng hoặc GPU để chạy.

## Chạy thử nhanh

1. Mở ứng dụng: ảnh mẫu tự tạo được xử lý tự động với nhiễu muối tiêu 5%, cửa sổ 3×3, seed 42.
2. Chọn **Mở ảnh…** để đưa ảnh sạch vào làm chuẩn; giữ ảnh màu RGB hoặc ảnh xám 8 bit và kích thước gốc.
3. Chọn loại/mức nhiễu, cửa sổ, seed và ngưỡng Sobel. Bấm **TẠO NHIỄU & SO SÁNH**.
4. Quan sát đồng thời bốn ảnh trong tab **01 Kết quả lọc**: ảnh gốc, ảnh nhiễu, lọc trung bình và lọc trung vị. MSE/PSNR hiển thị ngay dưới các ảnh kết quả. Lăn chuột để zoom, kéo ảnh để di chuyển; các khung dùng chung mức zoom và vùng xem.
5. Bấm **Lưu ảnh kết quả…** để lưu riêng ảnh nhiễu, ảnh lọc trung bình hoặc ảnh lọc trung vị. Xem Sobel trong tab **02 Bảo toàn biên · Sobel**, bảng MSE/PSNR trong **03 Đánh giá · MSE / PSNR**. Dùng con lăn để zoom, kéo ảnh để di chuyển để quan sát chi tiết. Bấm **F11** để xem toàn màn hình, **Esc** để quay lại.
6. Để so sánh nhiều cấu hình, lần lượt thay đổi mức nhiễu, cửa sổ hoặc seed rồi chạy lại; ghi nhận các chỉ số trong tab Đánh giá.

Các số thập phân dùng dấu chấm: `0.5`. Tỷ lệ muối tiêu trong giao diện nhập theo **phần trăm**, ví dụ `5` nghĩa là 5%.

## Khử nhiễu ảnh đã bị nhiễu

1. Tại **CHẾ ĐỘ XỬ LÝ**, chọn **Khử nhiễu ảnh có sẵn**.
2. Bấm **Mở ảnh nhiễu…**, chọn ảnh cần xử lý. Giữ ảnh màu RGB hoặc ảnh xám 8 bit.
3. Chọn cửa sổ lọc và bấm **KHỬ NHIỄU & SO SÁNH**. Cả hai bộ lọc xử lý trực tiếp cùng ảnh đầu vào, không thêm nhiễu.
4. Nếu có ảnh sạch cùng nội dung, kích thước và vị trí, trong cột bên trái, bấm **Chọn ảnh sạch đối chiếu…** rồi chạy lại để tính MSE, PSNR, MSE Sobel và F1.
5. Nếu không có ảnh sạch, lựa chọn ảnh chuẩn được ẩn, bốn chỉ số trong bảng hiện `—`; vẫn xem được ảnh, Sobel, thời gian lọc và lưu ảnh kết quả.

Mở ảnh đầu vào mới hoặc đổi chế độ sẽ bỏ ảnh đối chiếu và kết quả cũ. Giao diện lưu riêng ảnh nhiễu hoặc ảnh sau lọc dưới dạng PNG.

## Thực nghiệm trên giao diện

Mở ảnh sạch, chọn loại và mức nhiễu, cửa sổ và seed rồi bấm **TẠO NHIỄU & SO SÁNH**. Đọc MSE, PSNR, MSE Sobel và F1 trong tab Đánh giá; quan sát tab Sobel để so sánh khả năng giữ biên. Giữ nguyên seed và mức nhiễu khi so sánh các cửa sổ trên cùng ảnh.

Các số liệu, ảnh và báo cáo thực nghiệm đã tạo trong `outputs/` được giữ để đối chiếu. Không còn lệnh chạy thực nghiệm hàng loạt trong ứng dụng.

Ảnh 16 bit/HDR cần được chuẩn bị thành ảnh 8 bit trước khi mở.

## Các tệp chính

```text
app.py                          Điểm khởi chạy giao diện
run.bat                         Khởi chạy nhanh trên Windows
requirements.txt                NumPy và Pillow
spatial_denoise/
    core.py                     Nhiễu, bộ lọc, Sobel, chỉ số, pipeline
    imaging.py                  Đọc ảnh, tạo ảnh mẫu và hiển thị Sobel
    gui.py                      Giao diện Tkinter, xử lý nền
    viewer.py                   Zoom, kéo ảnh và vùng xem đồng bộ
docs/
    TAI_LIEU_DE_TAI.md           Lý thuyết, công thức, giả mã, thiết kế thí nghiệm
    HUONG_DAN_SU_DUNG.md         Hướng dẫn thao tác, cấu hình và đầu ra
tests/                          Kiểm thử thuật toán, đọc/lưu ảnh và giao diện
samples/                        Ba ảnh tổng hợp 512×512 có sẵn để mở thử
```

## Quy ước và phạm vi

- Giữ màu RGB hoặc ảnh xám uint8 0–255, không giảm độ phân giải ngầm. Ảnh màu được lọc riêng từng kênh; MSE/PSNR tính trên toàn bộ kênh. Sobel dùng độ sáng (0.299 R + 0.587 G + 0.114 B). Ảnh trong suốt được ghép trên nền trắng.
- Hai bộ lọc đều dùng viền lặp (`edge`); cửa sổ lẻ từ 3 đến 31. Giao diện đơn ảnh cung cấp 3, 5, 7.
- Sobel 3×3 được chia hệ số 8; cùng ngưỡng cho mọi ảnh. Hình Sobel hiển thị nhân 2 trên thang cố định; chỉ số tính từ gradient thực, chưa cắt ngưỡng.
- MSE/PSNR so với ảnh sạch gốc. Không dùng ảnh nhiễu làm chuẩn và không diễn giải các số này như đánh giá chất lượng khi không có ảnh sạch.
- F1 biên so với Sobel ảnh sạch, khớp điểm ảnh chính xác; đây là chỉ số phụ thuộc ngưỡng, không phải đánh giá theo nhãn biên thủ công.
- Seed và ngưỡng Sobel nằm trong cột thiết lập bên trái; nút xử lý/lưu luôn ở cuối cột bên trái.
- Cấu hình đang hiển thị ở dòng **KẾT QUẢ** là cấu hình đã thực sự chạy. Đổi trường nhập chưa làm thay đổi kết quả; cần bấm chạy lại. Lưu ảnh kết quả dùng lần xử lý đã chạy.
- Ảnh mẫu giúp kiểm tra thuật toán; cần thêm ảnh chụp thực tế trước khi kết luận cho báo cáo.

## Kiểm thử

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Kiểm thử gồm oracle vòng lặp cho bộ lọc, hành vi viền ảnh, nhiễu tái lập, phép tính chỉ số, Sobel trên ảnh biết trước, đọc ảnh, lưu PNG và luồng giao diện. Không cần cài thư viện kiểm thử riêng.

Tài liệu đầy đủ: [Tài liệu đề tài](docs/TAI_LIEU_DE_TAI.md), [Hướng dẫn sử dụng](docs/HUONG_DAN_SU_DUNG.md).

Kết quả đã chạy: [Biên bản kiểm thử và số liệu mẫu](docs/KET_QUA_KIEM_THU.md). Báo cáo thực nghiệm xuất cả đồ thị PSNR PNG/SVG và bảng HTML có thể mở không cần mạng.
