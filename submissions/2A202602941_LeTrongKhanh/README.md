# Lab Day 2 — Lê Trọng Khanh — 2A202602941

Bài nộp DeepWeeds fold 0, chạy thực tế trên Kaggle/Tesla T4. Cấu hình cuối F01 là ConvNeXt-Tiny + công thức T11 + suy luận ba scale 224/256/288 + temperature scaling khớp trên val.

| Chỉ số test (3 seed, std mẫu) | F01 | T00 + 1-view |
|---|---|---|
| Accuracy | 0.983081 ± 0.000718 | 0.978994 ± 0.000329 |
| Macro-F1 | 0.978534 ± 0.000615 | 0.973326 ± 0.000561 |
| ECE | 0.006258 ± 0.000733 | 0.008927 ± 0.001139 |

Δ macro-F1 = +0.005208. ECE F01 trước TS là 0.004140, thấp hơn sau TS: hiệu chuẩn không giúp ECE test. Phần I tự chấm **18/20**, không phải điểm tổng toàn bài.

## Sản phẩm

- [results.xlsx](results.xlsx): 7 sheet theo GUIDE, top 10 theo val, mean/std chung kết, so sánh ECE.
- [report.md](report.md): thiết lập, ablation, suy luận, chung kết, phân tích lỗi và hạn chế.
- [code/lab_day2.ipynb](code/lab_day2.ipynb): notebook đã chạy, giữ nguyên output làm bằng chứng.
- `code/`: dataset, model, losses, train, inference, benchmark và kiểm tra bài nộp.
- `curves/`: 26 ảnh training B/T/F; `predictions/`: 44 CSV, gồm test của T00, F01, F01uncal cho seed 0/1/2.
- `figures/`: EDA, pipeline, tradeoff, confusion và ảnh lỗi.
- `logs/`: JSON đã tải từ Kaggle, cấu hình đầy đủ và history CSV trích từ stdout notebook cho 26 run.
- `eval_out/`: chỉ số tính lại bằng `eval.py`, grade và validation/hash CSV.

**URL notebook Kaggle đã chạy:** chưa được cung cấp. Link notebook cục bộ ở trên có toàn bộ code và output; cần bổ sung URL chia sẻ thật trước khi nộp nếu giảng viên yêu cầu link Kaggle.

## Chạy lại notebook

Thực hiện trong repo bài lab gốc có `eval.py`, `README.md`, `GUIDE.md`, `RUBRIC.md`, `starter/` và thư mục bài nộp này. Notebook là phiên bản Kaggle; bước setup tìm repo trong `/kaggle/input`, không cần đổi đường dẫn Windows.

1. Nén code repo thành `lab_day2_code.zip`, giữ thư mục `submissions/2A202602941_LeTrongKhanh/code/` và `eval.py` gốc trong repo; không kèm ảnh dataset, checkpoint hoặc output lớn. Add Input chứa ZIP/repo vào Kaggle.
2. Import `code/lab_day2.ipynb` vào Kaggle, bật Internet và GPU. Phiên chạy ghi nhận Tesla T4, Python **3.13.15**, torch **2.11.0+cu128**, timm **1.0.29**. Thư viện phụ dùng numpy, pandas, torchvision, Pillow, matplotlib, openpyxl; version chính xác của chúng chưa được ghi trong output. Nếu cần tái lập chặt hơn, lưu `pip freeze` ở lần chạy mới.
3. Dùng thư mục output mới cho lần tái lập; chạy setup → tải dữ liệu/checksum → EDA/pipeline → 5 backbone → training ablation/3 seed → suy luận val → chốt final plan → test → xuất sản phẩm.
4. Tập ảnh là DeepWeeds nguyên bản; dùng fold 0 tải từ tác giả. Giữ nguyên train/val/test; không gộp val vào train. Seed 0/1/2 chỉ thay khởi tạo, thứ tự batch, augmentation.
5. Notebook khóa test theo từng run bằng `test_started.json`; không xóa khóa để chạy lại và chọn điểm tốt hơn. Lưu cả output, checkpoint/logit, `temperature.json` và version thư viện khi cần tái lập đúng nhiệt độ từng seed.

Trong bài nộp hiện tại không kèm checkpoint và file nhiệt độ từng seed. Cách tính T nằm trong notebook, `logs/final_plan.json` ghi cấu hình đã chốt. Suy luận scale trung bình softmax các view rồi lấy log xác suất làm đầu vào TS. Val/test của F01 trong `predictions/` là bản chung kết ba scale + TS; `noise.json` là 1-view checkpoint khi so sánh công thức.

## Kiểm tra kết quả đã tải (không cần GPU)

Từ thư mục gốc repo, cần Python, numpy, pandas; các CSV nhãn gốc nằm ở `data/labels/`.

```powershell
python submissions/2A202602941_LeTrongKhanh/code/verify_submission.py
python -m unittest discover -s tests
```

Script đối chiếu 44 file với split gốc, kiểm tra softmax/argmax, log metric, ảnh curves, giao/hợp tập và ghi `eval_out/validation.json`. Script chỉ đọc dự đoán đã có; không chạy inference trên test. Log history được trích từ các dòng epoch của notebook; không tạo số liệu mới.

```powershell
python eval.py score --pred "submissions/2A202602941_LeTrongKhanh/predictions/F01_seed*_test.csv" --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F01 --out submissions/2A202602941_LeTrongKhanh/eval_out
python eval.py score --pred "submissions/2A202602941_LeTrongKhanh/predictions/T00_seed*_test.csv" --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag T00 --out submissions/2A202602941_LeTrongKhanh/eval_out
python eval.py grade --final "submissions/2A202602941_LeTrongKhanh/predictions/F01_seed*_test.csv" --baseline "submissions/2A202602941_LeTrongKhanh/predictions/T00_seed*_test.csv" --uncal "submissions/2A202602941_LeTrongKhanh/predictions/F01uncal_seed*_test.csv" --final-val "submissions/2A202602941_LeTrongKhanh/predictions/F01_seed*_val.csv" --test-csv data/labels/test_subset0.csv --val-csv data/labels/val_subset0.csv --labels data/labels/labels.csv --latency-p95-ms 22.093431649 --latency-method proper --out submissions/2A202602941_LeTrongKhanh/eval_out
```

## Kiểm tra trước khi nộp

- [x] Đủ bảng Excel, báo cáo, code, curves, predictions và README.
- [x] 5 backbone, 7 trục training, 9 cấu hình inference; 3 seed cho mốc và chung kết.
- [x] 44 CSV đúng định dạng, khớp đủ 3.501 ảnh val hoặc 3.507 ảnh test theo fold 0.
- [x] 26 curves và 26 history, cấu hình/tag pretrained có trong JSON; notebook giữ output thật.
- [x] Mean/std dùng ddof=1; số liệu chung kết khớp `eval.py`; tự chấm phần I 18/20.
- [x] 38 test của repo pass; kiểm tra focal γ=0, CutMix và BN trong notebook có output.
- [x] p95 scale 22,09 ms trên T4 batch 1, warmup 10, 100 lượt, synchronize; chưa tính PIL/normalize/H2D.
- [x] Nêu kết quả âm TS, giới hạn 1 fold, ít seed ablation, pretrained tag khác nhau và rủi ro lệch miền.
- [ ] Bổ sung URL chia sẻ notebook Kaggle chính xác.

Có một nhãn train khác `labels.csv` trong nguồn tác giả (`20170714-110407-3.jpg`); giữ nguyên split, giải thích trong báo cáo và ghi trong validation. Không sửa `eval.py` gốc, không đưa ảnh dataset/checkpoint vào git.
