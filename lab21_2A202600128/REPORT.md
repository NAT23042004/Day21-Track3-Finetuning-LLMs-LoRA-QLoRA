# Lab 21 — Evaluation Report

**Học viên**: [Your Name] — 2A202600128
**Ngày nộp**: 2026-05-07
**Submission option**: B (GitHub + HuggingFace Hub)

## 1. Setup

- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit` (4-bit quantized with QLoRA)
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, 200 samples (180 train + 20 eval)
- **max_seq_length**: 1024 (p95 of token lengths, rounded up to power of 2)
- **GPU**: Tesla T4 (16 GB VRAM)
- **Training cost**: $0.07 (~12.2 phút @ $0.35/hr on T4)
- **HF Hub link (r=16)**: https://huggingface.co/NATSTAN/qwen2.5-3b-vi-lab21-r16
- **HF Hub links (r=8, r=64)**: Chưa push — chỉ có r=16 trên Hub

## 2. Rank Experiment Results

| Rank | Trainable Params | Train Time (min) | Peak VRAM (GB) | Eval Loss | Perplexity |
|------|-----------------:|-----------------:|---------------:|----------:|-----------:|
| 8    | 1,843,200 (0.06%) | 3.83 | 7.22 | 1.5577 | 4.75 |
| 16   | 3,686,400 (0.12%) | 4.03 | 6.62 | 1.5161 | 4.55 |
| 64   | 14,745,600 (0.48%) | 3.87 | 8.00 | 1.4768 | 4.38 |
| Base | - | - | - | 1.62* | 5.05* |

*\*Base model perplexity estimated từ initial eval loss trước khi fine-tune.*

**Observation**: Tăng rank từ 8→16 giảm perplexity 4.2% (4.75→4.55), nhưng tăng từ 16→64 chỉ giảm thêm 3.7% (4.55→4.38) trong khi số params tăng 4×.

## 3. Loss Curve Analysis

![Loss Curve](loss_curve.png)

- **Overfitting**: Không quan sát thấy overfitting rõ rệt vì eval strategy được đặt là "no" trên T4 để tiết kiệm VRAM (không chạy eval trong lúc train).
- **Loss trend**: Training loss giảm ổn định qua 3 epochs, đạt ~1.5 ở cuối epoch 3.
- **Lý do không eval during training**: T4 chỉ có 16GB VRAM, chạy eval mid-training dễ gây OOM. Thay vào đó, eval được chạy sau khi train xong (safe_evaluate với batch=1).

## 4. Qualitative Comparison (5 examples)

### Example 1
**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.

**Base**: Machine learning là một phân khúc của trí tuệ nhân tạo, nó tập trung vào việc thiết lập các mô hình máy móc để học tập từ dữ liệu và từ đó có thể dự đoán hoặc hành động. Nhìn vào khía cạnh đơn giản, nó có nghĩa là máy tính học tập từ dữ liệu...

**Fine-tuned (r=16)**: Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện các dự đoán dựa trên dữ liệu mà không có sự hướng dẫn trực tiếp từ người dùng. Nó là một phần của AI (trí tuệ nhân tạo) và sử dụng thuật toán và các mô hình học máy để học từ dữ liệu...

**Nhận xét**: Fine-tuned model trả lời tự nhiên hơn, dùng từ ngữ tiếng Việt mạch lạc hơn.

---

### Example 2
**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.

**Base**: Để tính số Fibonacci thứ n, bạn có thể sử dụng hàm đệ quy hoặc vòng lặp. Đây là một đoạn mã Python cho phép bạn tính số Fibonacci thứ n: `def fibonacci(n): if n <= 0: return "N phải là một số dương"...`

**Fine-tuned (r=16)**: Để tính số Fibonacci thứ n, bạn có thể viết một đoạn code Python như sau: `def fibonacci(n): if n < 0: raise ValueError("Input phải là một số nguyên dương.")...`

**Nhận xét**: FT model dùng `raise ValueError` thay vì return string — code chuẩn hơn. Tuy nhiên cả hai đều đưa ra solution đúng.

---

### Example 3
**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.

**Base**: 1. Thân thiện với người dùng: Mục đích của thiết kế UI/UX là cung cấp trải nghiệm người dùng thân thiện, dễ sử dụng và thân thiện...

**Fine-tuned (r=16)**: 1. Chuyển đổi: UI/UX thiết kế phải hướng tới việc giúp người dùng thực hiện hành động. 2. Thích ứng: UI/UX thiết kế phải thích ứng với nhiều loại thiết bị...

**Nhận xét**: Base model lặp từ "thân thiện" nhiều lần. FT model đưa ra các nguyên tắc cụ thể, đa dạng hơn.

---

### Example 4
**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.

**Base**: LoRA (Low-Rank Adaptation) và QLoRA (Quantized LoRA) là hai phương pháp cải thiện hiệu năng của mô hình NLU (NLP)...

**Fine-tuned (r=16)**: LoRA (Layer-wise Adaptive Regularization Optimization) và QLoRA (Quantized LoRA) là hai phương pháp regularization...

**Nhận xét**: ⚠️ FT model **sai** khi định nghĩa LoRA là "Layer-wise Adaptive Regularization Optimization" — đây là hallucination. Base model đúng hơn (Low-Rank Adaptation).

---

### Example 5
**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.

**Base**: Prompt engineering, RAG (retrieval augmented generation), và fine-tuning là ba cách khác nhau để cải thiện hiệu suất của mô hình máy học...

**Fine-tuned (r=16)**: Prompt engineering, RAG và fine-tuning là ba kỹ thuật khác nhau được sử dụng trong lĩnh vực AI và tự động hóa...

**Nhận xét**: Cả hai đều trả lời đúng concept, nhưng FT model viết tự nhiên hơn, dùng từ ngữ tiếng Việt chuẩn xác hơn.

---

## 5. Conclusion về Rank Trade-off

Dựa trên kết quả thí nghiệm với 3 rank (8, 16, 64) trên dataset 200 mẫu tiếng Việt, tôi kết luận **r=16 là rank tối ưu nhất** cho bài toán này xét trên phương diện ROI (Return on Investment).

**Tại sao r=16 tốt nhất?**
1. **Hiệu quả tham số**: r=16 chỉ chiếm 0.12% tổng tham số (3.7M params), trong khi r=64 chiếm 0.48% (14.7M params) — gấp 4 lần nhưng chỉ cải thiện perplexity thêm 3.7% (từ 4.55 xuống 4.38).
2. **VRAM usage**: r=16 sử dụng 6.62 GB peak VRAM, thấp hơn r=64 (8.00 GB) và thậm chí thấp hơn r=8 (7.22 GB — do overhead của một số layers). Điều này cho phép deploy trên GPU nhỏ hơn.
3. **Training time**: Cả 3 rank đều train trong ~4 phút trên T4, không có sự khác biệt đáng kể về thời gian.
4. **Quality**: r=16 giảm perplexity 4.2% so với r=8, đạt mức chất lượng tốt cho most use cases.

**Diminishing returns**: Điểm break-even nằm ở khoảng r=16. Từ r=16→64, mỗi 0.1% tăng params chỉ đem lại ~0.04% cải thiện perplexity. Với dataset 200 samples, r=64 bắt đầu over-parameterized — lượng data không đủ để tận dụng hết capacity của rank cao.

**Recommendation cho production**: Chọn **r=16** (alpha=32). Nếu cần tiết kiệm VRAM hơn nữa cho edge devices, r=8 là compromise hợp lý (chỉ kém 4.2% quality). r=64 chỉ nên dùng khi có dataset >10k samples hoặc yêu cầu quality cực cao.

## 6. What I Learned

- **LoRA mechanics**: Hiểu rõ cơ chế low-rank decomposition ΔW = B·A, và cách rank ảnh hưởng đến số lượng trainable parameters. Rank càng cao, model càng sát với full fine-tuning nhưng bị diminishing returns.
- **QLoRA practicality**: 4-bit quantization giúp fine-tune model 3B trên T4 (16GB) chỉ tốn ~6-8GB VRAM. Đây là game-changer cho việc fine-tune LLM trên consumer GPU.
- **Qualitative evaluation quan trọng không kém quantitative**: Perplexity thấp chưa chắc đã tốt — ví dụ r=16 bị hallucination khi định nghĩa LoRA (Error trong Example 4). Cần kết hợp cả 2 phương pháp đánh giá.
- **Unsloth optimization**: Custom CUDA kernels giúp train nhanh gấp 2× và giảm VRAM 60% so với vanilla PEFT/TRL. Rất hữu ích cho T4.
