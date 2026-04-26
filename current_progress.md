\[Kang, Athens\] 1\. Thonburian Whisper Distilled  (Thai Audio \-\> Eng Text, Thai Audio \-\> Thai   
Text)

## **CPU Summary (FP32 vs INT8)**

| Report | Samples | FP32 CER | INT8 CER | Delta CER | FP32 RTF-T | INT8 RTF-T | Speedup | FP32 BLEU | INT8 BLEU | H1 CER\<5% | H2 BLEU drop\<3 | H3 RTF\<0.5 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| progress-001 (medium-combined, 100\) | 100 | 0.103 | 0.100 | \-0.003 | 10.856 | 1.241 | 8.75x | 3.15 | 5.99 | pass | pass | fail |
| progress-002 (medium-combined, full) | 500 | 0.205 | 0.083 | \-0.123 | 11.009 | 1.248 | 8.82x | \- | \- | fail | pending | fail |
| progress-003 (distill-medium, full) | 500 | 0.098 | 0.096 | \-0.001 | 6.146 | 0.640 | 9.61x | \- | \- | pass | pending | fail |

RTF-T means transcribe RTF mean. Speedup is FP32 RTF divided by INT8 RTF.

## **GPU Summary (vs CPU INT8 Baselines)**

| Run | CPU RTF-T | GPU RTF-T | Speedup | CPU CER | GPU CER | Delta CER | GPU BLEU | GPU chrF++ | H\_GPU1 Speed | H\_GPU2 CER±1% |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| medium-combined GPU FP16 (run\_009) | 1.248 | 0.210 | 5.95x | 0.083 | 0.085 | \+0.003 | \- | \- | pass | pass |
| medium-combined GPU INT8 (run\_010) | 1.248 | 0.267 | 4.68x | 0.083 | 0.088 | \+0.005 | \- | \- | fail | pass |
| distill-medium GPU FP16 (run\_011) | 0.640 | 0.070 | 9.08x | 0.096 | 0.094 | \-0.002 | \- | \- | pass | pass |
| distill-medium GPU INT8 (run\_012) | 0.640 | 0.080 | 8.00x | 0.096 | 0.096 | \-0.000 | \- | \- | pass | pass |

### **NLLB-600M Translation Benchmark (Thai to English)**

- GPU (T4)

| โมเดล (Model) | จำนวนข้อมูล (Samples) | คุณภาพการแปล (BLEU Score) | Latency เฉลี่ย (วินาที/ประโยค) | ความเร็วการแปล (WPM) |
| :---- | :---- | :---- | :---- | :---- |
| **NLLB-600M (FP32)** | 500 | 17.87 | 0.099 | 13,034.76 |
| **NLLB-600M (INT8)** | 500 | 17.81 | 0.054 | 23,931.42 |

- FP32

Source (TH) : โกลเวอร์พูดในการแถลงข่าวที่มีญาติของเหยื่อมาประมาณ 20 คน  
Prediction  : Glover spoke in a press release with about 20 relatives of the victims.  
Reference   : Glover spoke at a news conference that included about 20 relatives of the victims.  
Sentence BLEU Score: 45.75

- INT8

Source (TH) : โกลเวอร์พูดในการแถลงข่าวที่มีญาติของเหยื่อมาประมาณ 20 คน  
Prediction  : Glover spoke in a press release with about 20 relatives of the victims.  
Reference   : Glover spoke at a news conference that included about 20 relatives of the victims.  
Sentence BLEU Score: 45.75

- CPU

| โมเดล (Model) | จำนวนข้อมูล (Samples) | คุณภาพการแปล (BLEU Score) | Latency เฉลี่ย (วินาที/ประโยค) | ความเร็วการแปล (WPM) |
| :---- | :---- | :---- | :---- | :---- |
| **NLLB-600M (FP32)** | 500 | 16.35 | 3.217 | 401.24 |
| **NLLB-600M (INT8)** | 500 | 16.00 | 1.155 | 1117.80 |

### **3.3 TTS — Text-to-Speech** 

**3.3.1 Model: Piper (en\_US-lessac-medium), Engine: ONNX**

**Core Performance Metrics**

| Metric Category | Metric Name | Measured Result (Avg) |
| :---- | :---- | :---- |
| Speed | RTF (Real-Time Factor) | 0.1859 |
|  | Inference Latency | 1.0038 s |
|  | Throughput (CPS) | 102.32 chars/sec |
| Resource | Model Load Time | 2.7510 s |
|  | RAM Impact | 37.55 MB  |
|  | CPU Utilization | 62.8 % (Colab vCPU) |

**Consistency and Stability Analysis (RTF vs. Sentence Length)**

To ensure that the performance of the TTS system remains predictable under various workloads, we conducted a Consistency Test comparing the efficiency of the model across different input lengths, ranging from short phrases to complex paragraphs.

| Sentence Type | Character Length | Inference Latency (s) | Real-Time Factor (RTF) | Audio Duration (s) |
| :---- | :---- | :---- | :---- | :---- |
| Shortest Sentence | 25 chars | 0.2584 s | 0.1568 | 1.65 s |
| Longest Sentence | 168 chars | 1.4744 s | 0.1817 | 8.12 s |
| Overall Average | 102 chars | 1.0038 s | 0.1859 | 5.40 s |

To ensure statistical reliability and fulfill advanced evaluation requirements, the TTS naturalness was validated against the human ground-truth LJSpeech dataset. We employed a comprehensive evaluation strategy using both 'Cross-Model Validation' via OpenAI Whisper (ASR) for intelligibility, and Full-Reference metrics for acoustic fidelity.

**1\. Intelligibility and Naturalness (Large-Scale Validation)** A large-scale validation across 100 diverse sentences was conducted to determine how accurately the AI-generated speech reflects natural human pronunciation.

* Total Sample Size: 100 sentences  
* Average Word Error Rate (WER): 0.1419  
* Accuracy Rate: 85.81%  
* Best Case WER: 0.0000 (Perfect Intelligibility)  
* Worst Case WER: 1.0000

**2\. Advanced Acoustic Metrics (PESQ & PSNR)** To measure the signal-level distortion between the generated audio and the human reference, we calculated the Perceptual Evaluation of Speech Quality (PESQ) and Peak Signal-to-Noise Ratio (PSNR).

* PESQ Score: 1.0482  
* PSNR: 14.57 dB

**3.3.2 Model: Kokoro-82M (PyTorch)**

**Core Performance Metrics**

| Metric Category | Metric Name | Measured Result (Avg) |
| :---- | :---- | :---- |
| Speed | RTF (Real-Time Factor) | 1.4555 |
|  | Inference Latency | 9.2086 s |
|  | Throughput (CPS) | 10.75 chars/sec |
| Resource | Model Load Time | 2.9184 s |
|  | RAM Impact | 288.62 MB |
|  | CPU Utilization | 62.4 % (Colab vCPU) |

**Consistency and Stability Analysis (RTF vs. Sentence Length)**

**1\. Intelligibility and Naturalness (Large-Scale Validation)** A large-scale validation across 100 diverse sentences was conducted to determine how accurately the AI-generated speech reflects natural human pronunciation.

* Total Sample Size: 100 sentences  
* Average Word Error Rate (WER): 0.0979  
* Accuracy Rate: 90.21%  
* Best Case WER: 0.0000 (Perfect Intelligibility)  
* Worst Case WER: 1.0000

**2\. Advanced Acoustic Metrics (PESQ & PSNR)** To measure the signal-level distortion between the generated audio and the human reference, we calculated the Perceptual Evaluation of Speech Quality (PESQ) and Peak Signal-to-Noise Ratio (PSNR).

* PESQ Score: 1.0985  
* PSNR: 16.88 dB

**3.3.3 Model: Kokoro-82M (Experimental INT8 Quantized)**

**Core Performance Metrics**

| Metric Category | Metric Name | Measured Result (Avg) |
| :---- | :---- | :---- |
| Speed | RTF (Real-Time Factor) | 1.4282 |
|  | Inference Latency | 9.0739 s |
|  | Throughput (CPS) | 10.93 chars/sec |
| Resource | Model Load Time | 3.1710 s |
|  | RAM Impact | 36.52 MB |
|  | CPU Utilization | 62.7 % |

**Speech Quality and Advanced Audio Metrics (Checking Quantization Drop)**

* **Average Word Error Rate (WER):** 0.0972 (Accuracy: 90.28%)  
* **PESQ Score:** 1.1037  
* **PSNR:** 16.76 dB

\[Nun\] 4\. Typhoon ASR Real Time (Thai Audio \-\> Thai Text)  
**Model**: typhoon-ai/typhoon-asr-realtime  
**Engine**: NeMo (FastConformer-Transducer)

| Metric Category | Metric Name | Measured Result |
| :---- | :---- | :---- |
| Speed | RTF (Real-Time Factor) | 0.0200 (50× real-time) |
|  | RTF P90 | 0.0256 |
|  | Inference Latency (avg) | 0.2248 s |
|  | Audio Duration (avg) | 11.87 s |
| Accuracy | CER (Character Error Rate) | 0.1591 (84.1% accuracy) |
| Resource | Real-Time Capable | ✓ Yes |

**Benchmark Results:**  
Total Sample Size: 100 sentences (FLEURS th\_th test set)  
Average Inference Latency: 0.2248 s  
Average Audio Duration: 11.87 s  
Worst Case RTF: 0.0368  
Best Case RTF: 0.0142  
Corpus CER: 0.1591

**INT8 Dynamic Quantization** (run\_002)  
Method: torch.quantization.quantize\_dynamic (Linear layers → qint8)  
Dataset: google/fleurs th\_th test (1021 sentences)

| Metric | FP32 (Baseline) | INT8 (Quantized) | Change |
| :---- | :---- | :---- | :---- |
| WER | 0.2998 | 0.2982 | \-0.5% |
| RTF Mean | 0.0225 | 0.0215 | \-4.6% |
| RTF P90 | 0.0292 | 0.0280 | \-3.9% |
| Real-Time | ✓ Yes | ✓ Yes | — |

**Hypothesis Results:**  
H3 (WER degradation \< 5%): SUPPORTED ✓ — only \-0.5% relative WER increase  
H4 (RTF improvement): SUPPORTED ✓ — \-4.6% RTF change

INT8 quantization preserves accuracy with minimal WER degradation. The FastConformer-Transducer bottleneck is the RNN-T autoregressive decoding loop, not Linear layer compute.

**CUDA FP32 GPU Acceleration** (run\_003)  
Device: NVIDIA RTX 5060 Ti 16GB (Blackwell, compute 12.0)  
Dataset: google/fleurs th\_th test (1021 sentences)

| Metric | CPU FP32 | CUDA FP32 | Change |
| :---- | :---- | :---- | :---- |
| WER | 0.2998 | 0.2999 | \+0.0% |
| RTF Mean | 0.0225 | 0.0063 | \+71.8% faster |
| RTF P90 | 0.0292 | 0.0093 | \+67.9% faster |
| Real-Time | ✓ Yes | ✓ Yes | — |

**Hypothesis Results:**  
H5 (CUDA RTF \< CPU RTF): SUPPORTED ✓ — \+71.8% RTF improvement on GPU  
H6 (CUDA WER matches CPU \< 1%): SUPPORTED ✓ — \+0.0% relative WER difference

CUDA delivers 72% RTF speedup vs CPU (RTF 0.006 vs 0.023). FastConformer Transformer encoder benefits from GPU parallelism. WER is essentially identical across CPU and GPU.

**CUDA FP16 Half Precision** (run\_004)  
Method: model.half() — converts all parameters to float16 on CUDA  
Dataset: google/fleurs th\_th test (1021 sentences)

| Metric | CUDA FP32 | CUDA FP16 | Change |
| :---- | :---- | :---- | :---- |
| WER | 0.2999 | 0.2999 | \+0.1% vs FP32 |
| RTF Mean | 0.0063 | 0.0217 | \+242.1% |
| RTF P90 | 0.0093 | 0.0604 | \+546.0% |
| Real-Time | ✓ Yes | ✓ Yes | — |

**Hypothesis Results:**  
H7 (FP16 RTF \< FP32 RTF on GPU): REFUTED ✗ — \+242.1% RTF change vs CUDA FP32  
H8 (FP16 WER matches FP32 \< 1%): SUPPORTED ✓ — \+0.1% relative WER difference

Naive model.half() FP16 conversion is \+242% slower than FP32 on GPU (RTF 0.022 vs 0.006). The NeMo RNN-T autoregressive decoding loop suffers from FP16↔FP32 type-casting overhead. Proper GPU optimization would require ONNX+TensorRT or AMP (automatic mixed precision) rather than blanket model.half().

**CUDA TensorRT via torch.compile** (run\_005)  
Method: torch.compile(backend='torch\_tensorrt', dynamic=True) with TensorRT 10.15  
Device: NVIDIA RTX 5060 Ti 16GB (Blackwell, compute 12.0)  
Dataset: google/fleurs th\_th test (1021 sentences)

| Metric | CUDA FP32 | CUDA TRT | Change |
| :---- | :---- | :---- | :---- |
| WER | 0.2999 | 0.2999 | \+0.0% vs FP32 |
| RTF Mean | 0.0063 | 0.0081 | \+27.6% |
| RTF P90 | 0.0093 | 0.0095 | \+1.2% |
| Real-Time | ✓ Yes | ✓ Yes | — |

**Hypothesis Results:**  
H9 (TRT RTF \< FP32 RTF on GPU): REFUTED ✗ — \+27.6% RTF change vs CUDA FP32  
H10 (TRT WER matches FP32 \< 1%): SUPPORTED ✓ — \+0.0% relative WER difference  
	  
TensorRT via torch.compile is \+28% slower than native CUDA FP32 (RTF 0.0081 vs 0.0063). The RNN-T autoregressive decoder contains data-dependent control flow (aten.\_local\_scalar\_dense) that prevents TensorRT engine compilation. torch\_tensorrt falls back to GraphModule forward, providing no speedup. CUDA FP32 remains the optimal deployment configuration.

\[Euro\] 5\. LLM API for translation

| Model | Samples | BLEU Score | Average Latency (Second/Sentence) | WPM | Cost / sentence |
| :---: | :---: | :---: | :---: | :---: | :---: |
| gpt-4.1-nano | 500 | 23.18 | 0.748 | 1765.53 |  $0.000020 |
