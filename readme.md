# Edge-Infer-RS: High-Performance Inference Framework

**Edge-Infer-RS** is a production-grade, trait-abstracted batching inference engine built in Rust. It provides a high-throughput bridge between asynchronous request streams and hardware-accelerated deep learning runtimes, with a focus on low-latency, CPU-efficient execution at the edge.

---

## Technical Architecture

### 1. Multi-Backend Abstraction

The system uses a **trait-based runtime abstraction** to decouple scheduling logic from model execution. By implementing the `ModelBackend` trait, different inference runtimes can be swapped without modifying the core scheduler.

Supported / planned backends:
- ONNX Runtime
- LibTorch
- TensorRT

This design mirrors production inference systems by separating the **control plane** (batching, scheduling, metrics) from the **execution plane** (kernel runtime).

---

### 2. Dynamic Batching Scheduler

The scheduler maximizes hardware utilization while respecting latency SLAs through two execution triggers:

- **Size Trigger**  
  Inference executes immediately when the batch reaches `max_batch` (e.g., 8), maximizing throughput under high load.

- **Time Trigger**  
  Inference executes after `timeout_ms` (e.g., 50 ms) if the batch is not full, ensuring bounded latency during low-traffic periods.

This dual-trigger mechanism enables stable tail latency while maintaining near-optimal batch saturation.

---

### 3. Numerical Integrity & Preprocessing

To ensure correctness and comparability with industry benchmarks, the engine implements standard vision preprocessing and postprocessing pipelines:

- **Input Normalization**  
  Applies ImageNet normalization:  
  `mean = [0.485, 0.456, 0.406]`  
  `std  = [0.229, 0.224, 0.225]`

- **Probability Distribution**  
  Uses a numerically stable Softmax implementation to convert raw logits into calibrated confidence scores.

---

## Performance Metrics (Apple M-Series)

Metrics captured during a saturation test of **2,000 concurrent requests** distributed across **50 parallel threads**.

| Metric                         | Result                     |
|--------------------------------|----------------------------|
| Throughput                     | 56.27 req/sec              |
| P50 Latency (Median)           | 166.81 ms                  |
| P99 Latency (Worst)            | 178.88 ms                  |
| Avg. Engine Infer Time         | 87.63 ms (per batch of 8)  |
| Batch Saturation               | 99.6% (Size-Triggered)     |

---

## Performance Analysis

- **Tail Latency Stability**  
  The P99–P50 delta is only **12 ms**, indicating a highly stable scheduler with minimal contention and predictable execution behavior.

- **Hardware Alignment**  
  The worker pool effectively saturates the Apple ARM performance cores, achieving approximately **11 ms per image** when executed within a full batch.

---

## Features

### Inference Telemetry & Auditing

Every inference request is logged to `inference_audit.csv`, enabling:

- **Model Drift Detection**  
  Monitoring confidence score degradation over time.
- **Latency Auditing**  
  Evaluating performance under varying load conditions.
- **Cross-Model Verification**  
  Comparing predictions across models (e.g., ResNet vs. MobileNet) for identical inputs.

---

## Technical Stack

- **Runtime:** ONNX Runtime (`ort` 2.0-rc)
- **Math / Tensors:** `ndarray`
- **Serialization:** `serde`, `serde_json`
- **Vision:** `image` (Lanczos3 resampling)
- **Concurrency:** `std::sync` (`mpsc`, `Arc`, `Mutex`, `Barrier`)

---

## Usage

### 1. Configuration

Define models and batching constraints in `config.json`:

```json
{
  "models": [
    {
      "name": "mobilenet",
      "path": "../model/mobilenet_v2.onnx",
      "input_node": "input",
      "output_node": "output",
      "max_batch": 8,
      "timeout_ms": 50,
      "channels": 3,
      "height": 224,
      "width": 224
    },
    {
      "name": "resnet",
      "path": "../model/resnet50.onnx",
      "input_node": "input",
      "output_node": "output",
      "max_batch": 4,
      "timeout_ms": 100,
      "channels": 3,
      "height": 224,
      "width": 224
    },
    {
      "name": "efficientnet",
      "path": "../model/efficientnet_b0.onnx",
      "input_node": "input",
      "output_node": "output",
      "max_batch": 8,
      "timeout_ms": 50,
      "channels": 3,
      "height": 224,
      "width": 224
    }
  ]
}

## 2. Execution

Run the full ML audit and performance benchmark:

```bash
cargo run --release
