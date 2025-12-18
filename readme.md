# Edge-Infer-RS

A high-performance, asynchronous batching inference engine for **MobileNetV2**, built in Rust using **ONNX Runtime**. This project demonstrates an efficient bridge between high-frequency request streams and hardware-constrained deep learning models through dynamic batching and lock-aware scheduling.



## Performance Metrics (Apple M-Series)

The following metrics were captured on a MacBook Pro during a saturation test of **2,000 concurrent requests** distributed across **50 parallel client threads**.

| Metric | Result |
| :--- | :--- |
| **Throughput** | **56.27 req/sec** |
| **P50 Latency (Median)** | **166.81 ms** |
| **P99 Latency (Worst)** | **178.88 ms** |
| **Avg. Engine Infer Time** | **87.63 ms (per batch of 8)** |
| **Batch Saturation** | **99.6% (Size-Triggered)** |

### Analysis
* **Tail Latency Stability**: The P50 and P99 are separated by only **12ms**. This indicates an extremely stable scheduler with minimal jitter and no significant thread contention.
* **Batching Efficiency**: Out of 250 processed batches, **249 were filled to capacity (8/8)**. This confirms that the producer-consumer bridge is tuned for maximum hardware utilization.
* **Hardware Alignment**: By using **4 parallel workers**, the engine saturates the Performance cores of the Apple Silicon chip, achieving approximately 11ms per-image inference within the batch.

---

## Technical Architecture

### 1. Dynamic Batching Scheduler
The `BatchScheduler` balances throughput and latency by grouping individual requests into batches based on two specific triggers:
* **Size Trigger**: Execution is forced once the batch reaches **8 requests**.
* **Time Trigger**: Execution is forced after **50ms**, ensuring that low-traffic periods do not lead to indefinite latency.



### 2. High-Concurrency Client Pool
To simulate real-world production load, the benchmark uses a **Synchronized Barrier** to release 50 threads simultaneously. This tests the system's ability to handle burst traffic scenarios without crashing or deadlocking.

### 3. Lock-Aware Design
The implementation utilizes a **Lock-Receive-Unlock** pattern with `std::sync::mpsc`. By minimizing the duration worker threads hold Mutexes, the system prevents lock convoys, allowing producers to push data even while workers are processing heavy inference tasks.

---

## Technical Stack
* **Runtime**: [ONNX Runtime (ort)](https://github.com/pykeio/ort)
* **Math/Tensors**: `ndarray`
* **Concurrency**: Rust Standard Library (`mpsc`, `Arc`, `Mutex`, `Barrier`)
* **Model**: MobileNetV2 (ImageNet classification)

## Usage

### Run the Benchmark
```bash
cargo run --release


| Category | Metric | Value |
|--------|-------|------|
| **Batching** | Avg Batch Size | 8.00 |
|  | Size-Triggered Batches | 249 |
|  | Time-Triggered Batches | 1 |
| **Latency** | P50 End-to-End | 166.80 ms |
|  | P99 End-to-End | 178.87 ms |
| **Performance** | Avg Inference Time (batch) | 87.62 ms |
|  | Scheduler Overhead | 79.18 ms |
|  | Sustained Throughput | 56.27 req/s |
