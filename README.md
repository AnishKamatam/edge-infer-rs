# Edge-Infer-RS

A Rust-based **batched inference engine** for MobileNetV2 using **ONNX Runtime**.

I built this to understand what actually happens **after** “just run inference” — batching, queueing, tail latency, and what breaks when you push CPU inference hard.

This is not a wrapper or a demo. It’s a small inference system with real scheduling behavior and real measurements.

---

## What this does

- Accepts high-concurrency inference requests
- Batches them under a latency bound
- Runs **one ONNX inference per batch**
- Routes results back to callers
- Measures what actually matters: latency, throughput, queue depth, and saturation

Everything is CPU-only.

---

## Performance (Apple Silicon)

Measured on a MacBook Pro (Apple M-series) using a **saturation test with 2,000 total requests** released across **50 concurrent client threads**.

| Metric | Result |
|------|------|
| Sustained Throughput | **56.27 req/sec** |
| P50 End-to-End Latency | **166.81 ms** |
| P99 End-to-End Latency | **178.88 ms** |
| Avg Inference Time | **87.63 ms (batch of 8)** |
| Batch Fill Rate | **99.6% size-triggered** |

---

## What’s interesting here

- **Tail latency is tight**  
  P50 and P99 are only ~12ms apart. No queue collapse, no jitter spikes.

- **Batching actually works**  
  249 out of 250 batches were full (8/8). The scheduler consistently amortizes inference cost.

- **CPU limits show up fast**  
  Throughput plateaus early because memory bandwidth becomes the bottleneck. Bigger batches don’t magically help.

This matches what real CPU inference systems run into.

---

## How batching works

The scheduler flushes a batch when **either**:

- it reaches **8 requests**, or
- **50ms** passes since the first request arrived

This keeps latency bounded during low traffic while still batching aggressively under load.

---

## Load testing setup

To simulate burst traffic, the benchmark uses a synchronized barrier to release **50 threads at once**.  
This stresses queueing behavior, not just raw inference speed.

Requests block until their result is returned — no async cheating.

---

## Metrics snapshot

| Category | Metric | Value |
|--------|-------|------|
| Batching | Avg Batch Size | 8.00 |
|  | Size-Triggered Batches | 249 |
|  | Time-Triggered Batches | 1 |
| Latency | P50 End-to-End | 166.80 ms |
|  | P99 End-to-End | 178.87 ms |
| Performance | Avg Inference Time (batch) | 87.62 ms |
|  | Scheduler Overhead | 79.18 ms |
|  | Sustained Throughput | 56.27 req/s |

---

## Stack

- ONNX Runtime (`ort`)
- `ndarray`
- Rust std concurrency (`mpsc`, `Arc`, `Mutex`, `Barrier`)
- MobileNetV2 (ImageNet)

---

## Running it

```bash
cargo run --release
