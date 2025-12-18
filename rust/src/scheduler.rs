use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, SyncSender};
use std::thread;
use std::time::{Duration, Instant};
use ndarray::{Array4, s};
use crate::engine::InferenceEngine;

const QUEUE_CAPACITY: usize = 5000;

#[derive(Default, Clone)]
pub struct RawMetrics {
    pub infer_durations: Vec<Duration>,
    pub total_latencies: Vec<Duration>,
    pub batch_sizes: Vec<usize>,
    pub queue_depths: Vec<usize>, 
    pub timeout_triggered: usize,
    pub size_triggered: usize,
}

pub struct BatchRequest {
    pub data: Array4<f32>,
    pub response_tx: SyncSender<Vec<f32>>,
    pub arrival_time: Instant,
}

pub struct BatchScheduler {
    request_tx: SyncSender<BatchRequest>,
    pub metrics: Arc<Mutex<RawMetrics>>,
    queue_count: Arc<AtomicUsize>,
}

impl BatchScheduler {
    pub fn new(model_path: String, max_batch: usize, timeout_ms: u64) -> Self {
        let (request_tx, request_rx) = sync_channel::<BatchRequest>(QUEUE_CAPACITY);
        let metrics = Arc::new(Mutex::new(RawMetrics::default()));
        let shared_rx = Arc::new(Mutex::new(request_rx));
        let queue_count = Arc::new(AtomicUsize::new(0));

        for _ in 0..4 {
            let rx_lock = Arc::clone(&shared_rx);
            let m_clone = Arc::clone(&metrics);
            let m_path = model_path.clone();
            let count_clone = Arc::clone(&queue_count);

            thread::spawn(move || {
                let mut engine = InferenceEngine::new(&m_path, "input", "logits").expect("Failed to load model");
                let mut batch_buffer = Array4::<f32>::zeros((max_batch, 3, 224, 224));
                let mut batch_queue = Vec::with_capacity(max_batch);
                
                loop {
                    let first_req = match rx_lock.lock().unwrap().recv() {
                        Ok(req) => {
                            count_clone.fetch_sub(1, Ordering::SeqCst);
                            req
                        },
                        Err(_) => break,
                    };

                    let start_wait = Instant::now();
                    batch_buffer.slice_mut(s![0..1, .., .., ..]).assign(&first_req.data);
                    batch_queue.push(first_req);

                    let mut triggered_by_timeout = true;
                    while batch_queue.len() < max_batch && start_wait.elapsed().as_millis() < timeout_ms as u128 {
                        let next_req = { rx_lock.lock().unwrap().try_recv().ok() };
                        if let Some(req) = next_req {
                            count_clone.fetch_sub(1, Ordering::SeqCst);
                            let slot = batch_queue.len();
                            batch_buffer.slice_mut(s![slot..slot+1, .., .., ..]).assign(&req.data);
                            batch_queue.push(req);
                            if batch_queue.len() == max_batch { triggered_by_timeout = false; }
                        } else {
                            thread::yield_now();
                        }
                    }

                    let bs = batch_queue.len();
                    let input = batch_buffer.slice(s![0..bs, .., .., ..]).to_owned();
                    
                    let t_start = Instant::now();
                    if let Ok(results) = engine.run_batch(input) {
                        let infer_dur = t_start.elapsed();
                        let mut m = m_clone.lock().unwrap();
                        m.infer_durations.push(infer_dur);
                        m.batch_sizes.push(bs);
                        if triggered_by_timeout { m.timeout_triggered += 1; } else { m.size_triggered += 1; }
                        
                        for (req, res) in batch_queue.drain(..).zip(results) {
                            m.total_latencies.push(req.arrival_time.elapsed());
                            let _ = req.response_tx.send(res);
                        }
                    }
                }
            });
        }
        Self { request_tx, metrics, queue_count }
    }

    pub fn predict(&self, data: Array4<f32>) -> Vec<f32> {
        let (tx, rx) = sync_channel(1);
        self.queue_count.fetch_add(1, Ordering::SeqCst);
        let _ = self.request_tx.send(BatchRequest { data, response_tx: tx, arrival_time: Instant::now() });
        rx.recv_timeout(Duration::from_secs(30)).unwrap_or_default()
    }
}
