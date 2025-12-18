use std::sync::{Arc, Mutex};
use std::sync::mpsc::{sync_channel, SyncSender};
use std::thread;
use std::time::{Duration, Instant};
use ndarray::{Array4, s};
use crate::backend::ModelBackend;
use crate::engine::OnnxBackend;

#[allow(dead_code)]
pub struct ModelSpec {
    pub name: String,
    pub path: String,
    pub input_node: String,
    pub output_node: String,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub max_batch: usize,
    pub timeout_ms: u64,
}

pub struct BatchRequest {
    pub data: Array4<f32>,
    pub response_tx: SyncSender<Vec<f32>>,
    pub _arrival_time: Instant,
}

pub struct BatchScheduler {
    pub _spec: Arc<ModelSpec>,
    request_tx: SyncSender<BatchRequest>,
}

impl BatchScheduler {
    pub fn new(spec: ModelSpec) -> Self {
        let spec = Arc::new(spec);
        let (request_tx, request_rx) = sync_channel::<BatchRequest>(5000);
        let shared_rx = Arc::new(Mutex::new(request_rx));

        let s_clone = Arc::clone(&spec);
        thread::spawn(move || {
            let mut backend: Box<dyn ModelBackend> = Box::new(
                OnnxBackend::new(&s_clone.path, &s_clone.input_node, &s_clone.output_node)
                    .expect("Backend init failed")
            );

            println!("    [Worker {}] Started using backend: {}", s_clone.name, backend.name());

            let mut batch_buffer = Array4::<f32>::zeros((
                s_clone.max_batch, s_clone.channels, s_clone.height, s_clone.width
            ));
            let mut batch_queue = Vec::with_capacity(s_clone.max_batch);
            
            loop {
                let first_req = match shared_rx.lock().unwrap().recv() {
                    Ok(req) => req,
                    Err(_) => break,
                };

                let start_wait = Instant::now();
                batch_buffer.slice_mut(s![0..1, .., .., ..]).assign(&first_req.data);
                batch_queue.push(first_req);

                while batch_queue.len() < s_clone.max_batch && 
                      start_wait.elapsed().as_millis() < s_clone.timeout_ms as u128 {
                    if let Ok(req) = shared_rx.lock().unwrap().try_recv() {
                        let slot = batch_queue.len();
                        batch_buffer.slice_mut(s![slot..slot+1, .., .., ..]).assign(&req.data);
                        batch_queue.push(req);
                    } else {
                        thread::yield_now();
                    }
                }

                let bs = batch_queue.len();
                let input = batch_buffer.slice(s![0..bs, .., .., ..]).to_owned();
                
                if let Ok(results) = backend.run_batch(input) {
                    for (req, res) in batch_queue.drain(..).zip(results) {
                        let _ = req.response_tx.send(res);
                    }
                }
            }
        });

        Self { _spec: spec, request_tx }
    }

    pub fn predict(&self, data: Array4<f32>) -> Vec<f32> {
        let (tx, rx) = sync_channel(1);
        let _ = self.request_tx.send(BatchRequest { 
            data, response_tx: tx, _arrival_time: Instant::now() 
        });
        rx.recv_timeout(Duration::from_secs(30)).unwrap_or_default()
    }
}