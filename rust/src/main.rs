mod engine;
mod preprocess;
mod scheduler;
mod topk;

use scheduler::{BatchScheduler, RawMetrics};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};
use std::path::Path;

fn find_file(name: &str) -> String {
    if Path::new(name).exists() { name.to_string() } 
    else { format!("../{}", name) }
}

pub fn print_metrics_report(m: &RawMetrics, wall_time: Duration, total_reqs: usize) {
    let mut infer = m.infer_durations.clone();
    let mut total = m.total_latencies.clone();
    if infer.is_empty() || total.is_empty() { return; }
    
    infer.sort();
    total.sort();

    let p50 = total[total.len() / 2];
    let p95 = total[(total.len() as f32 * 0.95) as usize];
    let p99 = total[(total.len() as f32 * 0.99) as usize];

    let avg_infer: Duration = infer.iter().sum::<Duration>() / infer.len() as u32;
    let total_rps = total_reqs as f64 / wall_time.as_secs_f64();

    println!("\n{}", "=".repeat(40));
    println!("METRICS");
    println!("{}", "-".repeat(40));
    
    println!("BATCH BEHAVIOR");
    println!("  Avg Batch Size:  {:.2}", m.batch_sizes.iter().sum::<usize>() as f32 / m.batch_sizes.len() as f32);
    println!("  Size-Triggered:  {} batches", m.size_triggered);
    println!("  Time-Triggered:  {} batches", m.timeout_triggered);

    println!("\nLATENCY (End-to-End)");
    println!("  P50 (Median):    {:?}", p50);
    println!("  P95 (Tail):      {:?}", p95);
    println!("  P99 (Worst):     {:?}", p99);

    println!("\nENGINE PERFORMANCE");
    println!("  Avg Infer Time:  {:?}", avg_infer);
    println!("  Scheduler Oh:    {:?}", p50.checked_sub(avg_infer).unwrap_or(Duration::ZERO));
    println!("  System RPS:      {:.2} req/sec", total_rps);
    
    println!("{}\n", "=".repeat(40));
}

fn run_benchmark(total_reqs: usize) -> Result<(), Box<dyn std::error::Error>> {
    let model_path = find_file("model/mobilenet_v2.onnx");
    let image_path = find_file("assets/test.png");
    
    let scheduler = Arc::new(BatchScheduler::new(model_path, 8, 50)); 
    let input_tensor = preprocess::load_image(&image_path)?;
    
    let num_clients = 50;
    let reqs_per_client = total_reqs / num_clients;
    let barrier = Arc::new(Barrier::new(num_clients + 1));
    let mut handles = vec![];

    for _ in 0..num_clients {
        let sc = Arc::clone(&scheduler);
        let img = input_tensor.clone();
        let b = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            b.wait(); 
            for _ in 0..reqs_per_client { sc.predict(img.clone()); }
        }));
    }

    let start_time = Instant::now();
    barrier.wait(); 
    for h in handles { h.join().unwrap(); }
    let wall_time = start_time.elapsed();

    let metrics = scheduler.metrics.lock().unwrap().clone();
    print_metrics_report(&metrics, wall_time, total_reqs);

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting system benchmark...");
    run_benchmark(2000)?;
    Ok(())
}
