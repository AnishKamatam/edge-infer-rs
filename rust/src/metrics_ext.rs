use crate::scheduler::RawMetrics;
use std::time::Duration;

pub fn print_advanced_report(m: &RawMetrics, wall_time: Duration, total_reqs: usize) {
    let mut infer = m.infer_durations.clone();
    let mut total = m.total_latencies.clone();
    infer.sort();
    total.sort();

    let p50 = total[total.len() / 2];
    let p95 = total[(total.len() as f32 * 0.95) as usize];
    let p99 = total[(total.len() as f32 * 0.99) as usize];

    let avg_infer: Duration = infer.iter().sum::<Duration>() / infer.len() as u32;
    let total_rps = total_reqs as f64 / wall_time.as_secs_f64();

    println!("\n{}", "=".repeat(40));
    println!("ADVANCED PERFORMANCE REPORT");
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
    
    println!("{}", "=".repeat(40));
}