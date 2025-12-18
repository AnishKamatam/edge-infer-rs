mod engine;     // Register engine.rs
mod preprocess; // Register preprocess.rs
mod topk;       // Register topk.rs

use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Now the compiler will find engine::InferenceEngine
    let mut engine = engine::InferenceEngine::new("../model/mobilenet_v2.onnx", "input", "logits")?;
    
    let input = preprocess::load_image("../assets/test.png")?;

    // Warmup (important for ORT and CoreML)
    println!("Warming up...");
    for _ in 0..5 {
        let _ = engine.run(input.clone())?;
    }

    let runs = 20;
    let mut total = std::time::Duration::ZERO;

    println!("Benchmarking {} runs...", runs);
    for _ in 0..runs {
        let start = Instant::now();
        let _ = engine.run(input.clone())?;
        total += start.elapsed();
    }

    let avg = total / runs;
    println!("------------------------------------");
    println!("Avg latency: {:.2?}", avg);
    
    let final_results = engine.run(input)?;
    topk::print_top_k(&final_results, 5);

    Ok(())
}