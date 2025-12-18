mod engine;
mod preprocess;
mod scheduler;
mod topk;
mod backend;
mod telemetry;

use scheduler::{BatchScheduler, ModelSpec};
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;
use std::fs;
use std::path::Path;
use std::time::Instant;

#[derive(Deserialize)]
struct ModelConfig {
    name: String, path: String, input_node: String, output_node: String,
    max_batch: usize, timeout_ms: u64, channels: usize, height: usize, width: usize,
}

#[derive(Deserialize)]
struct ServerConfig { models: Vec<ModelConfig> }

struct InferenceServer {
    models: HashMap<String, Arc<BatchScheduler>>,
    labels: Vec<String>,
}

impl InferenceServer {
    fn from_config(config_path: &str) -> Self {
        let config_str = fs::read_to_string(config_path).expect("Unable to read config.json");
        let config: ServerConfig = serde_json::from_str(&config_str).expect("JSON error");
        let labels = topk::load_labels("labels.txt");
        let mut server = Self { models: HashMap::new(), labels };

        for cfg in config.models {
            if !Path::new(&cfg.path).exists() { continue; }
            println!("BOOTING: [{}]", cfg.name);
            let spec = ModelSpec {
                name: cfg.name.clone(), path: cfg.path, input_node: cfg.input_node,
                output_node: cfg.output_node, max_batch: cfg.max_batch, timeout_ms: cfg.timeout_ms,
                channels: cfg.channels, height: cfg.height, width: cfg.width,
            };
            server.models.insert(cfg.name, Arc::new(BatchScheduler::new(spec)));
        }
        server
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let server = Arc::new(InferenceServer::from_config("config.json"));
    let image_names = ["test.png", "test1.png", "test2.png", "test3.png", "test4.png"];
    let mut test_images = Vec::new();

    for name in image_names {
        let path = format!("../assets/{}", name);
        if Path::new(&path).exists() {
            test_images.push((name.to_string(), preprocess::load_image(&path)?));
        }
    }

    if test_images.is_empty() {
        println!("No images found in ../assets/. Please check the folder.");
        return Ok(());
    }

    println!("\nRUNNING THOROUGH ML AUDIT...");
    let separator = "=".repeat(75);
    println!("{}", separator);
    println!("{:<12} | {:<12} | {:<20} | {:<10}", "MODEL", "IMAGE", "PREDICTION", "CONF %");
    println!("{}", "-".repeat(75));

    for (img_name, img_data) in &test_images {
        for model_key in ["resnet", "mobilenet", "efficientnet"] {
            if let Some(sched) = server.models.get(model_key) {
                let start = Instant::now();
                let results = sched.predict(img_data.clone());
                let latency = start.elapsed();

                let top = topk::get_top_results(&results, 1, &server.labels);
                if let Some((label, prob)) = top.first() {
                    println!(
                        "{:<12} | {:<12} | {:<20} | {:.2}%", 
                        model_key.to_uppercase(), img_name, label, prob * 100.0
                    );

                    telemetry::log_inference(telemetry::TelemetryRecord {
                        model: model_key.to_string(),
                        image: img_name.to_string(),
                        label: label.to_string(),
                        confidence: *prob,
                        latency,
                    });
                }
            }
        }
    }
    println!("{}\n[Audit complete. Results saved to inference_audit.csv]", separator);

    Ok(())
}