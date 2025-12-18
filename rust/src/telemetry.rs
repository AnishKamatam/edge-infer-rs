use std::fs::OpenOptions;
use std::io::Write;
use std::time::Duration;

pub struct TelemetryRecord {
    pub model: String,
    pub image: String,
    pub label: String,
    pub confidence: f32,
    pub latency: Duration,
}

pub fn log_inference(record: TelemetryRecord) {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("inference_audit.csv")
        .unwrap();

    if file.metadata().unwrap().len() == 0 {
        writeln!(file, "model,image,label,confidence,latency_ms").unwrap();
    }

    writeln!(
        file, 
        "{},{},{},{:.4},{}", 
        record.model, 
        record.image, 
        record.label, 
        record.confidence, 
        record.latency.as_millis()
    ).unwrap();
}