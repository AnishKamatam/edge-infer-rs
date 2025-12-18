mod engine;
mod preprocess;
mod topk;

use engine::InferenceEngine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut vision_engine = InferenceEngine::new(
        "../model/mobilenet_v2.onnx", 
        "input", 
        "logits"
    )?;

    let input = preprocess::load_image("../assets/test.png")?;
    let results = vision_engine.run(input)?;
    topk::print_top_k(&results, 5);
    
    Ok(())
}