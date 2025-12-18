mod model;
mod preprocess;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "../model/mobilenet_v2.onnx";
    let image_path = "../assets/test.png";

    let mut model = model::Model::new(model_path)?;
    let input = preprocess::load_image(image_path)?;
    let logits = model.predict(input)?;

    println!("Inference successful. Output size: {}", logits.len());
    
    if let Some((idx, val)) = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()) {
        println!("Top prediction: class {} (score: {:.4})", idx, val);
    }

    Ok(())
}