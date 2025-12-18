use ort::session::Session;
use ort::value::Value;
use ndarray::Array4;
use std::error::Error;

pub struct Model {
    session: Session,
}

impl Model {
    pub fn new(model_path: &str) -> Result<Self, Box<dyn Error>> {
        let session = Session::builder()?
            .commit_from_file(model_path)?;
        Ok(Self { session })
    }

    pub fn predict(&mut self, input: Array4<f32>) -> Result<Vec<f32>, Box<dyn Error>> {
        let shape = input.shape().to_vec();
        let data = input.into_raw_vec();
        let input_tensor = Value::from_array((shape, data))?;
        
        let outputs = self.session.run(ort::inputs!["input" => input_tensor])?;
        let (_shape, data_slice) = outputs["logits"].try_extract_tensor::<f32>()?;
        
        Ok(data_slice.to_vec())
    }
}