use crate::backend::ModelBackend;
use ort::session::Session;
use ort::value::Value;
use ndarray::Array4;
use std::error::Error;

pub struct OnnxBackend {
    session: Session,
    input_node: String,
}

impl OnnxBackend {
    pub fn new(path: &str, input_node: &str, _output_node: &str) -> Result<Self, Box<dyn Error>> {
        let session = Session::builder()?
            .commit_from_file(path)?;
        
        Ok(Self {
            session,
            input_node: input_node.to_string(),
        })
    }
}

impl ModelBackend for OnnxBackend {
    fn name(&self) -> &str { "ONNXRuntime-v2.0-rc" }

    fn run_batch(&mut self, input: Array4<f32>) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
        let shape = input.shape().to_vec();
        let flat_data = input.into_raw_vec();
        let value = Value::from_array((shape, flat_data))?;
        
        let outputs = self.session.run(vec![(self.input_node.as_str(), value)])?;
        let (out_shape, out_data) = outputs[0].try_extract_tensor::<f32>()?;
        
        let batch_size = out_shape[0] as usize;
        let total_elements = out_data.len();
        let elements_per_batch = total_elements / batch_size;
        
        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let start = i * elements_per_batch;
            let end = start + elements_per_batch;
            results.push(out_data[start..end].to_vec());
        }
        
        Ok(results)
    }
}