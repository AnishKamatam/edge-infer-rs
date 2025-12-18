use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Value;
use ndarray::Array4;
use std::error::Error;
use std::path::Path;

pub struct InferenceEngine {
    session: Session,
    input_node_name: String,
    output_node_name: String,
}

impl InferenceEngine {
    pub fn new(
        model_path: impl AsRef<Path>,
        input_name: &str,
        output_name: &str,
    ) -> Result<Self, Box<dyn Error>> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)? 
            .commit_from_file(&model_path)?;

        Ok(Self {
            session,
            input_node_name: input_name.to_string(),
            output_node_name: output_name.to_string(),
        })
    }

    pub fn run_batch(&mut self, input: Array4<f32>) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
        let shape = input.shape().to_vec();
        let data = input.into_raw_vec();
        let input_tensor = Value::from_array((shape, data))?;
        let outputs = self.session.run(ort::inputs![self.input_node_name.as_str() => input_tensor])?;
        let (out_shape, data_slice) = outputs[self.output_node_name.as_str()].try_extract_tensor::<f32>()?;
        let num_classes = out_shape[1] as usize;
        
        Ok(data_slice.chunks(num_classes).map(|chunk| chunk.to_vec()).collect())
    }
}
