use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::execution_providers::CoreMLExecutionProvider;
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
    /// Loads ONNX model with optimization and CoreML execution provider
    pub fn new(
        model_path: impl AsRef<Path>,
        input_name: &str,
        output_name: &str,
    ) -> Result<Self, Box<dyn Error>> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([CoreMLExecutionProvider::default().build()])?
            .commit_from_file(model_path)?;

        Ok(Self {
            session,
            input_node_name: input_name.to_string(),
            output_node_name: output_name.to_string(),
        })
    }

    /// Runs inference on preprocessed input array
    pub fn run(&mut self, input: Array4<f32>) -> Result<Vec<f32>, Box<dyn Error>> {
        let shape = input.shape().to_vec();
        let data = input.into_raw_vec();
        let input_tensor = Value::from_array((shape, data))?;
        
        let outputs = self.session.run(ort::inputs![self.input_node_name.as_str() => input_tensor])?;
        let (_shape, data_slice) = outputs[self.output_node_name.as_str()]
            .try_extract_tensor::<f32>()?;

        Ok(data_slice.to_vec())
    }
}
