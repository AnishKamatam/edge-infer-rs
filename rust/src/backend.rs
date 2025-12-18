use ndarray::Array4;
use std::error::Error;

pub trait ModelBackend: Send + Sync {
    fn run_batch(&mut self, input: Array4<f32>) -> Result<Vec<Vec<f32>>, Box<dyn Error>>;
    fn name(&self) -> &str;
}