use image::{io::Reader as ImageReader, imageops::FilterType, GenericImageView};
use ndarray::Array4;
use std::error::Error;

pub fn load_image(path: &str) -> Result<Array4<f32>, Box<dyn Error>> {
    let img = ImageReader::open(path)?.decode()?;
    let resized = img.resize_exact(224, 224, FilterType::Triangle);
    
    let mut array = Array4::zeros((1, 3, 224, 224));
    for (x, y, pixel) in resized.pixels() {
        array[[0, 0, y as usize, x as usize]] = (pixel[0] as f32 / 255.0 - 0.485) / 0.229;
        array[[0, 1, y as usize, x as usize]] = (pixel[1] as f32 / 255.0 - 0.456) / 0.224;
        array[[0, 2, y as usize, x as usize]] = (pixel[2] as f32 / 255.0 - 0.406) / 0.225;
    }
    Ok(array)
}