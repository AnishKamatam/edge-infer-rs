use image::GenericImageView;
use ndarray::Array4;

pub fn load_image(path: &str) -> Result<Array4<f32>, Box<dyn std::error::Error>> {
    let img = image::open(path)?;
    let resized = img.resize_exact(224, 224, image::imageops::FilterType::Lanczos3);
    
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];

    let mut array = Array4::zeros((1, 3, 224, 224));

    for (x, y, pixel) in resized.pixels() {
        let r = pixel[0] as f32 / 255.0;
        let g = pixel[1] as f32 / 255.0;
        let b = pixel[2] as f32 / 255.0;

        array[[0, 0, y as usize, x as usize]] = (r - mean[0]) / std[0];
        array[[0, 1, y as usize, x as usize]] = (g - mean[1]) / std[1];
        array[[0, 2, y as usize, x as usize]] = (b - mean[2]) / std[2];
    }

    Ok(array)
}