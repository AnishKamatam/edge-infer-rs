use std::fs;

pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|x| (x - max_logit).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|x| x / sum).collect()
}

pub fn get_top_results(results: &[f32], k: usize, labels: &[String]) -> Vec<(String, f32)> {
    let probabilities = softmax(results);
    let mut indexed: Vec<(usize, f32)> = probabilities.into_iter().enumerate().collect();
    
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    indexed.into_iter()
        .take(k)
        .map(|(idx, prob)| {
            let label = labels.get(idx).cloned().unwrap_or_else(|| format!("ID:{}", idx));
            (label, prob)
        })
        .collect()
}

pub fn load_labels(path: &str) -> Vec<String> {
    fs::read_to_string(path)
        .expect("Failed to read labels.txt")
        .lines()
        .map(|s| s.to_string())
        .collect()
}