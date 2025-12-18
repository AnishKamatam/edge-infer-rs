#[allow(dead_code)]
pub fn print_top_k(logits: &[f32], k: usize) {
    let mut indexed_logits: Vec<(usize, &f32)> = logits.iter().enumerate().collect();
    // Sort descending
    indexed_logits.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("Top {} Predictions:", k);
    for i in 0..k {
        let (idx, score) = indexed_logits[i];
        println!("  [{}] Class Index: {} (Score: {:.4})", i + 1, idx, score);
    }
}