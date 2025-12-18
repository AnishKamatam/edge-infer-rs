/// Converts logits to probabilities via softmax and prints top k predictions
pub fn print_top_k(logits: &[f32], k: usize) {
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f32> = logits.iter().map(|x| (x - max_logit).exp()).collect();
    let sum_exps: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|x| x / sum_exps).collect();

    let mut pairs: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\nTop {k} Predictions:");
    println!("--------------------------");
    for (rank, (idx, prob)) in pairs.into_iter().take(k).enumerate() {
        println!("{:>2}. Class {:<4} | Confidence: {:.2}%", rank + 1, idx, prob * 100.0);
    }
}
