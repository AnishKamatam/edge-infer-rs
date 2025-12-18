pub fn print_top_k(logits: &[f32], k: usize) {
    let mut pairs: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Top {k} predictions (class_id, score):");
    for (rank, (idx, score)) in pairs.into_iter().take(k).enumerate() {
        println!("{}. {}  ({:.4})", rank + 1, idx, score);
    }
}
