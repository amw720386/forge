//! Token sampling strategies.

/// Greedy: argmax over logits `[vocab]`.
pub fn greedy(logits: &[f32]) -> u32 {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i as u32
}

/// Temperature softmax sampling (temperature <= 0 → greedy).
pub fn sample_temperature(logits: &[f32], temperature: f32, mut rng: impl FnMut() -> f32) -> u32 {
    if !(temperature > 0.0) {
        return greedy(logits);
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut weights = Vec::with_capacity(logits.len());
    let mut sum = 0.0f64;
    for &v in logits {
        let w = ((v - max) / temperature).exp() as f64;
        weights.push(w);
        sum += w;
    }
    let mut r = (rng() as f64) * sum;
    for (i, w) in weights.iter().enumerate() {
        r -= w;
        if r <= 0.0 {
            return i as u32;
        }
    }
    (logits.len() - 1) as u32
}

/// Top-k then temperature (k==0 → full vocab).
pub fn sample_top_k(
    logits: &[f32],
    k: usize,
    temperature: f32,
    rng: impl FnMut() -> f32,
) -> u32 {
    if k == 0 || k >= logits.len() {
        return sample_temperature(logits, temperature, rng);
    }
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    idx.truncate(k);
    let filtered: Vec<f32> = idx.iter().map(|&i| logits[i]).collect();
    let local = sample_temperature(&filtered, temperature, rng) as usize;
    idx[local] as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_max() {
        assert_eq!(greedy(&[1.0, 5.0, 3.0]), 1);
    }

    #[test]
    fn greedy_tie_break_lowest_index() {
        assert_eq!(greedy(&[2.0, 2.0, 1.0]), 0);
        assert_eq!(greedy(&[1.0, 3.0, 3.0]), 1);
    }
}
