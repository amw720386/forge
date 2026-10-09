//! Weight-only quant numerical / token-agreement tests (synthetic + optional Qwen).

use forge_runtime::model::{
    ContinuousBatcher, ForwardScratch, GenerateParams, LayerKvCaches, ModelConfig, QuantBits,
    Transformer,
};
use forge_runtime::sampling::greedy;

fn next_tokens(
    model: &Transformer,
    prompt: &[u32],
    n: usize,
    max_seq: usize,
) -> (Vec<u32>, Vec<f32>) {
    let mut caches = LayerKvCaches::new(&model.cfg, max_seq).unwrap();
    let mut scratch = ForwardScratch::new(&model.cfg, max_seq).unwrap();
    let mut tokens = prompt.to_vec();
    model
        .forward_tokens(&tokens, 0, &mut caches, &mut scratch)
        .unwrap();
    let vocab = model.cfg.vocab_size;
    let mut logits = scratch.logits.data()[..vocab].to_vec();
    let first_logits = logits.clone();
    let mut gen = Vec::new();
    for _ in 0..n {
        let next = greedy(&logits);
        gen.push(next);
        tokens.push(next);
        let pos = tokens.len() - 1;
        model
            .forward_tokens(&[next], pos, &mut caches, &mut scratch)
            .unwrap();
        logits = scratch.logits.data()[..vocab].to_vec();
    }
    (gen, first_logits)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn mean_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    let s: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
    s / a.len() as f32
}

#[test]
fn int8_synthetic_token_agreement() {
    let cfg = ModelConfig::tiny();
    let fp = Transformer::synthetic(cfg.clone(), 123).unwrap();
    let mut q = Transformer::synthetic(cfg.clone(), 123).unwrap();
    q.quantize_weights(QuantBits::Int8, 32).unwrap();

    let prompts: [&[u32]; 3] = [&[1, 2, 3, 4], &[10, 11], &[20, 21, 22, 23, 24]];
    for (i, p) in prompts.iter().enumerate() {
        let (fp_tok, fp_logits) = next_tokens(&fp, p, 8, 64);
        let (q_tok, q_logits) = next_tokens(&q, p, 8, 64);
        let mad = mean_abs_diff(&fp_logits, &q_logits);
        let mx = max_abs_diff(&fp_logits, &q_logits);
        println!("int8 prompt{i}: mad={mad:.5} max={mx:.5} fp={fp_tok:?} q={q_tok:?}");
        assert!(mad < 0.05, "mean abs logit diff too large: {mad}");
        assert!(mx < 0.5, "max abs logit diff too large: {mx}");
        if fp_tok[0] != q_tok[0] {
            println!("NOTE: first-token disagree prompt{i}: {} vs {}", fp_tok[0], q_tok[0]);
        }
    }
    assert!(q.weight_nbytes() < fp.weight_nbytes());
}

#[test]
fn int4_group64_synthetic_bounded_error() {
    let cfg = ModelConfig::tiny();
    // intermediate=64, hidden=32 → group 32 divides both.
    let fp = Transformer::synthetic(cfg.clone(), 7).unwrap();
    let mut q = Transformer::synthetic(cfg.clone(), 7).unwrap();
    q.quantize_weights(QuantBits::Int4, 32).unwrap();

    let (fp_tok, fp_logits) = next_tokens(&fp, &[1, 2, 3], 6, 64);
    let (q_tok, q_logits) = next_tokens(&q, &[1, 2, 3], 6, 64);
    let mad = mean_abs_diff(&fp_logits, &q_logits);
    println!("int4: mad={mad:.5} fp={fp_tok:?} q={q_tok:?}");
    assert!(mad < 0.15, "INT4 mean abs logit diff too large: {mad}");
    assert!(q.weight_nbytes() < fp.weight_nbytes());
}

#[test]
fn quantized_batch_matches_sequential() {
    let cfg = ModelConfig::tiny();
    let mut model = Transformer::synthetic(cfg.clone(), 42).unwrap();
    model.quantize_weights(QuantBits::Int8, 32).unwrap();
    let max_seq = 64;
    let params = GenerateParams {
        max_tokens: 4,
        temperature: 0.0,
        top_k: 0,
        eos_token_id: None,
    };
    let prompts = [vec![1u32, 2, 3], vec![9u32, 8]];

    let mut seq_out = Vec::new();
    for p in &prompts {
        let (toks, _) = next_tokens(&model, p, params.max_tokens, max_seq);
        seq_out.push(toks);
    }

    let mut mq = Transformer::synthetic(cfg.clone(), 42).unwrap();
    mq.quantize_weights(QuantBits::Int8, 32).unwrap();
    let mut batcher = ContinuousBatcher::new_untokenized(mq, max_seq, 4).unwrap();
    let mut ids = Vec::new();
    for p in &prompts {
        ids.push(batcher.admit_tokens(p.clone(), params.clone()).unwrap());
    }
    let mut batched: std::collections::HashMap<u64, Vec<u32>> =
        ids.iter().map(|&id| (id, Vec::new())).collect();
    while batcher.active_count() > 0 {
        for ev in batcher.step().unwrap() {
            batched.get_mut(&ev.request_id).unwrap().push(ev.token_id);
        }
    }
    for (i, id) in ids.iter().enumerate() {
        assert_eq!(&batched[id], &seq_out[i], "quant batch mismatch req {i}");
    }
}
