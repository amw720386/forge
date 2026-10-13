//! prefill vs KV-cached decode, shapes, GQA.

use forge_runtime::model::{ForwardScratch, LayerKvCaches, ModelConfig, Transformer};

fn approx_eq(a: &[f32], b: &[f32], atol: f32, rtol: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol = atol + rtol * x.abs().max(y.abs());
        assert!(diff <= tol, "mismatch at {i}: got {x} want {y} (tol {tol})");
    }
}

#[test]
fn prefill_vs_token_by_token_kv_cache() {
    let cfg = ModelConfig::tiny();
    let model = Transformer::synthetic(cfg.clone(), 42).unwrap();
    let max_seq = 16;
    let tokens: Vec<u32> = vec![3, 7, 11, 2, 5];

    // Full prefill logits for every position
    let mut caches_full = LayerKvCaches::new(&cfg, max_seq).unwrap();
    let mut scratch_full = ForwardScratch::new(&cfg, max_seq).unwrap();
    let v = cfg.vocab_size;
    let full = model
        .forward_logits_all(&tokens, 0, &mut caches_full, &mut scratch_full)
        .unwrap()
        .data()
        .to_vec();

    // Token-by-token with KV cache — each step's logits match the prefill row.
    let mut caches_step = LayerKvCaches::new(&cfg, max_seq).unwrap();
    let mut scratch_step = ForwardScratch::new(&cfg, max_seq).unwrap();
    for (pos, &tok) in tokens.iter().enumerate() {
        model
            .forward_tokens(&[tok], pos, &mut caches_step, &mut scratch_step)
            .unwrap();
        approx_eq(
            &scratch_step.logits.data()[..v],
            &full[pos * v..(pos + 1) * v],
            1e-4,
            1e-3,
        );
    }
}

#[test]
fn hidden_states_prefill_vs_cached() {
    let cfg = ModelConfig::tiny();
    let model = Transformer::synthetic(cfg.clone(), 7).unwrap();
    let max_seq = 16;
    let tokens: Vec<u32> = vec![1, 2, 3, 4];

    let mut c1 = LayerKvCaches::new(&cfg, max_seq).unwrap();
    let mut s1 = ForwardScratch::new(&cfg, max_seq).unwrap();
    let h_full = model.forward_hidden(&tokens, 0, &mut c1, &mut s1).unwrap();

    let mut c2 = LayerKvCaches::new(&cfg, max_seq).unwrap();
    let mut s2 = ForwardScratch::new(&cfg, max_seq).unwrap();
    // Prefill all but last, then decode last
    let prefix = &tokens[..tokens.len() - 1];
    model.forward_tokens(prefix, 0, &mut c2, &mut s2).unwrap();
    model
        .forward_tokens(&[*tokens.last().unwrap()], prefix.len(), &mut c2, &mut s2)
        .unwrap();
    // Hidden for last token only after decode — compare last row of full
    let h = cfg.hidden_size;
    let last_full = &h_full.data()[(tokens.len() - 1) * h..];
    // Recompute hidden for last position via full path already in h_full;
    // after decode, scratch.x holds only the last token's hidden (t=1).
    let mut normed = forge_runtime::Tensor::zeros(&[1, h]).unwrap();
    let x_pref = forge_runtime::Tensor::from_vec(&[1, h], s2.x.data()[..h].to_vec()).unwrap();
    forge_runtime::tensor::rmsnorm_into(&x_pref, &model.final_norm, cfg.rms_norm_eps, &mut normed)
        .unwrap();
    approx_eq(normed.data(), last_full, 1e-4, 1e-3);
}

#[test]
fn cache_len_grows() {
    let cfg = ModelConfig::tiny();
    let model = Transformer::synthetic(cfg.clone(), 1).unwrap();
    let mut caches = LayerKvCaches::new(&cfg, 8).unwrap();
    let mut scratch = ForwardScratch::new(&cfg, 8).unwrap();
    model
        .forward_tokens(&[1, 2], 0, &mut caches, &mut scratch)
        .unwrap();
    assert_eq!(caches.layers[0].len(), 2);
    model
        .forward_tokens(&[3], 2, &mut caches, &mut scratch)
        .unwrap();
    assert_eq!(caches.layers[0].len(), 3);
}

#[test]
fn rejects_bad_token() {
    let cfg = ModelConfig::tiny();
    let model = Transformer::synthetic(cfg.clone(), 1).unwrap();
    let mut caches = LayerKvCaches::new(&cfg, 8).unwrap();
    let mut scratch = ForwardScratch::new(&cfg, 8).unwrap();
    assert!(model
        .forward_tokens(&[cfg.vocab_size as u32], 0, &mut caches, &mut scratch)
        .is_err());
}
