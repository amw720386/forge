//! `.forge` serialization / validation / inference parity tests.

use forge_runtime::checkpoint::{
    load_forge_path, save_forge, validate_checkpoint_bytes, LoadOptions, SaveOptions,
};
use forge_runtime::model::{
    ForwardScratch, GenerateParams, LayerKvCaches, ModelConfig, QuantBits, Transformer,
};
use forge_runtime::sampling::greedy;
use std::path::PathBuf;

fn tmp_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("forge_test_{name}_{}.forge", std::process::id()));
    p
}

fn greedy_n(model: &Transformer, prompt: &[u32], n: usize) -> Vec<u32> {
    let max_seq = 64;
    let mut caches = LayerKvCaches::new(&model.cfg, max_seq).unwrap();
    let mut scratch = ForwardScratch::new(&model.cfg, max_seq).unwrap();
    let mut tokens = prompt.to_vec();
    model
        .forward_tokens(&tokens, 0, &mut caches, &mut scratch)
        .unwrap();
    let vocab = model.cfg.vocab_size;
    let mut logits = scratch.logits.data()[..vocab].to_vec();
    let mut out = Vec::new();
    for _ in 0..n {
        let next = greedy(&logits);
        out.push(next);
        tokens.push(next);
        let pos = tokens.len() - 1;
        model
            .forward_tokens(&[next], pos, &mut caches, &mut scratch)
            .unwrap();
        logits = scratch.logits.data()[..vocab].to_vec();
    }
    out
}

#[test]
fn roundtrip_fp32_synthetic() {
    let cfg = ModelConfig::tiny();
    let model = Transformer::synthetic(cfg, 11).unwrap();
    let path = tmp_path("fp32");
    save_forge(
        &model,
        &path,
        SaveOptions {
            name: "tiny".into(),
            bos_token_id: 1,
            eos_token_id: 2,
        },
    )
    .unwrap();
    let bytes = std::fs::read(&path).unwrap();
    validate_checkpoint_bytes(&bytes).unwrap();
    let loaded = load_forge_path(&path, LoadOptions { memory_map: false }).unwrap();
    let a = greedy_n(&model, &[1, 2, 3], 4);
    let b = greedy_n(&loaded.model, &[1, 2, 3], 4);
    assert_eq!(a, b);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn roundtrip_int8_matches_inmemory_quant() {
    let cfg = ModelConfig::tiny();
    let mut model = Transformer::synthetic(cfg.clone(), 21).unwrap();
    model.quantize_weights(QuantBits::Int8, 32).unwrap();
    let path = tmp_path("int8");
    save_forge(&model, &path, SaveOptions::default()).unwrap();
    let loaded = load_forge_path(&path, LoadOptions { memory_map: true }).unwrap();
    assert_eq!(loaded.meta.quant_bits, 8);
    assert_eq!(loaded.meta.quant_group_size, 32);
    // Owned tensors must remain usable after the mapped source is removed.
    std::fs::remove_file(&path).unwrap();
    let a = greedy_n(&model, &[5, 6, 7], 5);
    let b = greedy_n(&loaded.model, &[5, 6, 7], 5);
    assert_eq!(
        a, b,
        "checkpoint INT8 must match in-memory quantized greedy"
    );
}

#[test]
fn roundtrip_int4() {
    let cfg = ModelConfig::tiny();
    let mut model = Transformer::synthetic(cfg, 33).unwrap();
    model.quantize_weights(QuantBits::Int4, 32).unwrap();
    let path = tmp_path("int4");
    save_forge(&model, &path, SaveOptions::default()).unwrap();
    let loaded = load_forge_path(&path, LoadOptions { memory_map: false }).unwrap();
    assert_eq!(loaded.meta.quant_bits, 4);
    let a = greedy_n(&model, &[1, 2], 3);
    let b = greedy_n(&loaded.model, &[1, 2], 3);
    assert_eq!(a, b);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn rejects_truncated_and_bad_magic() {
    let cfg = ModelConfig::tiny();
    let model = Transformer::synthetic(cfg, 1).unwrap();
    let path = tmp_path("bad");
    save_forge(&model, &path, SaveOptions::default()).unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    assert!(validate_checkpoint_bytes(&bytes).is_ok());
    bytes.truncate(64);
    assert!(validate_checkpoint_bytes(&bytes).is_err());
    let mut bad = std::fs::read(&path).unwrap();
    bad[0] = b'X';
    assert!(validate_checkpoint_bytes(&bad).is_err());
    // Corrupt a byte in data section (after header) to break sha
    let mut corrupt = std::fs::read(&path).unwrap();
    if corrupt.len() > 200 {
        let i = corrupt.len() - 10;
        corrupt[i] ^= 0xff;
        assert!(validate_checkpoint_bytes(&corrupt).is_err());
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn batch_still_works_after_forge_load() {
    use forge_runtime::model::ContinuousBatcher;
    let cfg = ModelConfig::tiny();
    let mut model = Transformer::synthetic(cfg.clone(), 44).unwrap();
    model.quantize_weights(QuantBits::Int8, 32).unwrap();
    let path = tmp_path("batch");
    save_forge(&model, &path, SaveOptions::default()).unwrap();
    let loaded = load_forge_path(&path, LoadOptions::default()).unwrap();
    let params = GenerateParams {
        max_tokens: 3,
        temperature: 0.0,
        top_k: 0,
        eos_token_id: None,
    };
    let mut batcher = ContinuousBatcher::new_untokenized(loaded.model, 64, 2).unwrap();
    let id = batcher.admit_tokens(vec![1, 2, 3], params).unwrap();
    let mut n = 0;
    while batcher.active_count() > 0 {
        for ev in batcher.step().unwrap() {
            assert_eq!(ev.request_id, id);
            n += 1;
        }
    }
    assert_eq!(n, 3);
    let _ = std::fs::remove_file(&path);
}
