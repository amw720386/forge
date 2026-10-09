use forge_runtime::checkpoint::{load_forge_path, save_forge, LoadOptions, SaveOptions};
use forge_runtime::model::*;

fn config() -> serde_json::Value {
    serde_json::json!({"model_type":"llama","hidden_act":"silu","hidden_size":32,"intermediate_size":64,
        "num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"vocab_size":128,
        "max_position_embeddings":128,"rms_norm_eps":1e-5,"rope_theta":100000,
        "tie_word_embeddings":true,"bos_token_id":1,"eos_token_id":2,"attention_bias":false})
}
fn llama() -> Transformer {
    let original = Transformer::synthetic(ModelConfig::tiny(), 17).unwrap();
    let mut layers: Vec<_> = original.layers.iter().map(|l| l.weights.clone()).collect();
    for layer in &mut layers {
        layer.q_proj.bias = None;
        layer.k_proj.bias = None;
        layer.v_proj.bias = None;
    }
    Transformer::from_weights(
        parse_hf_config(&config().to_string()).unwrap(),
        TransformerWeights {
            embed: original.embed,
            layers,
            final_norm: original.final_norm,
            lm_head: None,
        },
    )
    .unwrap()
}
#[test]
fn detects_families_and_rejects_unsupported() {
    let c = config();
    assert_eq!(
        parse_hf_config(&c.to_string()).unwrap().architecture,
        Architecture::Llama
    );
    for (key, value) in [
        ("model_type", serde_json::json!("mistral")),
        ("hidden_act", serde_json::json!("gelu")),
        ("num_attention_heads", serde_json::json!(0)),
        ("num_key_value_heads", serde_json::json!(3)),
        (
            "rope_scaling",
            serde_json::json!({"type":"linear","factor":2}),
        ),
        ("mlp_bias", serde_json::json!(true)),
        ("rope_interleaved", serde_json::json!(true)),
        ("eos_token_id", serde_json::json!(999)),
    ] {
        let mut v = c.clone();
        v[key] = value;
        assert!(parse_hf_config(&v.to_string()).is_err(), "{key}");
    }
    let mut q = c.clone();
    q["model_type"] = serde_json::json!("qwen2");
    q["attention_bias"] = serde_json::json!(true);
    assert_eq!(
        parse_hf_config(&q.to_string()).unwrap().architecture,
        Architecture::Qwen2
    );
    let mut missing = c;
    missing.as_object_mut().unwrap().remove("rope_theta");
    assert!(parse_hf_config(&missing.to_string()).is_err());
}
#[test]
fn llama_cached_and_quantized_checkpoint_roundtrips() {
    for bits in [None, Some(QuantBits::Int8), Some(QuantBits::Int4)] {
        let mut model = llama();
        if let Some(b) = bits {
            model.quantize_weights(b, 8).unwrap();
        }
        let tokens = [3, 4, 5, 6];
        let mut caches = LayerKvCaches::new(&model.cfg, 16).unwrap();
        let mut scratch = ForwardScratch::new(&model.cfg, 16).unwrap();
        model
            .forward_tokens(&tokens, 0, &mut caches, &mut scratch)
            .unwrap();
        let expected = scratch.logits.data().to_vec();
        caches.clear();
        let mut decode = ForwardScratch::new_decode(&model.cfg).unwrap();
        for (pos, &t) in tokens.iter().enumerate() {
            model
                .forward_tokens(&[t], pos, &mut caches, &mut decode)
                .unwrap();
        }
        for (&a, &b) in expected.iter().zip(decode.logits.data()) {
            assert!((a - b).abs() < 1e-5);
        }
        let path = std::env::temp_dir().join(format!(
            "forge-phase16-{}-{bits:?}.forge",
            std::process::id()
        ));
        save_forge(&model, &path, SaveOptions::default()).unwrap();
        let loaded = load_forge_path(&path, LoadOptions { memory_map: true }).unwrap();
        assert_eq!(loaded.model.cfg.architecture, Architecture::Llama);
        assert_eq!(loaded.meta.eos_token_id, 2);
        caches.clear();
        loaded
            .model
            .forward_tokens(&tokens, 0, &mut caches, &mut scratch)
            .unwrap();
        assert_eq!(expected, scratch.logits.data());
        std::fs::remove_file(path).unwrap();
    }
}
#[test]
fn graph_identity_tracks_execution_properties() {
    let cfg = ModelConfig::tiny();
    let original = GraphModelIdentity::new(&cfg, ExecutionPrecision::CudaFp32, 1);
    let mut changed = cfg.clone();
    changed.architecture = Architecture::Llama;
    assert_ne!(
        original,
        GraphModelIdentity::new(&changed, ExecutionPrecision::CudaFp32, 1)
    );
    changed = cfg.clone();
    changed.rope_theta = 100000.0;
    assert_ne!(
        original,
        GraphModelIdentity::new(&changed, ExecutionPrecision::CudaFp32, 1)
    );
    assert_ne!(
        original,
        GraphModelIdentity::new(&cfg, ExecutionPrecision::CudaBf16, 1)
    );
    assert_ne!(
        original,
        GraphModelIdentity::new(&cfg, ExecutionPrecision::CudaFp32, 2)
    );
    let key = DecodeGraphKey {
        model: original,
        batch: 4,
        kv_block_size: 16,
        kv_blocks: 32,
        allocations: [1, 2, 3, 4, 5],
        workspace: [0; 19],
        reuse_bf16: false,
        gemm_lt: true,
        partitioned_attention: false,
        paged_decode_fp32_kv: true,
    };
    let mut different = key.clone();
    different.partitioned_attention = true;
    assert_ne!(key, different);
    different = key.clone();
    different.allocations[2] = 6;
    assert_ne!(key, different);
    different = key.clone();
    different.batch = 8;
    assert_ne!(key, different);
}
#[test]
fn precision_preserves_fp32_kv_and_conservative_dispatch() {
    for p in [
        ExecutionPrecision::CpuFp32,
        ExecutionPrecision::CpuInt8,
        ExecutionPrecision::CpuInt4,
        ExecutionPrecision::CudaFp32,
        ExecutionPrecision::CudaBf16,
    ] {
        assert_eq!(p.contract().kv, ScalarType::Fp32);
        assert_eq!(p.contract().normalization, ScalarType::Fp32);
    }
    assert!(!ExecutionPrecision::CudaBf16.partitioned_attention(8, &[513; 8]));
    assert!(ExecutionPrecision::CudaBf16.partitioned_attention(8, &[512; 8]));
    assert!(!ExecutionPrecision::CudaFp32.partitioned_attention(1, &[127]));
}

#[test]
fn mixed_safetensors_and_precise_loading_errors() {
    use safetensors::{
        tensor::{serialize_to_file, TensorView},
        Dtype,
    };
    let dir = std::env::temp_dir().join(format!("forge-phase16-hf-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), config().to_string()).unwrap();
    let cfg = parse_hf_config(&config().to_string()).unwrap();
    let tensors: Vec<_> = cfg
        .architecture
        .tensor_shapes(&cfg)
        .into_iter()
        .enumerate()
        .map(|(index, (name, shape))| {
            let dtype = if index % 3 == 0 {
                Dtype::BF16
            } else if index % 3 == 1 {
                Dtype::F16
            } else {
                Dtype::F32
            };
            let width = if dtype == Dtype::F32 { 4 } else { 2 };
            let data = vec![0u8; shape.iter().product::<usize>() * width];
            (name, shape, dtype, data)
        })
        .collect();
    let views: Vec<_> = tensors
        .iter()
        .map(|(n, s, d, b)| (n.clone(), TensorView::new(*d, s.clone(), b).unwrap()))
        .collect();
    serialize_to_file(views, &None, &dir.join("model.safetensors")).unwrap();
    let model = load_hf_dir(&dir).unwrap();
    assert_eq!(model.cfg.architecture, Architecture::Llama);
    assert!(model.layers[0].weights.q_proj.bias.is_none());
    drop(model);
    let views: Vec<_> = tensors
        .iter()
        .skip(1)
        .map(|(n, s, d, b)| (n.clone(), TensorView::new(*d, s.clone(), b).unwrap()))
        .collect();
    serialize_to_file(views, &None, &dir.join("model.safetensors")).unwrap();
    let error = load_hf_dir(&dir).err().unwrap().to_string();
    assert!(error.contains("model.embed_tokens.weight"));
    let views: Vec<_> = tensors
        .iter()
        .map(|(n, s, d, b)| {
            let shape = if n == "model.norm.weight" {
                vec![1, 32]
            } else {
                s.clone()
            };
            (n.clone(), TensorView::new(*d, shape, b).unwrap())
        })
        .collect();
    serialize_to_file(views, &None, &dir.join("model.safetensors")).unwrap();
    assert!(load_hf_dir(&dir)
        .err()
        .unwrap()
        .to_string()
        .contains("model.norm.weight: expected"));
    std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();
    let mut bad = cfg;
    bad.tokenizer_sha256 = Some("deadbeef".into());
    assert!(ForgeTokenizer::for_model(&dir, &bad)
        .unwrap_err()
        .to_string()
        .contains("identity"));
    for name in ["config.json", "model.safetensors", "tokenizer.json"] {
        std::fs::remove_file(dir.join(name)).unwrap();
    }
    std::fs::remove_dir(dir).unwrap();
}

#[test]
fn rejects_overflowing_position_and_incompatible_caches() {
    let model = llama();
    let mut caches = LayerKvCaches::new(&model.cfg, 16).unwrap();
    let mut scratch = ForwardScratch::new_decode(&model.cfg).unwrap();
    assert!(model
        .forward_tokens(&[3], usize::MAX, &mut caches, &mut scratch)
        .is_err());
    caches.layers.pop();
    assert!(model
        .forward_tokens(&[3], 0, &mut caches, &mut scratch)
        .is_err());
}
