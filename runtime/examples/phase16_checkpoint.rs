//! One complete model at a time: quantize, serialize, drop, reload, compare.
use forge_runtime::checkpoint::{save_forge, SaveOptions};
use forge_runtime::model::{
    load_model_path, ForgeTokenizer, ForwardScratch, LayerKvCaches, QuantBits,
};
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let mut model = load_model_path(&args[1]).unwrap();
    if let Some(bits) = args.get(4).and_then(|x| x.parse::<u8>().ok()) {
        model
            .quantize_weights(QuantBits::from_u8(bits).unwrap(), 64)
            .unwrap();
    }
    let tok = ForgeTokenizer::for_model(&args[3], &model.cfg).unwrap();
    let tokens = tok
        .encode(
            "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
            false,
        )
        .unwrap();
    let mut kv = LayerKvCaches::new(&model.cfg, 128).unwrap();
    let mut scratch = ForwardScratch::new(&model.cfg, 128).unwrap();
    model
        .forward_tokens(&tokens, 0, &mut kv, &mut scratch)
        .unwrap();
    let expected = scratch.logits.data().to_vec();
    save_forge(&model, &args[2], SaveOptions::default()).unwrap();
    drop(model);
    let reloaded = load_model_path(&args[2]).unwrap();
    ForgeTokenizer::for_model(&args[3], &reloaded.cfg).unwrap();
    kv.clear();
    reloaded
        .forward_tokens(&tokens, 0, &mut kv, &mut scratch)
        .unwrap();
    assert_eq!(expected, scratch.logits.data());
    let mut output = vec![];
    for step in 0..12 {
        let next = forge_runtime::sampling::greedy(scratch.logits.data());
        output.push(next);
        if next == reloaded.cfg.eos_token_id {
            break;
        }
        reloaded
            .forward_tokens(&[next], tokens.len() + step, &mut kv, &mut scratch)
            .unwrap();
    }
    println!(
        "CHECKPOINT_EXACT_ROUNDTRIP architecture={} weights_mib={:.2} ids={output:?} text={}",
        reloaded.cfg.architecture.name(),
        reloaded.weight_nbytes() as f64 / 1048576.0,
        tok.decode(&output, true).unwrap()
    );
}
