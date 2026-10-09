use forge_runtime::model::ForgeTokenizer;

#[test]
fn byte_fallback_stream_matches_complete_decode() {
    use tokenizers::{
        decoders::byte_fallback::ByteFallback, models::wordlevel::WordLevel, AddedToken, Tokenizer,
    };
    let words = [
        "[UNK]", "<0xF0>", "<0x9F>", "<0x98>", "<0x80>", " café", "!", " ", "<eos>",
    ];
    let vocab = words
        .iter()
        .enumerate()
        .map(|(i, w)| (w.to_string(), i as u32))
        .collect();
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut inner = Tokenizer::new(model);
    inner.with_decoder(Some(ByteFallback::default()));
    inner.add_special_tokens(&[AddedToken::from("<eos>", true)]);
    // Construct through the public on-disk loader, without weights.
    let dir = std::env::temp_dir().join(format!("forge-stream-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    inner.save(dir.join("tokenizer.json"), false).unwrap();
    let tok = ForgeTokenizer::from_dir(&dir, 0, 8).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    for ids in [vec![1, 2, 3, 4, 5, 6, 7, 8], vec![1, 2], vec![8], vec![]] {
        let mut emitted = String::new();
        let mut streamed = String::new();
        for end in 1..=ids.len() {
            let piece = tok
                .stream_piece(&ids[..end], &mut emitted, end == ids.len())
                .unwrap();
            if end < 4 && ids.len() > 4 {
                assert!(piece.is_empty());
            }
            streamed.push_str(&piece);
        }
        assert_eq!(streamed, tok.decode(&ids, true).unwrap());
        assert!(!streamed.contains("<eos>"));
    }
}
