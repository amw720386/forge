use forge_runtime::chat::{ChatTemplate, Message};

#[test]
fn qualified_templates_match_hugging_face_multiturn_prompts() {
    let cases: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/chat_templates.json")).unwrap();
    for case in cases.as_array().unwrap() {
        let messages = case["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| Message {
                role: m["role"].as_str().unwrap().into(),
                content: m["content"].as_str().unwrap().into(),
            })
            .collect::<Vec<_>>();
        let template = ChatTemplate::new(case["template"].as_str().unwrap().into()).unwrap();
        assert_eq!(
            template.render(&messages).unwrap(),
            case["formatted"].as_str().unwrap(),
            "{}",
            case["model"]
        );
    }
}

#[test]
#[ignore = "requires local qualified model directories; run sequentially"]
fn pretrained_chat_tokens_limits_eos_and_cancellation() {
    use forge_runtime::ffi::abi::RuntimeHandle;
    let cases: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/chat_templates.json")).unwrap();
    for case in cases.as_array().unwrap() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../models")
            .join(case["model"].as_str().unwrap());
        let mut runtime = RuntimeHandle::load(path.to_str().unwrap(), 256, 0).unwrap();
        let messages = case["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| Message {
                role: m["role"].as_str().unwrap().into(),
                content: m["content"].as_str().unwrap().into(),
            })
            .collect::<Vec<_>>();
        let prompt = runtime.chat_template().unwrap().render(&messages).unwrap();
        assert_eq!(prompt, case["formatted"].as_str().unwrap());
        assert_eq!(
            runtime.count_tokens(&prompt).unwrap(),
            case["tokens"].as_array().unwrap().len()
        );
        // A one-token completion must free its slot and include the final token.
        runtime.chat_begin(&prompt, 1).unwrap();
        let events = runtime.chat_step().unwrap();
        assert_eq!(events.len(), 1);
        assert!(events[0].done);
        // Cancel mid-request, then admit again using the same runtime.
        let id = runtime.chat_begin(&prompt, 64).unwrap();
        runtime.chat_step().unwrap();
        runtime.chat_cancel(id).unwrap();
        runtime.chat_begin(&prompt, 128).unwrap();
        let mut output = Vec::new();
        let mut streamed = String::new();
        loop {
            let events = runtime.chat_step().unwrap();
            output.extend(events.iter().map(|e| e.token_id));
            for event in &events {
                streamed.push_str(&event.piece);
            }
            if events.iter().any(|e| e.done) {
                break;
            }
        }
        assert!(output.len() <= 128);
        assert_eq!(streamed, runtime.decode_tokens(&output).unwrap());
        // The CLI's Rust interface and the server's ABI must sample identical IDs.
        use forge_runtime::ffi::abi::{
            forge_batch_admit, forge_batch_piece, forge_batch_step, ForgeBatchEvent,
        };
        let c_prompt = std::ffi::CString::new(prompt.as_str()).unwrap();
        let mut id = 0;
        let mut error = [0i8; 1024];
        let ptr = &mut *runtime as *mut RuntimeHandle;
        let mut abi_ids = Vec::new();
        let mut abi_text = String::new();
        unsafe {
            assert_eq!(
                forge_batch_admit(
                    ptr,
                    c_prompt.as_ptr(),
                    128,
                    0,
                    &mut id,
                    error.as_mut_ptr(),
                    error.len()
                ),
                0
            );
            loop {
                let mut events: [ForgeBatchEvent; 64] = std::array::from_fn(|_| std::mem::zeroed());
                let n = forge_batch_step(ptr, events.as_mut_ptr(), events.len());
                assert!(n > 0);
                for event in &events[..n as usize] {
                    abi_ids.push(event.token_id);
                    let size = forge_batch_piece(ptr, event.request_id, std::ptr::null_mut(), 0);
                    assert!(size > 0);
                    let mut piece = vec![0u8; size];
                    assert_eq!(
                        forge_batch_piece(ptr, event.request_id, piece.as_mut_ptr().cast(), size),
                        size
                    );
                    abi_text.push_str(std::str::from_utf8(&piece[..size - 1]).unwrap());
                }
                if events[..n as usize].iter().any(|e| e.done != 0) {
                    break;
                }
            }
        }
        assert_eq!(abi_ids, output);
        assert_eq!(abi_text, streamed);
        let config: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path.join("config.json")).unwrap())
                .unwrap();
        assert_eq!(
            *output.last().unwrap() as u64,
            config["eos_token_id"].as_u64().unwrap()
        );
        // The complete formatted prompt is checked, not just the latest turn.
        let long = prompt.repeat(256);
        assert!(runtime.count_tokens(&long).unwrap() >= runtime.context_size());
        assert!(runtime.chat_begin(&long, 1).is_err());
    }
}
