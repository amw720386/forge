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
        loop {
            let events = runtime.chat_step().unwrap();
            output.extend(events.iter().map(|e| e.token_id));
            if events.iter().any(|e| e.done) {
                break;
            }
        }
        assert!(output.len() <= 128);
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
