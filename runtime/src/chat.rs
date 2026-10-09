//! Local chat interface over the same continuous batcher used by the C ABI.
use crate::ffi::abi::{forge_runtime_load_ex, RuntimeHandle};
use crate::model::{BatchError, BatchEvent, GenerateParams, TokError};
use minijinja::{context, Environment};
use serde::Serialize;
use std::ffi::{CStr, CString};

#[derive(Clone, Debug, Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

pub struct ChatTemplate {
    env: Environment<'static>,
}

impl ChatTemplate {
    pub fn new(source: String) -> Result<Self, minijinja::Error> {
        let mut env = Environment::new();
        env.add_template_owned("chat", source)?;
        Ok(Self { env })
    }

    pub fn render(&self, messages: &[Message]) -> Result<String, minijinja::Error> {
        self.env.get_template("chat")?.render(context! {
            messages => messages, add_generation_prompt => true,
        })
    }
}

impl RuntimeHandle {
    /// Own the runtime with normal Rust drop semantics, sharing the ABI loader.
    pub fn load(path: &str, max_seq: usize, device: i32) -> Result<Box<Self>, String> {
        let path = CString::new(path).map_err(|e| e.to_string())?;
        let mut error = [0i8; 1024];
        // The loader returns an exclusively owned Box allocation on success.
        let ptr = unsafe {
            forge_runtime_load_ex(
                path.as_ptr(),
                max_seq,
                device,
                error.as_mut_ptr().cast(),
                error.len(),
            )
        };
        if ptr.is_null() {
            Err(unsafe { CStr::from_ptr(error.as_ptr().cast()) }
                .to_string_lossy()
                .into_owned())
        } else {
            Ok(unsafe { Box::from_raw(ptr) })
        }
    }

    pub fn chat_template(&self) -> Result<ChatTemplate, String> {
        let path = self.tokenizer_dir.join("tokenizer_config.json");
        let source =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let config: serde_json::Value = serde_json::from_str(&source).map_err(|e| e.to_string())?;
        let template = config
            .get("chat_template")
            .and_then(|v| v.as_str())
            .ok_or("tokenizer_config.json must contain a string chat_template")?;
        ChatTemplate::new(template.to_owned()).map_err(|e| format!("chat template: {e}"))
    }

    pub fn context_size(&self) -> usize {
        self.max_seq
    }

    pub fn count_tokens(&self, prompt: &str) -> Result<usize, TokError> {
        // Identical tokenization to batch admission, including special tokens.
        Ok(self.tokenizer.encode(prompt, true)?.len())
    }

    pub fn decode_tokens(&self, tokens: &[u32]) -> Result<String, TokError> {
        self.tokenizer.decode(tokens, true)
    }

    pub fn chat_begin(&mut self, prompt: &str, limit: usize) -> Result<u64, BatchError> {
        self.inner.admit(
            prompt,
            GenerateParams {
                max_tokens: limit,
                temperature: 0.0,
                top_k: 0,
                eos_token_id: Some(self.eos_token_id),
            },
        )
    }

    pub fn chat_step(&mut self) -> Result<Vec<BatchEvent>, BatchError> {
        self.inner.step()
    }
    pub fn chat_cancel(&mut self, id: u64) -> Result<(), BatchError> {
        self.inner.cancel(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn template_preserves_complete_history_and_generation_prefix() {
        let template = ChatTemplate::new("{% for message in messages %}{{ '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}".into()).unwrap();
        let messages = [
            ("user", "Hello"),
            ("assistant", "Hi"),
            ("user", "Remember?"),
        ]
        .into_iter()
        .map(|(role, content)| Message {
            role: role.into(),
            content: content.into(),
        })
        .collect::<Vec<_>>();
        assert_eq!(template.render(&messages).unwrap(), "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi<|im_end|>\n<|im_start|>user\nRemember?<|im_end|>\n<|im_start|>assistant\n");
    }
}
