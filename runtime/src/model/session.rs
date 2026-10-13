//! Streaming generation session (single in-flight request).

use super::generate::{GenError, GenerateParams};
use super::tokenizer::ForgeTokenizer;
use super::transformer::{ForwardScratch, LayerKvCaches, Transformer};
use crate::sampling::{greedy, sample_temperature, sample_top_k};

pub struct StreamSession {
    model: Transformer,
    tok: ForgeTokenizer,
    max_seq: usize,
    caches: LayerKvCaches,
    scratch: ForwardScratch,
    tokens: Vec<u32>,
    prompt_len: usize,
    emitted: String,
    params: GenerateParams,
    started: bool,
    finished: bool,
    rng_state: u64,
    last_logits: Vec<f32>,
}

impl StreamSession {
    pub fn new(model: Transformer, tok: ForgeTokenizer, max_seq: usize) -> Result<Self, GenError> {
        let caches = LayerKvCaches::new(&model.cfg, max_seq)?;
        let scratch = ForwardScratch::new(&model.cfg, max_seq)?;
        Ok(Self {
            model,
            tok,
            max_seq,
            caches,
            scratch,
            tokens: Vec::new(),
            prompt_len: 0,
            emitted: String::new(),
            params: GenerateParams::default(),
            started: false,
            finished: true,
            rng_state: 0xDEADBEEF,
            last_logits: Vec::new(),
        })
    }

    pub fn begin(&mut self, prompt: &str, params: GenerateParams) -> Result<(), GenError> {
        self.caches.clear();
        self.tokens = self.tok.encode(prompt, true)?;
        if self.tokens.is_empty() || self.tokens.len() >= self.max_seq {
            return Err(GenError::PromptTooLong);
        }
        self.prompt_len = self.tokens.len();
        self.emitted.clear();
        self.params = params;
        self.model
            .forward_tokens(&self.tokens, 0, &mut self.caches, &mut self.scratch)?;
        let vocab = self.model.cfg.vocab_size;
        self.last_logits = self.scratch.logits.data()[..vocab].to_vec();
        self.started = true;
        self.finished = false;
        Ok(())
    }

    /// Produce one new token. Returns `None` when the stream is finished.
    pub fn next_token(&mut self) -> Result<Option<(u32, String)>, GenError> {
        if self.finished || !self.started {
            return Ok(None);
        }
        let generated = self.tokens.len() - self.prompt_len;
        if generated >= self.params.max_tokens {
            self.finished = true;
            return Ok(None);
        }

        let logits = self.last_logits.clone();
        let next = self.sample(&logits);
        self.tokens.push(next);
        let done = self.params.eos_token_id == Some(next)
            || self.tokens.len() >= self.max_seq
            || self.tokens.len() - self.prompt_len >= self.params.max_tokens;
        let piece =
            self.tok
                .stream_piece(&self.tokens[self.prompt_len..], &mut self.emitted, done)?;

        if done {
            self.finished = true;
            return Ok(Some((next, piece)));
        }

        let pos = self.tokens.len() - 1;
        self.model
            .forward_tokens(&[next], pos, &mut self.caches, &mut self.scratch)?;
        let vocab = self.model.cfg.vocab_size;
        self.last_logits = self.scratch.logits.data()[..vocab].to_vec();
        Ok(Some((next, piece)))
    }

    pub fn cancel(&mut self) {
        self.finished = true;
    }

    pub fn finished(&self) -> bool {
        self.finished
    }

    fn sample(&mut self, logits: &[f32]) -> u32 {
        if self.params.temperature <= 0.0 {
            greedy(logits)
        } else if self.params.top_k > 0 {
            sample_top_k(logits, self.params.top_k, self.params.temperature, || {
                rng_f32(&mut self.rng_state)
            })
        } else {
            sample_temperature(logits, self.params.temperature, || {
                rng_f32(&mut self.rng_state)
            })
        }
    }
}

fn rng_f32(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 32) as f32) / (u32::MAX as f32)
}
