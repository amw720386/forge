use super::config::ModelConfig;
use super::linear::Linear;
use super::mlp::MlpWeights;
use crate::attention::{attend, AttentionConfig, AttnScratch};
use crate::cache::KvCache;
use crate::tensor::rows::{add_rows, copy_rows, rmsnorm_rows};
use crate::tensor::{Tensor, TensorError};

#[derive(Clone, Debug)]
pub struct DecoderWeights {
    pub input_layernorm: Tensor,          // [H]
    pub post_attention_layernorm: Tensor, // [H]
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub mlp: MlpWeights,
}

pub struct DecoderBlock {
    pub cfg: ModelConfig,
    pub attn_cfg: AttentionConfig,
    pub weights: DecoderWeights,
}

impl DecoderBlock {
    pub fn new(cfg: ModelConfig, weights: DecoderWeights) -> Result<Self, TensorError> {
        cfg.validate()?;
        let h = cfg.hidden_size;
        let kv = cfg.n_kv_heads * cfg.head_dim();
        for norm in [&weights.input_layernorm, &weights.post_attention_layernorm] {
            if norm.shape() != [h] {
                return Err(TensorError::ShapeMismatch {
                    expected: vec![h],
                    got: norm.shape().to_vec(),
                });
            }
        }
        for (linear, out, input, bias) in [
            (&weights.q_proj, h, h, cfg.architecture.qkv_bias()),
            (&weights.k_proj, kv, h, cfg.architecture.qkv_bias()),
            (&weights.v_proj, kv, h, cfg.architecture.qkv_bias()),
            (&weights.o_proj, h, h, false),
            (&weights.mlp.gate, cfg.intermediate_size, h, false),
            (&weights.mlp.up, cfg.intermediate_size, h, false),
            (&weights.mlp.down, h, cfg.intermediate_size, false),
        ] {
            if linear.weight.out_features() != out || linear.weight.in_features() != input {
                return Err(TensorError::ShapeMismatch {
                    expected: vec![out, input],
                    got: vec![linear.weight.out_features(), linear.weight.in_features()],
                });
            }
            if linear.bias.is_some() != bias
                || linear.bias.as_ref().is_some_and(|b| b.shape() != [out])
            {
                return Err(TensorError::Configuration(
                    "projection bias incompatible with architecture".into(),
                ));
            }
        }
        let attn_cfg = cfg.attention();
        Ok(Self {
            cfg,
            attn_cfg,
            weights,
        })
    }

    /// Forward one block for `t` token rows in `x` (`[max, H]` with valid prefix `t`).
    pub fn forward(
        &self,
        x: &mut Tensor,
        t: usize,
        position_offset: usize,
        cache: &mut KvCache,
        attn: &mut AttnScratch,
        residual: &mut Tensor,
        normed: &mut Tensor,
        attn_out: &mut Tensor,
        mlp_gate: &mut Tensor,
        mlp_up: &mut Tensor,
        mlp_out: &mut Tensor,
    ) -> Result<(), TensorError> {
        let h = self.cfg.hidden_size;
        let i_sz = self.cfg.intermediate_size;
        let q_w = self.cfg.n_heads * self.cfg.head_dim();
        if x.shape()[1] != h || t > x.shape()[0] {
            return Err(TensorError::ShapeMismatch {
                expected: vec![t, h],
                got: x.shape().to_vec(),
            });
        }
        ensure_capacity(residual, t, h)?;
        ensure_capacity(normed, t, h)?;
        ensure_capacity(attn_out, t, h)?;
        ensure_capacity(mlp_gate, t, i_sz)?;
        ensure_capacity(mlp_up, t, i_sz)?;
        ensure_capacity(mlp_out, t, h)?;

        let x_cols = x.shape()[1];
        let normed_cols = normed.shape()[1];
        let attn_q_cols = attn.q.shape()[1];
        let attn_k_cols = attn.k.shape()[1];
        let attn_v_cols = attn.v.shape()[1];
        let attn_ctx_cols = attn.ctx.shape()[1];
        let attn_out_cols = attn_out.shape()[1];
        let mlp_gate_cols = mlp_gate.shape()[1];
        let mlp_up_cols = mlp_up.shape()[1];
        let mlp_out_cols = mlp_out.shape()[1];

        copy_rows(x.data(), t, h, residual.data_mut())?;

        rmsnorm_rows(
            x.data(),
            t,
            h,
            x_cols,
            self.weights.input_layernorm.data(),
            self.cfg.rms_norm_eps,
            normed.data_mut(),
            normed_cols,
        )?;

        self.weights.q_proj.apply_rows(
            normed.data(),
            t,
            h,
            normed_cols,
            attn.q.data_mut(),
            attn_q_cols,
        )?;
        self.weights.k_proj.apply_rows(
            normed.data(),
            t,
            h,
            normed_cols,
            attn.k.data_mut(),
            attn_k_cols,
        )?;
        self.weights.v_proj.apply_rows(
            normed.data(),
            t,
            h,
            normed_cols,
            attn.v.data_mut(),
            attn_v_cols,
        )?;

        attend(&self.attn_cfg, attn, t, cache, position_offset)?;

        self.weights.o_proj.apply_rows(
            attn.ctx.data(),
            t,
            q_w,
            attn_ctx_cols,
            attn_out.data_mut(),
            attn_out_cols,
        )?;

        add_rows(residual.data(), attn_out.data(), t, h, x.data_mut())?;

        copy_rows(x.data(), t, h, residual.data_mut())?;
        rmsnorm_rows(
            x.data(),
            t,
            h,
            x_cols,
            self.weights.post_attention_layernorm.data(),
            self.cfg.rms_norm_eps,
            normed.data_mut(),
            normed_cols,
        )?;

        self.weights.mlp.gate.apply_rows(
            normed.data(),
            t,
            h,
            normed_cols,
            mlp_gate.data_mut(),
            mlp_gate_cols,
        )?;
        self.weights.mlp.up.apply_rows(
            normed.data(),
            t,
            h,
            normed_cols,
            mlp_up.data_mut(),
            mlp_up_cols,
        )?;
        {
            let n = t * i_sz;
            let status = unsafe {
                crate::ffi::kernels::forge_kernels_silu_f32(
                    n,
                    mlp_gate.data().as_ptr(),
                    mlp_gate.data_mut().as_mut_ptr(),
                )
            };
            crate::ffi::kernels::check(status).map_err(TensorError::Kernel)?;
            let status = unsafe {
                crate::ffi::kernels::forge_kernels_mul_f32(
                    n,
                    mlp_gate.data().as_ptr(),
                    mlp_up.data().as_ptr(),
                    mlp_gate.data_mut().as_mut_ptr(),
                )
            };
            crate::ffi::kernels::check(status).map_err(TensorError::Kernel)?;
        }
        self.weights.mlp.down.apply_rows(
            mlp_gate.data(),
            t,
            i_sz,
            mlp_gate_cols,
            mlp_out.data_mut(),
            mlp_out_cols,
        )?;

        add_rows(residual.data(), mlp_out.data(), t, h, x.data_mut())?;
        Ok(())
    }
}

fn ensure_capacity(t: &mut Tensor, rows: usize, cols: usize) -> Result<(), TensorError> {
    if t.shape().len() != 2 || t.shape()[1] != cols || t.shape()[0] < rows {
        *t = Tensor::zeros(&[rows.max(t.shape().get(0).copied().unwrap_or(1)), cols])?;
        if t.shape()[0] < rows {
            *t = Tensor::zeros(&[rows, cols])?;
        }
    }
    Ok(())
}
