use crate::compute::{BackendKind, Buffer, ComputeBackend, ComputeError};
use crate::model::{Linear, Transformer, WeightStorage};
use crate::tensor::Tensor;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuWeightDType {
    Fp32,
    Bf16,
}

pub struct GpuLinear {
    pub out: usize,
    pub in_features: usize,
    pub dtype: GpuWeightDType,
    weight: Buffer,
    bias: Option<Buffer>,
}

impl GpuLinear {
    pub(crate) fn apply_precast(
        &self,
        backend: &crate::compute::CudaBackend,
        x: &Buffer,
        rows: usize,
        y: &mut Buffer,
        y_cols: usize,
    ) -> Result<(), ComputeError> {
        backend.linear_bf16_xw(
            rows,
            self.out,
            self.in_features,
            x,
            &self.weight,
            self.bias.as_ref(),
            y,
            y_cols,
        )
    }

    pub fn upload_dtype(
        backend: &dyn ComputeBackend,
        linear: &Linear,
        dtype: GpuWeightDType,
    ) -> Result<Self, ComputeError> {
        let w_tensor = match &linear.weight {
            WeightStorage::Fp32(w) => w,
            WeightStorage::Quant(_) => {
                return Err(ComputeError::InvalidArgument(
                    "CUDA backend requires FP32 weights (GPU quantization unsupported)".into(),
                ));
            }
        };
        let (out, k) = (w_tensor.shape()[0], w_tensor.shape()[1]);
        let weight = upload_weight_matrix(backend, w_tensor, out, k, dtype)?;
        let bias = if let Some(ref b) = linear.bias {
            let mut buf = backend.alloc(out * 4)?;
            backend.copy_h2d(&mut buf, bytemuck_cast(b.data()))?;
            Some(buf)
        } else {
            None
        };
        Ok(Self {
            out,
            in_features: k,
            dtype,
            weight,
            bias,
        })
    }

    pub fn apply(
        &self,
        backend: &dyn ComputeBackend,
        x: &Buffer,
        rows: usize,
        x_cols: usize,
        y: &mut Buffer,
        y_cols: usize,
    ) -> Result<(), ComputeError> {
        match self.dtype {
            GpuWeightDType::Fp32 => backend.linear_f32(
                rows,
                self.out,
                self.in_features,
                x,
                x_cols,
                &self.weight,
                self.in_features,
                self.bias.as_ref(),
                y,
                y_cols,
            ),
            GpuWeightDType::Bf16 => backend.linear_bf16_w(
                rows,
                self.out,
                self.in_features,
                x,
                x_cols,
                &self.weight,
                self.in_features,
                self.bias.as_ref(),
                y,
                y_cols,
            ),
        }
    }
}

pub struct GpuMlp {
    pub gate: GpuLinear,
    pub up: GpuLinear,
    pub down: GpuLinear,
}

pub struct GpuDecoderWeights {
    pub input_layernorm: Buffer,
    pub post_attention_layernorm: Buffer,
    pub q_proj: GpuLinear,
    pub k_proj: GpuLinear,
    pub v_proj: GpuLinear,
    pub o_proj: GpuLinear,
    pub mlp: GpuMlp,
}

pub struct GpuModelWeights {
    pub embed: Buffer,
    pub final_norm: Buffer,
    /// None when the output projection uses the embedding allocation.
    pub lm_head: Option<GpuLinear>,
    pub layers: Vec<GpuDecoderWeights>,
    pub weight_bytes: usize,
}

fn upload_tensor(backend: &dyn ComputeBackend, t: &Tensor) -> Result<Buffer, ComputeError> {
    let bytes = t.data().len() * 4;
    let mut buf = backend.alloc(bytes)?;
    backend.copy_h2d(&mut buf, bytemuck_cast(t.data()))?;
    Ok(buf)
}

/// Upload a weight matrix as FP32 or convert host FP32 → device BF16 via temp buffer.
fn upload_weight_matrix(
    backend: &dyn ComputeBackend,
    t: &Tensor,
    rows: usize,
    cols: usize,
    dtype: GpuWeightDType,
) -> Result<Buffer, ComputeError> {
    let n = rows * cols;
    if t.data().len() < n {
        return Err(ComputeError::InvalidArgument(
            "weight tensor too small".into(),
        ));
    }
    match dtype {
        GpuWeightDType::Fp32 => {
            let mut weight = backend.alloc(n * 4)?;
            backend.copy_h2d(&mut weight, bytemuck_cast(&t.data()[..n]))?;
            Ok(weight)
        }
        GpuWeightDType::Bf16 => {
            let mut fp32_tmp = backend.alloc(n * 4)?;
            backend.copy_h2d(&mut fp32_tmp, bytemuck_cast(&t.data()[..n]))?;
            let mut bf16 = backend.alloc(n * 2)?;
            backend.f32_to_bf16(n, &fp32_tmp, &mut bf16)?;
            drop(fp32_tmp);
            Ok(bf16)
        }
    }
}

fn bytemuck_cast(data: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) }
}

/// FP32 upload path (wrapper around [`upload_model_dtype`]).
pub fn upload_model(
    backend: &dyn ComputeBackend,
    model: &Transformer,
) -> Result<GpuModelWeights, ComputeError> {
    upload_model_dtype(backend, model, GpuWeightDType::Fp32)
}

pub fn upload_model_dtype(
    backend: &dyn ComputeBackend,
    model: &Transformer,
    dtype: GpuWeightDType,
) -> Result<GpuModelWeights, ComputeError> {
    if backend.kind() != BackendKind::Cuda {
        return Err(ComputeError::InvalidArgument(
            "CUDA backend required".into(),
        ));
    }
    let mut weight_bytes = 0usize;

    // Embeddings follow weight dtype; norms stay FP32.
    let embed = upload_weight_matrix(
        backend,
        &model.embed,
        model.cfg.vocab_size,
        model.cfg.hidden_size,
        dtype,
    )?;
    weight_bytes += embed.bytes;
    let final_norm = upload_tensor(backend, &model.final_norm)?;
    weight_bytes += final_norm.bytes;

    let lm_shares_embed = match &model.lm_head.weight {
        WeightStorage::Fp32(t) => std::ptr::eq(t.data().as_ptr(), model.embed.data().as_ptr()),
        WeightStorage::Quant(_) => false,
    };
    let lm_head = if lm_shares_embed {
        None
    } else {
        let g = GpuLinear::upload_dtype(backend, &model.lm_head, dtype)?;
        weight_bytes += g.weight.bytes + g.bias.as_ref().map(|b| b.bytes).unwrap_or(0);
        Some(g)
    };

    let mut layers = Vec::with_capacity(model.layers.len());
    for layer in &model.layers {
        let w = &layer.weights;
        let gw = GpuDecoderWeights {
            input_layernorm: upload_tensor(backend, &w.input_layernorm)?,
            post_attention_layernorm: upload_tensor(backend, &w.post_attention_layernorm)?,
            q_proj: GpuLinear::upload_dtype(backend, &w.q_proj, dtype)?,
            k_proj: GpuLinear::upload_dtype(backend, &w.k_proj, dtype)?,
            v_proj: GpuLinear::upload_dtype(backend, &w.v_proj, dtype)?,
            o_proj: GpuLinear::upload_dtype(backend, &w.o_proj, dtype)?,
            mlp: GpuMlp {
                gate: GpuLinear::upload_dtype(backend, &w.mlp.gate, dtype)?,
                up: GpuLinear::upload_dtype(backend, &w.mlp.up, dtype)?,
                down: GpuLinear::upload_dtype(backend, &w.mlp.down, dtype)?,
            },
        };
        weight_bytes += gw.input_layernorm.bytes
            + gw.post_attention_layernorm.bytes
            + gw.q_proj.weight.bytes
            + gw.k_proj.weight.bytes
            + gw.v_proj.weight.bytes
            + gw.o_proj.weight.bytes
            + gw.mlp.gate.weight.bytes
            + gw.mlp.up.weight.bytes
            + gw.mlp.down.weight.bytes
            + gw.q_proj.bias.as_ref().map(|b| b.bytes).unwrap_or(0)
            + gw.k_proj.bias.as_ref().map(|b| b.bytes).unwrap_or(0)
            + gw.v_proj.bias.as_ref().map(|b| b.bytes).unwrap_or(0)
            + gw.o_proj.bias.as_ref().map(|b| b.bytes).unwrap_or(0)
            + gw.mlp.gate.bias.as_ref().map(|b| b.bytes).unwrap_or(0)
            + gw.mlp.up.bias.as_ref().map(|b| b.bytes).unwrap_or(0)
            + gw.mlp.down.bias.as_ref().map(|b| b.bytes).unwrap_or(0);
        layers.push(gw);
    }

    Ok(GpuModelWeights {
        embed,
        final_norm,
        lm_head,
        layers,
        weight_bytes,
    })
}
