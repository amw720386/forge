use super::linear::Linear;
use crate::tensor::ops::silu_inplace;
use crate::tensor::{Tensor, TensorError};

#[derive(Clone, Debug)]
pub struct MlpWeights {
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}

/// SwiGLU MLP: `down(silu(gate(x)) * up(x))`.
#[derive(Clone, Debug)]
pub struct Mlp {
    pub weights: MlpWeights,
}

impl Mlp {
    pub fn forward(
        &self,
        x: &Tensor,
        gate_buf: &mut Tensor,
        up_buf: &mut Tensor,
        out: &mut Tensor,
    ) -> Result<(), TensorError> {
        self.weights.gate.forward_into(x, gate_buf)?;
        self.weights.up.forward_into(x, up_buf)?;
        silu_inplace(gate_buf)?;
        // In-place: gate *= up (buffers may alias in the mathematical sense only).
        {
            let g = gate_buf.data_mut();
            let u = up_buf.data();
            assert_eq!(g.len(), u.len());
            for i in 0..g.len() {
                g[i] *= u[i];
            }
        }
        self.weights.down.forward_into(gate_buf, out)?;
        Ok(())
    }
}
