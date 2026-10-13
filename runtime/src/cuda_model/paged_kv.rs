//! Paged GPU KV-cache pool and per-request block tables.
//!
//! Physical blocks are fixed-size token slabs shared across requests. Each
//! request owns a logical→physical block table. Contiguous `GpuKvCache` remains
//! the reference path for single-request CUDA.

use crate::compute::{Buffer, ComputeBackend, ComputeError};
use std::env;

/// Default tokens per physical KV block.
pub const DEFAULT_BLOCK_SIZE: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PagedKvError {
    Exhausted { need: usize, free: usize },
    Overflow { len: usize, max_seq: usize },
    InvalidBlock { logical: usize, table_len: usize },
    DoubleFree(u32),
    ArithmeticOverflow(&'static str),
    Backend(ComputeError),
}

impl std::fmt::Display for PagedKvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exhausted { need, free } => {
                write!(f, "GPU KV block pool exhausted: need {need}, free {free}")
            }
            Self::Overflow { len, max_seq } => {
                write!(f, "paged KV overflow: len={len} max_seq={max_seq}")
            }
            Self::InvalidBlock { logical, table_len } => {
                write!(
                    f,
                    "invalid block table entry: logical={logical} table_len={table_len}"
                )
            }
            Self::DoubleFree(id) => write!(f, "double-free of physical block {id}"),
            Self::ArithmeticOverflow(what) => write!(f, "allocation size overflow: {what}"),
            Self::Backend(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for PagedKvError {}

impl From<ComputeError> for PagedKvError {
    fn from(e: ComputeError) -> Self {
        Self::Backend(e)
    }
}

/// Host-side free-list metadata (no device pointers). Unit-testable without CUDA.
#[derive(Debug, Clone)]
pub struct BlockAllocator {
    pub block_size: usize,
    pub num_blocks: usize,
    free: Vec<u32>,
    /// true = currently owned by some session (guards double-free).
    owned: Vec<bool>,
    allocated_count: usize,
}

impl BlockAllocator {
    pub fn new(block_size: usize, num_blocks: usize) -> Result<Self, PagedKvError> {
        if block_size == 0 || num_blocks == 0 {
            return Err(PagedKvError::ArithmeticOverflow("block_size/num_blocks"));
        }
        if num_blocks > u32::MAX as usize {
            return Err(PagedKvError::ArithmeticOverflow("num_blocks > u32::MAX"));
        }
        let mut free = Vec::with_capacity(num_blocks);
        for i in (0..num_blocks as u32).rev() {
            free.push(i);
        }
        Ok(Self {
            block_size,
            num_blocks,
            free,
            owned: vec![false; num_blocks],
            allocated_count: 0,
        })
    }

    pub fn free_count(&self) -> usize {
        self.free.len()
    }

    pub fn allocated_count(&self) -> usize {
        self.allocated_count
    }

    pub fn utilization(&self) -> f64 {
        if self.num_blocks == 0 {
            return 0.0;
        }
        self.allocated_count as f64 / self.num_blocks as f64
    }

    pub fn blocks_for_tokens(&self, tokens: usize) -> Result<usize, PagedKvError> {
        if tokens == 0 {
            return Ok(0);
        }
        tokens
            .checked_add(self.block_size - 1)
            .map(|x| x / self.block_size)
            .ok_or(PagedKvError::ArithmeticOverflow("blocks_for_tokens"))
    }

    pub fn alloc_one(&mut self) -> Result<u32, PagedKvError> {
        let id = self
            .free
            .pop()
            .ok_or(PagedKvError::Exhausted { need: 1, free: 0 })?;
        let idx = id as usize;
        if idx >= self.owned.len() || self.owned[idx] {
            return Err(PagedKvError::DoubleFree(id));
        }
        self.owned[idx] = true;
        self.allocated_count += 1;
        Ok(id)
    }

    pub fn alloc_n(&mut self, n: usize) -> Result<Vec<u32>, PagedKvError> {
        if n > self.free.len() {
            return Err(PagedKvError::Exhausted {
                need: n,
                free: self.free.len(),
            });
        }
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.alloc_one()?);
        }
        Ok(out)
    }

    pub fn free_one(&mut self, id: u32) -> Result<(), PagedKvError> {
        let idx = id as usize;
        if idx >= self.owned.len() {
            return Err(PagedKvError::InvalidBlock {
                logical: idx,
                table_len: self.owned.len(),
            });
        }
        if !self.owned[idx] {
            return Err(PagedKvError::DoubleFree(id));
        }
        self.owned[idx] = false;
        self.allocated_count = self.allocated_count.saturating_sub(1);
        self.free.push(id);
        Ok(())
    }

    pub fn free_many(&mut self, ids: &[u32]) -> Result<(), PagedKvError> {
        for &id in ids {
            self.free_one(id)?;
        }
        Ok(())
    }
}

/// Shared device KV block pool (all layers).
pub struct GpuKvBlockPool {
    pub n_layers: usize,
    pub kv_width: usize,
    pub block_size: usize,
    pub num_blocks: usize,
    pub max_seq: usize,
    alloc: BlockAllocator,
    k_pool: Buffer,
    v_pool: Buffer,
}

impl GpuKvBlockPool {
    pub fn bytes_per_block(
        n_layers: usize,
        block_size: usize,
        kv_width: usize,
    ) -> Result<usize, PagedKvError> {
        n_layers
            .checked_mul(2)
            .and_then(|x| x.checked_mul(block_size))
            .and_then(|x| x.checked_mul(kv_width))
            .and_then(|x| x.checked_mul(4))
            .ok_or(PagedKvError::ArithmeticOverflow("bytes_per_block"))
    }

    pub fn pool_bytes(
        n_layers: usize,
        num_blocks: usize,
        block_size: usize,
        kv_width: usize,
    ) -> Result<usize, PagedKvError> {
        // One of K or V: n_layers * num_blocks * block_size * kv_width * 4
        n_layers
            .checked_mul(num_blocks)
            .and_then(|x| x.checked_mul(block_size))
            .and_then(|x| x.checked_mul(kv_width))
            .and_then(|x| x.checked_mul(4))
            .ok_or(PagedKvError::ArithmeticOverflow("pool_bytes"))
    }

    pub fn resolve_num_blocks(
        max_batch: usize,
        max_seq: usize,
        block_size: usize,
    ) -> Result<usize, PagedKvError> {
        if let Ok(s) = env::var("FORGE_KV_NUM_BLOCKS") {
            if let Ok(n) = s.parse::<usize>() {
                if n >= 1 {
                    return Ok(n);
                }
            }
        }
        let per_seq = max_seq
            .checked_add(block_size - 1)
            .map(|x| x / block_size)
            .ok_or(PagedKvError::ArithmeticOverflow("per_seq blocks"))?;
        max_batch
            .checked_mul(per_seq)
            .ok_or(PagedKvError::ArithmeticOverflow("num_blocks"))
    }

    pub fn resolve_block_size() -> usize {
        env::var("FORGE_KV_BLOCK_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(DEFAULT_BLOCK_SIZE)
    }

    pub fn new(
        backend: &dyn ComputeBackend,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_batch: usize,
    ) -> Result<Self, PagedKvError> {
        let block_size = Self::resolve_block_size();
        let kv_width = n_kv_heads
            .checked_mul(head_dim)
            .ok_or(PagedKvError::ArithmeticOverflow("kv_width"))?;
        let num_blocks = Self::resolve_num_blocks(max_batch, max_seq, block_size)?;
        let pool_bytes = Self::pool_bytes(n_layers, num_blocks, block_size, kv_width)?;
        let alloc = BlockAllocator::new(block_size, num_blocks)?;
        Ok(Self {
            n_layers,
            kv_width,
            block_size,
            num_blocks,
            max_seq,
            alloc,
            k_pool: backend.alloc(pool_bytes)?,
            v_pool: backend.alloc(pool_bytes)?,
        })
    }

    pub fn free_count(&self) -> usize {
        self.alloc.free_count()
    }

    pub fn allocated_count(&self) -> usize {
        self.alloc.allocated_count()
    }

    pub fn utilization(&self) -> f64 {
        self.alloc.utilization()
    }

    pub fn k_pool(&self) -> &Buffer {
        &self.k_pool
    }

    pub fn v_pool(&self) -> &Buffer {
        &self.v_pool
    }

    pub fn k_pool_mut(&mut self) -> &mut Buffer {
        &mut self.k_pool
    }

    pub fn v_pool_mut(&mut self) -> &mut Buffer {
        &mut self.v_pool
    }

    /// Borrow both pool buffers mutably (for paged append).
    pub fn pools_mut(&mut self) -> (&mut Buffer, &mut Buffer) {
        (&mut self.k_pool, &mut self.v_pool)
    }

    /// Element offset for layer L, physical block P, row r within the block.
    pub fn row_offset(
        &self,
        layer: usize,
        physical: u32,
        row: usize,
    ) -> Result<usize, PagedKvError> {
        if layer >= self.n_layers || row >= self.block_size {
            return Err(PagedKvError::InvalidBlock {
                logical: layer,
                table_len: self.n_layers,
            });
        }
        let p = physical as usize;
        if p >= self.num_blocks {
            return Err(PagedKvError::InvalidBlock {
                logical: p,
                table_len: self.num_blocks,
            });
        }
        // ((L * num_blocks + P) * block_size + r) * kv_width
        layer
            .checked_mul(self.num_blocks)
            .and_then(|x| x.checked_add(p))
            .and_then(|x| x.checked_mul(self.block_size))
            .and_then(|x| x.checked_add(row))
            .and_then(|x| x.checked_mul(self.kv_width))
            .ok_or(PagedKvError::ArithmeticOverflow("row_offset"))
    }

    pub fn ensure_capacity_tokens(&self, tokens: usize) -> Result<(), PagedKvError> {
        let need = self.alloc.blocks_for_tokens(tokens)?;
        if need > self.alloc.free_count() {
            return Err(PagedKvError::Exhausted {
                need,
                free: self.alloc.free_count(),
            });
        }
        Ok(())
    }
}

/// Per-request paged KV metadata + device block-table mirror.
pub struct PagedKvSession {
    pub block_table: Vec<u32>,
    pub len: usize,
    pub max_seq: usize,
    /// Device u32[max_blocks_per_seq] for attention.
    device_table: Buffer,
    table_dirty: bool,
}

impl PagedKvSession {
    pub fn new(
        backend: &dyn ComputeBackend,
        max_seq: usize,
        block_size: usize,
    ) -> Result<Self, PagedKvError> {
        let max_blocks = max_seq
            .checked_add(block_size - 1)
            .map(|x| x / block_size)
            .ok_or(PagedKvError::ArithmeticOverflow("max_blocks"))?
            .max(1);
        let bytes = max_blocks
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or(PagedKvError::ArithmeticOverflow("device_table"))?;
        Ok(Self {
            block_table: Vec::new(),
            len: 0,
            max_seq,
            device_table: backend.alloc(bytes)?,
            table_dirty: true,
        })
    }

    pub fn clear(&mut self, pool: &mut GpuKvBlockPool) -> Result<(), PagedKvError> {
        pool.alloc.free_many(&self.block_table)?;
        self.block_table.clear();
        self.len = 0;
        self.table_dirty = true;
        Ok(())
    }

    /// Ensure block_table covers `self.len + additional` tokens; allocate as needed.
    pub fn grow_for(
        &mut self,
        pool: &mut GpuKvBlockPool,
        additional: usize,
    ) -> Result<(), PagedKvError> {
        let new_len = self
            .len
            .checked_add(additional)
            .ok_or(PagedKvError::ArithmeticOverflow("grow_for"))?;
        if new_len > self.max_seq {
            return Err(PagedKvError::Overflow {
                len: new_len,
                max_seq: self.max_seq,
            });
        }
        let need_blocks = if new_len == 0 {
            0
        } else {
            pool.alloc.blocks_for_tokens(new_len)?
        };
        while self.block_table.len() < need_blocks {
            let id = pool.alloc.alloc_one()?;
            self.block_table.push(id);
            self.table_dirty = true;
        }
        Ok(())
    }

    pub fn sync_device_table(&mut self, backend: &dyn ComputeBackend) -> Result<(), PagedKvError> {
        if !self.table_dirty && !self.block_table.is_empty() {
            return Ok(());
        }
        // Upload as little-endian u32 bytes.
        let mut bytes = Vec::with_capacity(self.block_table.len() * 4);
        for &id in &self.block_table {
            bytes.extend_from_slice(&id.to_le_bytes());
        }
        // Zero-pad remainder so stale ids are never read past table_len.
        let cap = self.device_table.bytes() / 4;
        while bytes.len() < cap * 4 {
            bytes.extend_from_slice(&0u32.to_le_bytes());
        }
        backend.copy_h2d(&mut self.device_table, &bytes)?;
        self.table_dirty = false;
        Ok(())
    }

    pub fn device_table(&self) -> &Buffer {
        &self.device_table
    }

    pub fn num_logical_blocks(&self) -> usize {
        self.block_table.len()
    }
}

/// Snapshot for metrics / benches (no pointers).
#[derive(Debug, Clone, Default)]
pub struct PagedKvStats {
    pub num_blocks: usize,
    pub block_size: usize,
    pub free_blocks: usize,
    pub allocated_blocks: usize,
    pub utilization: f64,
}

impl GpuKvBlockPool {
    pub fn stats(&self) -> PagedKvStats {
        PagedKvStats {
            num_blocks: self.num_blocks,
            block_size: self.block_size,
            free_blocks: self.free_count(),
            allocated_blocks: self.allocated_count(),
            utilization: self.utilization(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_and_free_roundtrip() {
        let mut a = BlockAllocator::new(16, 4).unwrap();
        assert_eq!(a.free_count(), 4);
        let ids = a.alloc_n(3).unwrap();
        assert_eq!(ids.len(), 3);
        assert_eq!(a.free_count(), 1);
        assert_eq!(a.allocated_count(), 3);
        a.free_many(&ids).unwrap();
        assert_eq!(a.free_count(), 4);
        assert_eq!(a.allocated_count(), 0);
    }

    #[test]
    fn exhaustion() {
        let mut a = BlockAllocator::new(16, 2).unwrap();
        a.alloc_n(2).unwrap();
        assert!(matches!(a.alloc_one(), Err(PagedKvError::Exhausted { .. })));
    }

    #[test]
    fn double_free_detected() {
        let mut a = BlockAllocator::new(16, 2).unwrap();
        let id = a.alloc_one().unwrap();
        a.free_one(id).unwrap();
        assert!(matches!(a.free_one(id), Err(PagedKvError::DoubleFree(_))));
    }

    #[test]
    fn blocks_for_tokens() {
        let a = BlockAllocator::new(16, 100).unwrap();
        assert_eq!(a.blocks_for_tokens(0).unwrap(), 0);
        assert_eq!(a.blocks_for_tokens(1).unwrap(), 1);
        assert_eq!(a.blocks_for_tokens(16).unwrap(), 1);
        assert_eq!(a.blocks_for_tokens(17).unwrap(), 2);
        assert_eq!(a.blocks_for_tokens(33).unwrap(), 3);
    }

    #[test]
    fn bytes_per_block_qwen() {
        // 24 layers, block 16, kv_width 128 → 384 KiB
        let b = GpuKvBlockPool::bytes_per_block(24, 16, 128).unwrap();
        assert_eq!(b, 24 * 2 * 16 * 128 * 4);
        assert_eq!(b, 384 * 1024);
    }

    #[test]
    fn grow_across_boundaries() {
        let mut a = BlockAllocator::new(16, 8).unwrap();
        // Simulate session growth without device.
        let mut table: Vec<u32> = Vec::new();
        let mut len = 0usize;
        let add = |table: &mut Vec<u32>, a: &mut BlockAllocator, len: &mut usize, n: usize| {
            let new_len = *len + n;
            let need = a.blocks_for_tokens(new_len).unwrap();
            while table.len() < need {
                table.push(a.alloc_one().unwrap());
            }
            *len = new_len;
        };
        add(&mut table, &mut a, &mut len, 16);
        assert_eq!(table.len(), 1);
        add(&mut table, &mut a, &mut len, 1);
        assert_eq!(table.len(), 2);
        assert_eq!(len, 17);
        a.free_many(&table).unwrap();
        assert_eq!(a.free_count(), 8);
    }
}
