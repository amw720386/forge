//! Process memory reporting and inference budgets.
//!
//! Estimates are **conservative planning aids**, not hard OS limits. Windows working-set /
//! commit charge and Linux RSS can diverge from in-process sums; true enforcement needs
//! OS job objects / cgroups.

use crate::model::ModelConfig;
use crate::tensor::TensorError;
use std::env;

/// Default caps sized for ~16–32 GiB developer machines running one backend.
pub const DEFAULT_MAX_SEQ: usize = 1024;
pub const DEFAULT_MAX_BATCH: usize = 4;
pub const DEFAULT_MEM_BUDGET_MIB: u64 = 8192;

#[derive(Clone, Debug)]
pub struct MemoryBudget {
    pub max_seq: usize,
    pub max_batch: usize,
    /// Soft planning budget (MiB). Load fails if the estimate exceeds this.
    pub budget_mib: u64,
    /// Refuse config if estimate > this fraction of detected physical RAM (0 = disabled).
    pub max_ram_fraction: f64,
}

impl Default for MemoryBudget {
    fn default() -> Self {
        Self::from_env()
    }
}

impl MemoryBudget {
    pub fn from_env() -> Self {
        let max_seq = env::var("FORGE_MAX_SEQ")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(DEFAULT_MAX_SEQ);
        let max_batch = env::var("FORGE_MAX_BATCH")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(DEFAULT_MAX_BATCH);
        let budget_mib = env::var("FORGE_MEM_BUDGET_MIB")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_MEM_BUDGET_MIB);
        let max_ram_fraction = env::var("FORGE_MEM_RAM_FRACTION")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.55);
        Self {
            max_seq,
            max_batch,
            budget_mib,
            max_ram_fraction,
        }
    }

    /// Estimate resident bytes for weights + prefill scratch + B KV caches + batch scratch.
    pub fn estimate_runtime_bytes(
        &self,
        cfg: &ModelConfig,
        weight_bytes: usize,
    ) -> Result<u64, TensorError> {
        let s = self.max_seq;
        let b = self.max_batch;
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let nh = cfg.n_heads;
        let kv_w = cfg.n_kv_heads * cfg.head_dim();
        let vocab = cfg.vocab_size;
        let layers = cfg.n_layers;

        // ForwardScratch: 5×[S,H] + 2×[S,I] + logits[1,V] + AttnScratch
        // AttnScratch: q/ctx [S,H], k/v [S,kv], scores [nh,S,S]
        let fwd = checked_mul(5, s)?
            .checked_mul(h)
            .and_then(|x| x.checked_mul(4))
            .ok_or(TensorError::SizeOverflow)?
            + checked_mul(2, s)?
                .checked_mul(i)
                .and_then(|x| x.checked_mul(4))
                .ok_or(TensorError::SizeOverflow)?
            + checked_mul(vocab, 4)?
            + checked_mul(2, s)?
                .checked_mul(h)
                .and_then(|x| x.checked_mul(4))
                .ok_or(TensorError::SizeOverflow)?
            + checked_mul(2, s)?
                .checked_mul(kv_w)
                .and_then(|x| x.checked_mul(4))
                .ok_or(TensorError::SizeOverflow)?
            + checked_mul(nh, s)?
                .checked_mul(s)
                .and_then(|x| x.checked_mul(4))
                .ok_or(TensorError::SizeOverflow)?;

        // KV: L layers × 2 (K,V) × S × kv_w × 4 × B
        let kv = checked_mul(layers, 2)?
            .checked_mul(s)
            .and_then(|x| x.checked_mul(kv_w))
            .and_then(|x| x.checked_mul(4))
            .and_then(|x| x.checked_mul(b))
            .ok_or(TensorError::SizeOverflow)?;

        // BatchScratch approx: B × (7H + 2I + V) × 4
        let batch = checked_mul(b, 7 * h + 2 * i + vocab)?
            .checked_mul(4)
            .ok_or(TensorError::SizeOverflow)?;

        let logits_slots = checked_mul(b, vocab)?
            .checked_mul(4)
            .ok_or(TensorError::SizeOverflow)?;

        let total = (weight_bytes as u64)
            .checked_add(fwd as u64)
            .and_then(|x| x.checked_add(kv as u64))
            .and_then(|x| x.checked_add(batch as u64))
            .and_then(|x| x.checked_add(logits_slots as u64))
            .ok_or(TensorError::SizeOverflow)?;
        // 15% allocator / fragmentation headroom
        Ok(total + total / 7)
    }

    pub fn check_allows(&self, cfg: &ModelConfig, weight_bytes: usize) -> Result<u64, TensorError> {
        let need = self.estimate_runtime_bytes(cfg, weight_bytes)?;
        let need_mib = need / (1024 * 1024);
        if need_mib > self.budget_mib {
            return Err(TensorError::BudgetExceeded {
                need_mib,
                budget_mib: self.budget_mib,
            });
        }
        if self.max_ram_fraction > 0.0 {
            if let Some(ram) = system_total_ram_bytes() {
                let cap = (ram as f64 * self.max_ram_fraction) as u64;
                if need > cap {
                    return Err(TensorError::BudgetExceeded {
                        need_mib,
                        budget_mib: cap / (1024 * 1024),
                    });
                }
            }
        }
        Ok(need)
    }
}

fn checked_mul(a: usize, b: usize) -> Result<usize, TensorError> {
    a.checked_mul(b).ok_or(TensorError::SizeOverflow)
}

/// Snapshot of process memory (best-effort, OS-specific).
#[derive(Clone, Debug, Default)]
pub struct ProcessMemory {
    /// Working set (Windows) / RSS (Linux), bytes.
    pub resident_bytes: u64,
    /// Peak working set / peak RSS when available.
    pub peak_resident_bytes: u64,
    /// Private bytes / commit charge when available (Windows).
    pub committed_bytes: u64,
    /// Soft page faults since process start (Windows `PageFaultCount` / Linux `minflt`).
    pub page_faults: u64,
    /// Hard / major faults when available (Linux `majflt`; Windows not split here).
    pub major_faults: u64,
    pub total_ram_bytes: Option<u64>,
    /// Available physical RAM when the OS reports it.
    pub avail_ram_bytes: Option<u64>,
}

impl ProcessMemory {
    pub fn sample() -> Self {
        let mut s = Self {
            total_ram_bytes: system_total_ram_bytes(),
            avail_ram_bytes: system_avail_ram_bytes(),
            ..Default::default()
        };
        sample_process(&mut s);
        s
    }

    pub fn format_short(&self) -> String {
        let avail = self
            .avail_ram_bytes
            .map(|a| format!(" avail={:.1} GiB", a as f64 / (1024.0 * 1024.0 * 1024.0)))
            .unwrap_or_default();
        format!(
            "RSS={:.1} MiB peak={:.1} MiB commit={:.1} MiB faults={}{}",
            self.resident_bytes as f64 / (1024.0 * 1024.0),
            self.peak_resident_bytes as f64 / (1024.0 * 1024.0),
            self.committed_bytes as f64 / (1024.0 * 1024.0),
            self.page_faults,
            self.total_ram_bytes
                .map(|t| format!(
                    " host_ram={:.1} GiB{avail}",
                    t as f64 / (1024.0 * 1024.0 * 1024.0)
                ))
                .unwrap_or(avail)
        )
    }
}

/// Host free/available physical RAM (planning aid — not a hard limit).
pub fn system_avail_ram_bytes() -> Option<u64> {
    system_avail_ram_bytes_impl()
}

#[cfg(windows)]
fn sample_process(s: &mut ProcessMemory) {
    use std::mem::MaybeUninit;
    #[repr(C)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }
    #[link(name = "psapi")]
    extern "system" {
        fn GetCurrentProcess() -> *mut core::ffi::c_void;
        fn GetProcessMemoryInfo(
            process: *mut core::ffi::c_void,
            counters: *mut ProcessMemoryCounters,
            cb: u32,
        ) -> i32;
    }
    unsafe {
        let mut pmc = MaybeUninit::<ProcessMemoryCounters>::zeroed();
        let ptr = pmc.as_mut_ptr();
        (*ptr).cb = std::mem::size_of::<ProcessMemoryCounters>() as u32;
        if GetProcessMemoryInfo(
            GetCurrentProcess(),
            ptr,
            std::mem::size_of::<ProcessMemoryCounters>() as u32,
        ) != 0
        {
            let c = pmc.assume_init();
            s.resident_bytes = c.working_set_size as u64;
            s.peak_resident_bytes = c.peak_working_set_size as u64;
            s.committed_bytes = c.pagefile_usage as u64;
            s.page_faults = c.page_fault_count as u64;
            // Windows aggregates soft+hard in PageFaultCount; major_faults left 0.
        }
    }
}

#[cfg(target_os = "linux")]
fn sample_process(s: &mut ProcessMemory) {
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if let Some(v) = line.strip_prefix("VmRSS:") {
                s.resident_bytes = parse_kb(v) * 1024;
            } else if let Some(v) = line.strip_prefix("VmHWM:") {
                s.peak_resident_bytes = parse_kb(v) * 1024;
            } else if let Some(v) = line.strip_prefix("VmSize:") {
                s.committed_bytes = parse_kb(v) * 1024;
            }
        }
    }
    if let Ok(stat) = std::fs::read_to_string("/proc/self/stat") {
        // fields: ... 10=minflt 12=majflt (1-indexed field 10 and 12 after comm)
        if let Some(after_comm) = stat.rsplit(')').next() {
            let parts: Vec<&str> = after_comm.split_whitespace().collect();
            // after ")": state is [0], minflt is [7], majflt is [9]
            if parts.len() > 9 {
                s.page_faults = parts[7].parse().unwrap_or(0);
                s.major_faults = parts[9].parse().unwrap_or(0);
            }
        }
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
fn sample_process(_s: &mut ProcessMemory) {}

#[cfg(target_os = "linux")]
fn parse_kb(s: &str) -> u64 {
    s.split_whitespace()
        .next()
        .and_then(|x| x.parse().ok())
        .unwrap_or(0)
}

#[cfg(windows)]
fn memory_status_ex() -> Option<(u64, u64)> {
    #[repr(C)]
    struct MemoryStatusEx {
        dw_length: u32,
        dw_memory_load: u32,
        ull_total_phys: u64,
        ull_avail_phys: u64,
        ull_total_page_file: u64,
        ull_avail_page_file: u64,
        ull_total_virtual: u64,
        ull_avail_virtual: u64,
        ull_avail_extended_virtual: u64,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GlobalMemoryStatusEx(stat: *mut MemoryStatusEx) -> i32;
    }
    unsafe {
        let mut st = std::mem::zeroed::<MemoryStatusEx>();
        st.dw_length = std::mem::size_of::<MemoryStatusEx>() as u32;
        if GlobalMemoryStatusEx(&mut st) != 0 {
            Some((st.ull_total_phys, st.ull_avail_phys))
        } else {
            None
        }
    }
}

#[cfg(windows)]
fn system_total_ram_bytes() -> Option<u64> {
    memory_status_ex().map(|(t, _)| t)
}

#[cfg(windows)]
fn system_avail_ram_bytes_impl() -> Option<u64> {
    memory_status_ex().map(|(_, a)| a)
}

#[cfg(target_os = "linux")]
fn system_total_ram_bytes() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in meminfo.lines() {
        if let Some(v) = line.strip_prefix("MemTotal:") {
            return Some(parse_kb(v) * 1024);
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn system_avail_ram_bytes_impl() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in meminfo.lines() {
        if let Some(v) = line.strip_prefix("MemAvailable:") {
            return Some(parse_kb(v) * 1024);
        }
    }
    None
}

#[cfg(not(any(windows, target_os = "linux")))]
fn system_total_ram_bytes() -> Option<u64> {
    None
}

#[cfg(not(any(windows, target_os = "linux")))]
fn system_avail_ram_bytes_impl() -> Option<u64> {
    None
}

/// Abort if available physical RAM is below a conservative floor (MiB).
pub fn require_available_ram_mib(min_avail_mib: u64) -> Result<(), String> {
    match system_avail_ram_bytes() {
        Some(avail) => {
            let avail_mib = avail / (1024 * 1024);
            if avail_mib < min_avail_mib {
                Err(format!(
                    "available RAM {avail_mib} MiB < required {min_avail_mib} MiB; \
                     refusing bench to avoid paging thrash"
                ))
            } else {
                Ok(())
            }
        }
        None => Ok(()), // unknown — rely on FORGE_BENCH_MEM_CAP_MIB only
    }
}

/// Refuse high-risk multi-backend benches when host RAM is low.
pub fn allow_multi_model_bench(required_mib: u64) -> Result<(), String> {
    let budget = env::var("FORGE_BENCH_MEM_CAP_MIB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(6144u64);
    if required_mib > budget {
        return Err(format!(
            "bench would need ~{required_mib} MiB peak but FORGE_BENCH_MEM_CAP_MIB={budget}; \
             load one backend per process or raise the cap explicitly"
        ));
    }
    if let Some(ram) = system_total_ram_bytes() {
        let frac = env::var("FORGE_BENCH_RAM_FRACTION")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.45);
        let cap = ((ram as f64) * frac) as u64 / (1024 * 1024);
        if required_mib > cap {
            return Err(format!(
                "bench estimate {required_mib} MiB exceeds {:.0}% of host RAM ({cap} MiB). \
                 Refusing to avoid thrashing. Set FORGE_BENCH_RAM_FRACTION or run single-backend.",
                frac * 100.0
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelConfig;

    #[test]
    fn estimate_qwen_reasonable() {
        let cfg = ModelConfig::qwen25_0_5b();
        let b = MemoryBudget {
            max_seq: 1024,
            max_batch: 4,
            budget_mib: 8192,
            max_ram_fraction: 0.0,
        };
        let e = b.estimate_runtime_bytes(&cfg, 1020 * 1024 * 1024).unwrap();
        let mib = e / (1024 * 1024);
        assert!(mib > 1000 && mib < 5000, "unexpected estimate {mib} MiB");
    }
}
