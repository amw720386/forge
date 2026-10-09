# Forge checkpoint format (`.forge` v1)

Minimal little-endian binary format for Forge weight-only checkpoints.
Inspired by ideas in Safetensors (named tensors, explicit dtypes) and GGUF
(versioned header + typed tensors), but **not** a clone of either.

## Layout

```text
[0, 128)     Fixed header
[meta_off, meta_off+meta_len)   Metadata blob
[data_off, data_off+data_len)   64-byte-aligned tensor payloads
```

### Header (128 bytes)

| Offset | Type | Field |
|---|---|---|
| 0 | `[u8;4]` | Magic `FRG1` |
| 4 | `u32` | `version` = 1 |
| 8 | `u32` | `header_size` = 128 |
| 12 | `u32` | `arch_id` (1 = Qwen2; 2 = supported Llama decoder) |
| 16 | `u32` | `tensor_count` |
| 20 | `u32` | `flags` (bit0 = SHA-256 present) |
| 24 | `u64` | `meta_offset` |
| 32 | `u64` | `meta_length` |
| 40 | `u64` | `data_offset` |
| 48 | `u64` | `data_length` |
| 56 | `[u8;32]` | SHA-256 of metadata blob |
| 88 | `[u8;32]` | SHA-256 of concatenated tensor payloads (no padding) |
| 120 | `[u8;8]` | reserved zero |

### Metadata blob

```text
u32 json_len | utf8 JSON ModelMeta | u32 table_len | tensor table
```

`ModelMeta` JSON fields: name, architecture, hidden/intermediate sizes, layer/head
counts, vocab, RoPE, rms eps, tie flag, `quant_bits`, `quant_group_size`, bos/eos ids.

Phase 16 adds optional `pad_token_id` and `tokenizer_sha256`. Missing optional
fields in existing Qwen checkpoints deserialize to `None`. Architecture is `qwen2`
or the supported `llama` decoder and must agree with the header ID. Unsupported
IDs are rejected. Version/magic/alignment and quantization payloads remain unchanged;
old Qwen files need no migration. Older readers safely reject the new Llama ID.

Family-specific Q/K/V bias presence and all projection/norm/embedding shapes are
validated when constructing the model. Quantized output-head dimensions are checked.
A malformed present output head is not silently replaced with tied embeddings.
Tokenizer contents stay external; new HF-derived exports record the exact tokenizer
SHA-256 and startup checks it. Legacy files without a digest keep their sibling
tokenizer compatibility, without a claim of verified identity.

### Tensor table entry (variable)

```text
u16 name_len | name utf8
u8  ndim (1..8)
u64 dims[ndim]
u8  dtype   (1=F32, 2=I8, 3=I4, 4=F32Scale)
u8  quant   (0=none, 1=weight-only symmetric)
u32 group_size
u64 data_rel_offset   // relative to data_offset
u64 data_nbytes
```

### Payloads

- **F32**: little-endian `f32` row-major.
- **I8**: packed `int8` `[out, in]`.
- **I4**: nibble-packed rows; stride `(in+1)/2` (see [QUANTIZATION.md](QUANTIZATION.md)).
- **F32Scale**: little-endian scales `[out, n_groups]`.

Offsets are validated for overflow, file bounds, and non-overlapping ranges before use.

## Writing

`forge-quantize` / `save_forge` write to `*.forge.tmp` then `rename` for crash safety.

## Loading

`load_forge_path` validates header + SHA-256, builds owned `Tensor` / `QuantMatrix`
buffers (kernels never hold dangling mmap views). Optional `FORGE_MMAP=1` uses
`memmap2` for I/O; data is still copied into Rust-owned storage for FFI lifetime safety.

## Non-goals (v1)

- Tokenizer weights inside the file (tokenizer.json stays beside the model).
- Zero-copy kernel views over mmap (planned once lifetimes are explicit in the ABI).
