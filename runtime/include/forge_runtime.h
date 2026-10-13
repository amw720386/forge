#ifndef FORGE_RUNTIME_H
#define FORGE_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef FORGE_RUNTIME_API
#define FORGE_RUNTIME_API
#endif

typedef enum forge_status {
  FORGE_OK = 0,
  FORGE_ERR_INVALID_ARGUMENT = 1,
  FORGE_ERR_DIMENSION_MISMATCH = 2,
  FORGE_ERR_OUT_OF_MEMORY = 3,
  FORGE_ERR_INTERNAL = 4,
  FORGE_ERR_BUSY = 5,
  FORGE_ERR_NOT_FOUND = 6,
  FORGE_ERR_FINISHED = 7
} forge_status;

typedef struct forge_runtime forge_runtime;

FORGE_RUNTIME_API const char* forge_version(void);

FORGE_RUNTIME_API forge_status forge_matmul_f32(size_t m, size_t n, size_t k,
                                                const float* a, size_t lda,
                                                const float* b, size_t ldb,
                                                float* c, size_t ldc);

FORGE_RUNTIME_API forge_status forge_matvec_f32(size_t m, size_t n,
                                                const float* a, size_t lda,
                                                const float* x, float* y);

/**
 * Load model from a Hugging Face directory. Blocking. On failure returns NULL
 * and writes a message into err_buf (if non-NULL).
 * Ownership: caller must forge_runtime_free.
 *
 * The runtime owns a continuous batcher (FORGE_MAX_BATCH, default 4).
 * Single-flight generate_* APIs remain for compatibility; prefer batch_*.
 */
FORGE_RUNTIME_API forge_runtime* forge_runtime_load(const char* model_dir,
                                                    size_t max_seq,
                                                    char* err_buf,
                                                    size_t err_len);

/**
 * device:
 *   0 = CPU
 *   1 = CUDA FP32 weights (when runtime built with CUDA)
 *   2 = CUDA BF16 weights (activations remain FP32)
 */
FORGE_RUNTIME_API forge_runtime* forge_runtime_load_ex(const char* model_dir,
                                                       size_t max_seq,
                                                       int device,
                                                       char* err_buf,
                                                       size_t err_len);

FORGE_RUNTIME_API void forge_runtime_free(forge_runtime* rt);

FORGE_RUNTIME_API size_t forge_batch_capacity(forge_runtime* rt);
FORGE_RUNTIME_API size_t forge_batch_active(forge_runtime* rt);
FORGE_RUNTIME_API size_t forge_batch_free_slots(forge_runtime* rt);

/** scheduler / KV observability (zeros when unsupported). No pointers. */
typedef struct forge_batch_sched_stats {
  uint64_t active_decode;
  uint64_t active_prefill;
  uint64_t last_scheduled_decode;
  uint64_t last_scheduled_prefill;
  uint64_t kv_num_blocks;
  uint64_t kv_block_size;
  uint64_t kv_free_blocks;
  uint64_t kv_allocated_blocks;
  uint64_t prefill_stall_ns;
} forge_batch_sched_stats_t;

FORGE_RUNTIME_API forge_status
forge_batch_sched_stats(forge_runtime* rt, forge_batch_sched_stats_t* out);

/**
 * Admit a prompt into the continuous batcher.
 * CUDA: enqueues Prefilling (chunked); CPU: may still serialize prefill.
 * Returns FORGE_ERR_BUSY when no free slot (backpressure).
 * Returns FORGE_ERR_OUT_OF_MEMORY when GPU KV blocks are exhausted.
 */
FORGE_RUNTIME_API forge_status forge_batch_admit(forge_runtime* rt,
                                                 const char* prompt_utf8,
                                                 int32_t max_tokens,
                                                 int32_t temperature_milli,
                                                 uint64_t* out_request_id,
                                                 char* err_buf, size_t err_len);

FORGE_RUNTIME_API forge_status forge_batch_cancel(forge_runtime* rt,
                                                  uint64_t request_id);

typedef struct forge_batch_event {
  uint64_t request_id;
  uint32_t token_id;
  uint8_t done;
  uint8_t _pad[3];
  char piece[64];
} forge_batch_event;

/**
 * One shared continuous-batching step. Writes up to `cap` events into `out`.
 * `cap` must be at least forge_batch_capacity(rt); smaller buffers are rejected
 * before generation advances. The inline piece is a UTF-8 preview; use
 * forge_batch_piece before the next step to retrieve the complete text.
 * Returns the number of events written (>=0), or a negative forge_status.
 */
FORGE_RUNTIME_API int32_t forge_batch_step(forge_runtime* rt,
                                           forge_batch_event* out, size_t cap);

/**
 * Begin greedy/sampled generation (single-flight compatibility).
 * temperature_milli is temperature*1000 (e.g. 700 → 0.7). <=0 → greedy.
 */
FORGE_RUNTIME_API forge_status forge_generate_begin(
    forge_runtime* rt, const char* prompt_utf8, int32_t max_tokens,
    int32_t temperature_milli, char* err_buf, size_t err_len);

/**
 * Produce the next token. Blocking decode step.
 * Writes UTF-8 piece into piece_buf (may be empty for special tokens).
 * Sets *token_id. Returns FORGE_ERR_FINISHED when no more tokens.
 */
FORGE_RUNTIME_API forge_status forge_generate_next(forge_runtime* rt,
                                                   uint32_t* token_id,
                                                   char* piece_buf,
                                                   size_t piece_len);

FORGE_RUNTIME_API void forge_generate_cancel(forge_runtime* rt);

/* Full last-step piece; returns required bytes including NUL. Short buffers are
 * untouched. Retrieve before the next batch step. Zero means invalid
 * runtime/request. */
FORGE_RUNTIME_API size_t forge_batch_piece(forge_runtime* rt,
                                           uint64_t request_id, char* buf,
                                           size_t len);

/* Render and count with the loaded tokenizer. Required bytes including NUL;
 * short output buffers untouched, zero on error. Caller serializes runtime
 * access. */
FORGE_RUNTIME_API size_t forge_chat_prepare(forge_runtime* rt,
                                            const char* messages_json,
                                            char* out, size_t len,
                                            char* err_buf, size_t err_len);

#ifdef __cplusplus
}
#endif

#endif /* FORGE_RUNTIME_H */
