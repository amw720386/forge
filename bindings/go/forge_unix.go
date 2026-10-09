//go:build unix

package forge

/*
#cgo CFLAGS: -I${SRCDIR}/../../runtime/include
#cgo linux LDFLAGS: -L${SRCDIR}/../../runtime/target/release -lforge_runtime -Wl,-rpath,${SRCDIR}/../../runtime/target/release -ldl -lpthread -lm -lstdc++
#cgo darwin LDFLAGS: -L${SRCDIR}/../../runtime/target/release -lforge_runtime -Wl,-rpath,${SRCDIR}/../../runtime/target/release -lpthread -lm -lc++

#include "forge_runtime.h"
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"unsafe"
)

func nativeVersion() string {
	return C.GoString(C.forge_version())
}

func nativeMatmul(m, n, k int, a, b, c []float32) Status {
	st := C.forge_matmul_f32(
		C.size_t(m), C.size_t(n), C.size_t(k),
		(*C.float)(unsafe.Pointer(&a[0])), C.size_t(k),
		(*C.float)(unsafe.Pointer(&b[0])), C.size_t(n),
		(*C.float)(unsafe.Pointer(&c[0])), C.size_t(n),
	)
	return Status(st)
}

func nativeMatvec(m, n int, a, x, y []float32) Status {
	st := C.forge_matvec_f32(
		C.size_t(m), C.size_t(n),
		(*C.float)(unsafe.Pointer(&a[0])), C.size_t(n),
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&y[0])),
	)
	return Status(st)
}

func nativeLoadEx(modelDir string, maxSeq int, device int) (uintptr, error) {
	cdir := C.CString(modelDir)
	defer C.free(unsafe.Pointer(cdir))
	errBuf := make([]byte, 512)
	rt := C.forge_runtime_load_ex(
		cdir,
		C.size_t(maxSeq),
		C.int(device),
		(*C.char)(unsafe.Pointer(&errBuf[0])),
		C.size_t(len(errBuf)),
	)
	if rt == nil {
		n := 0
		for n < len(errBuf) && errBuf[n] != 0 {
			n++
		}
		return 0, fmt.Errorf("forge_runtime_load_ex: %s", string(errBuf[:n]))
	}
	return uintptr(unsafe.Pointer(rt)), nil
}

func nativeFree(ptr uintptr) {
	C.forge_runtime_free((*C.forge_runtime)(unsafe.Pointer(ptr)))
}

func nativeGenBegin(ptr uintptr, prompt string, maxTokens int, temperatureMilli int32) error {
	cp := C.CString(prompt)
	defer C.free(unsafe.Pointer(cp))
	errBuf := make([]byte, 512)
	st := C.forge_generate_begin(
		(*C.forge_runtime)(unsafe.Pointer(ptr)),
		cp,
		C.int32_t(maxTokens),
		C.int32_t(temperatureMilli),
		(*C.char)(unsafe.Pointer(&errBuf[0])),
		C.size_t(len(errBuf)),
	)
	if Status(st) != StatusOK {
		n := 0
		for n < len(errBuf) && errBuf[n] != 0 {
			n++
		}
		msg := string(errBuf[:n])
		if msg != "" {
			return fmt.Errorf("%w: %s", Status(st), msg)
		}
		return Status(st)
	}
	return nil
}

func nativeGenNext(ptr uintptr) (uint32, string, bool, error) {
	var token C.uint32_t
	pieceBuf := make([]byte, 256)
	st := C.forge_generate_next(
		(*C.forge_runtime)(unsafe.Pointer(ptr)),
		&token,
		(*C.char)(unsafe.Pointer(&pieceBuf[0])),
		C.size_t(len(pieceBuf)),
	)
	if Status(st) == StatusFinished {
		return 0, "", true, nil
	}
	if Status(st) != StatusOK {
		return 0, "", true, Status(st)
	}
	n := 0
	for n < len(pieceBuf) && pieceBuf[n] != 0 {
		n++
	}
	return uint32(token), string(pieceBuf[:n]), false, nil
}

func nativeGenCancel(ptr uintptr) {
	C.forge_generate_cancel((*C.forge_runtime)(unsafe.Pointer(ptr)))
}

func nativeBatchCapacity(ptr uintptr) int {
	return int(C.forge_batch_capacity((*C.forge_runtime)(unsafe.Pointer(ptr))))
}

func nativeBatchActive(ptr uintptr) int {
	return int(C.forge_batch_active((*C.forge_runtime)(unsafe.Pointer(ptr))))
}

func nativeBatchFreeSlots(ptr uintptr) int {
	return int(C.forge_batch_free_slots((*C.forge_runtime)(unsafe.Pointer(ptr))))
}

func nativeBatchAdmit(ptr uintptr, prompt string, maxTokens int, temperatureMilli int32) (uint64, error) {
	cp := C.CString(prompt)
	defer C.free(unsafe.Pointer(cp))
	errBuf := make([]byte, 512)
	var id C.uint64_t
	st := C.forge_batch_admit(
		(*C.forge_runtime)(unsafe.Pointer(ptr)),
		cp,
		C.int32_t(maxTokens),
		C.int32_t(temperatureMilli),
		&id,
		(*C.char)(unsafe.Pointer(&errBuf[0])),
		C.size_t(len(errBuf)),
	)
	if Status(st) != StatusOK {
		n := 0
		for n < len(errBuf) && errBuf[n] != 0 {
			n++
		}
		msg := string(errBuf[:n])
		if msg != "" {
			return 0, fmt.Errorf("%w: %s", Status(st), msg)
		}
		return 0, Status(st)
	}
	return uint64(id), nil
}

func nativeBatchCancel(ptr uintptr, requestID uint64) error {
	st := C.forge_batch_cancel((*C.forge_runtime)(unsafe.Pointer(ptr)), C.uint64_t(requestID))
	if Status(st) != StatusOK {
		return Status(st)
	}
	return nil
}

func nativeBatchStep(ptr uintptr) ([]BatchEvent, error) {
	buf := make([]C.forge_batch_event, nativeBatchCapacity(ptr))
	n := C.forge_batch_step(
		(*C.forge_runtime)(unsafe.Pointer(ptr)),
		&buf[0],
		C.size_t(len(buf)),
	)
	if n < 0 {
		return nil, Status(-n)
	}
	out := make([]BatchEvent, 0, int(n))
	for i := C.int32_t(0); i < n; i++ {
		ev := buf[i]
		size := C.forge_batch_piece((*C.forge_runtime)(unsafe.Pointer(ptr)), ev.request_id, nil, 0)
		if size == 0 {
			return nil, StatusInternal
		}
		data := make([]byte, int(size))
		C.forge_batch_piece((*C.forge_runtime)(unsafe.Pointer(ptr)), ev.request_id, (*C.char)(unsafe.Pointer(&data[0])), size)
		piece := string(data[:len(data)-1])
		out = append(out, BatchEvent{
			RequestID: uint64(ev.request_id),
			TokenID:   uint32(ev.token_id),
			Text:      piece,
			Done:      ev.done != 0,
		})
	}
	return out, nil
}

func nativeBatchSchedStats(ptr uintptr) (BatchSchedStats, error) {
	var st C.forge_batch_sched_stats_t
	rc := C.forge_batch_sched_stats((*C.forge_runtime)(unsafe.Pointer(ptr)), &st)
	if Status(rc) != StatusOK {
		return BatchSchedStats{}, Status(rc)
	}
	return BatchSchedStats{
		ActiveDecode:         uint64(st.active_decode),
		ActivePrefill:        uint64(st.active_prefill),
		LastScheduledDecode:  uint64(st.last_scheduled_decode),
		LastScheduledPrefill: uint64(st.last_scheduled_prefill),
		KvNumBlocks:          uint64(st.kv_num_blocks),
		KvBlockSize:          uint64(st.kv_block_size),
		KvFreeBlocks:         uint64(st.kv_free_blocks),
		KvAllocatedBlocks:    uint64(st.kv_allocated_blocks),
		PrefillStallNs:       uint64(st.prefill_stall_ns),
	}, nil
}
