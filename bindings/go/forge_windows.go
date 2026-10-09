//go:build windows

package forge

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"syscall"
	"unsafe"
)

var (
	loadOnce sync.Once
	loadErr  error

	dll               *syscall.DLL
	procVersion       *syscall.Proc
	procMatmul        *syscall.Proc
	procMatvec        *syscall.Proc
	procRuntimeLoad   *syscall.Proc
	procRuntimeLoadEx *syscall.Proc
	procRuntimeFree   *syscall.Proc
	procGenBegin         *syscall.Proc
	procGenNext          *syscall.Proc
	procGenCancel        *syscall.Proc
	procBatchCapacity    *syscall.Proc
	procBatchActive      *syscall.Proc
	procBatchFreeSlots   *syscall.Proc
	procBatchAdmit       *syscall.Proc
	procBatchCancel      *syscall.Proc
	procBatchStep        *syscall.Proc
	procBatchSchedStats  *syscall.Proc
)

func ensureLoaded() error {
	loadOnce.Do(func() {
		name := "forge_runtime.dll"
		candidates := []string{name}
		if root := os.Getenv("FORGE_RUNTIME_DIR"); root != "" {
			candidates = append([]string{filepath.Join(root, name)}, candidates...)
		}
		candidates = append(candidates,
			filepath.Join("..", "..", "runtime", "target", "release", name),
			filepath.Join("runtime", "target", "release", name),
		)
		var last error
		for _, path := range candidates {
			d, err := syscall.LoadDLL(path)
			if err != nil {
				last = err
				continue
			}
			dll = d
			break
		}
		if dll == nil {
			loadErr = fmt.Errorf("load %s: %w (set FORGE_RUNTIME_DIR)", name, last)
			return
		}
		procs := []struct {
			name string
			dst  **syscall.Proc
		}{
			{"forge_version", &procVersion},
			{"forge_matmul_f32", &procMatmul},
			{"forge_matvec_f32", &procMatvec},
			{"forge_runtime_load", &procRuntimeLoad},
			{"forge_runtime_load_ex", &procRuntimeLoadEx},
			{"forge_runtime_free", &procRuntimeFree},
			{"forge_generate_begin", &procGenBegin},
			{"forge_generate_next", &procGenNext},
			{"forge_generate_cancel", &procGenCancel},
			{"forge_batch_capacity", &procBatchCapacity},
			{"forge_batch_active", &procBatchActive},
			{"forge_batch_free_slots", &procBatchFreeSlots},
			{"forge_batch_admit", &procBatchAdmit},
			{"forge_batch_cancel", &procBatchCancel},
			{"forge_batch_step", &procBatchStep},
			{"forge_batch_sched_stats", &procBatchSchedStats},
		}
		for _, p := range procs {
			*p.dst, loadErr = dll.FindProc(p.name)
			if loadErr != nil {
				return
			}
		}
	})
	return loadErr
}

func nativeVersion() string {
	if err := ensureLoaded(); err != nil {
		return ""
	}
	r, _, _ := procVersion.Call()
	return windowsCString(r)
}

func nativeMatmul(m, n, k int, a, b, c []float32) Status {
	if err := ensureLoaded(); err != nil {
		return StatusInternal
	}
	r, _, _ := procMatmul.Call(
		uintptr(m), uintptr(n), uintptr(k),
		uintptr(unsafe.Pointer(&a[0])), uintptr(k),
		uintptr(unsafe.Pointer(&b[0])), uintptr(n),
		uintptr(unsafe.Pointer(&c[0])), uintptr(n),
	)
	return Status(r)
}

func nativeMatvec(m, n int, a, x, y []float32) Status {
	if err := ensureLoaded(); err != nil {
		return StatusInternal
	}
	r, _, _ := procMatvec.Call(
		uintptr(m), uintptr(n),
		uintptr(unsafe.Pointer(&a[0])), uintptr(n),
		uintptr(unsafe.Pointer(&x[0])),
		uintptr(unsafe.Pointer(&y[0])),
	)
	return Status(r)
}

func nativeLoadEx(modelDir string, maxSeq int, device int) (uintptr, error) {
	if err := ensureLoaded(); err != nil {
		return 0, err
	}
	dir, err := syscall.BytePtrFromString(modelDir)
	if err != nil {
		return 0, err
	}
	errBuf := make([]byte, 512)
	r, _, _ := procRuntimeLoadEx.Call(
		uintptr(unsafe.Pointer(dir)),
		uintptr(maxSeq),
		uintptr(device),
		uintptr(unsafe.Pointer(&errBuf[0])),
		uintptr(len(errBuf)),
	)
	if r == 0 {
		return 0, fmt.Errorf("forge_runtime_load_ex: %s", cStringFromBytes(errBuf))
	}
	return r, nil
}

func nativeFree(ptr uintptr) {
	if err := ensureLoaded(); err != nil {
		return
	}
	procRuntimeFree.Call(ptr)
}

func nativeGenBegin(ptr uintptr, prompt string, maxTokens int, temperatureMilli int32) error {
	p, err := syscall.BytePtrFromString(prompt)
	if err != nil {
		return err
	}
	errBuf := make([]byte, 512)
	r, _, _ := procGenBegin.Call(
		ptr,
		uintptr(unsafe.Pointer(p)),
		uintptr(int32(maxTokens)),
		uintptr(temperatureMilli),
		uintptr(unsafe.Pointer(&errBuf[0])),
		uintptr(len(errBuf)),
	)
	st := Status(r)
	if st != StatusOK {
		msg := cStringFromBytes(errBuf)
		if msg != "" {
			return fmt.Errorf("%w: %s", st, msg)
		}
		return st
	}
	return nil
}

func nativeGenNext(ptr uintptr) (uint32, string, bool, error) {
	var token uint32
	pieceBuf := make([]byte, 256)
	r, _, _ := procGenNext.Call(
		ptr,
		uintptr(unsafe.Pointer(&token)),
		uintptr(unsafe.Pointer(&pieceBuf[0])),
		uintptr(len(pieceBuf)),
	)
	st := Status(r)
	if st == StatusFinished {
		return 0, "", true, nil
	}
	if st != StatusOK {
		return 0, "", true, st
	}
	return token, cStringFromBytes(pieceBuf), false, nil
}

func nativeGenCancel(ptr uintptr) {
	procGenCancel.Call(ptr)
}

func nativeBatchCapacity(ptr uintptr) int {
	r, _, _ := procBatchCapacity.Call(ptr)
	return int(r)
}

func nativeBatchActive(ptr uintptr) int {
	r, _, _ := procBatchActive.Call(ptr)
	return int(r)
}

func nativeBatchFreeSlots(ptr uintptr) int {
	r, _, _ := procBatchFreeSlots.Call(ptr)
	return int(r)
}

type cBatchEvent struct {
	RequestID uint64
	TokenID   uint32
	Done      uint8
	_pad      [3]byte
	Piece     [64]byte
}

func nativeBatchAdmit(ptr uintptr, prompt string, maxTokens int, temperatureMilli int32) (uint64, error) {
	p, err := syscall.BytePtrFromString(prompt)
	if err != nil {
		return 0, err
	}
	errBuf := make([]byte, 512)
	var id uint64
	r, _, _ := procBatchAdmit.Call(
		ptr,
		uintptr(unsafe.Pointer(p)),
		uintptr(int32(maxTokens)),
		uintptr(temperatureMilli),
		uintptr(unsafe.Pointer(&id)),
		uintptr(unsafe.Pointer(&errBuf[0])),
		uintptr(len(errBuf)),
	)
	st := Status(r)
	if st != StatusOK {
		msg := cStringFromBytes(errBuf)
		if msg != "" {
			return 0, fmt.Errorf("%w: %s", st, msg)
		}
		return 0, st
	}
	return id, nil
}

func nativeBatchCancel(ptr uintptr, requestID uint64) error {
	r, _, _ := procBatchCancel.Call(ptr, uintptr(requestID))
	st := Status(r)
	if st != StatusOK {
		return st
	}
	return nil
}

func nativeBatchStep(ptr uintptr) ([]BatchEvent, error) {
	buf := make([]cBatchEvent, 64)
	r, _, _ := procBatchStep.Call(ptr, uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)))
	n := int32(r)
	if n < 0 {
		return nil, Status(-n)
	}
	out := make([]BatchEvent, 0, n)
	for i := int32(0); i < n; i++ {
		ev := buf[i]
		out = append(out, BatchEvent{
			RequestID: ev.RequestID,
			TokenID:   ev.TokenID,
			Text:      cStringFromBytes(ev.Piece[:]),
			Done:      ev.Done != 0,
		})
	}
	return out, nil
}

type cBatchSchedStats struct {
	ActiveDecode         uint64
	ActivePrefill        uint64
	LastScheduledDecode  uint64
	LastScheduledPrefill uint64
	KvNumBlocks          uint64
	KvBlockSize          uint64
	KvFreeBlocks         uint64
	KvAllocatedBlocks    uint64
	PrefillStallNs       uint64
}

func nativeBatchSchedStats(ptr uintptr) (BatchSchedStats, error) {
	if err := ensureLoaded(); err != nil {
		return BatchSchedStats{}, err
	}
	var st cBatchSchedStats
	r, _, _ := procBatchSchedStats.Call(ptr, uintptr(unsafe.Pointer(&st)))
	if Status(r) != StatusOK {
		return BatchSchedStats{}, Status(r)
	}
	return BatchSchedStats{
		ActiveDecode:         st.ActiveDecode,
		ActivePrefill:        st.ActivePrefill,
		LastScheduledDecode:  st.LastScheduledDecode,
		LastScheduledPrefill: st.LastScheduledPrefill,
		KvNumBlocks:          st.KvNumBlocks,
		KvBlockSize:          st.KvBlockSize,
		KvFreeBlocks:         st.KvFreeBlocks,
		KvAllocatedBlocks:    st.KvAllocatedBlocks,
		PrefillStallNs:       st.PrefillStallNs,
	}, nil
}

func windowsCString(ptr uintptr) string {
	if ptr == 0 {
		return ""
	}
	var bytes []byte
	for {
		b := *(*byte)(unsafe.Pointer(ptr + uintptr(len(bytes))))
		if b == 0 {
			break
		}
		bytes = append(bytes, b)
	}
	return string(bytes)
}

func cStringFromBytes(b []byte) string {
	for i, c := range b {
		if c == 0 {
			return string(b[:i])
		}
	}
	return string(b)
}
