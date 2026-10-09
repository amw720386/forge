package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	forge "github.com/amw720386/forge/bindings/go"

	"github.com/amw720386/forge/api/internal/metrics"
	"github.com/amw720386/forge/api/internal/queue"
	"github.com/amw720386/forge/api/internal/scheduler"
	"github.com/amw720386/forge/api/internal/server"
)

func main() {
	addr := flag.String("addr", "127.0.0.1:8080", "listen address")
	modelDir := flag.String("model", "models/Qwen2.5-0.5B-Instruct", "HF model directory or .forge checkpoint")
	maxSeq := flag.Int("max-seq", 1024, "max sequence length (also capped by FORGE_MAX_SEQ)")
	queueCap := flag.Int("queue", 16, "admission queue capacity")
	device := flag.String("device", "cpu", "execution device: cpu or cuda (cuda requires CUDA runtime build)")
	dtype := flag.String("dtype", "fp32", "weight dtype: fp32 (cpu), fp32 or bf16 (cuda)")
	flag.Parse()
	if flag.NArg() != 0 || *maxSeq <= 0 || *queueCap <= 0 {
		log.Fatal("use named flags with positive -max-seq and -queue values")
	}
	if *device == "cpu" && *dtype != "fp32" {
		log.Fatal("CPU requires -dtype fp32; quantization is selected by checkpoint")
	}
	if *dtype != "fp32" && *dtype != "bf16" {
		log.Fatalf("unknown --dtype %q (use fp32 or bf16)", *dtype)
	}
	if _, err := os.Stat(*modelDir); err != nil {
		log.Fatalf("model path %q: %v", *modelDir, err)
	}

	if dir := os.Getenv("FORGE_RUNTIME_DIR"); dir == "" {
		// Default DLL search next to release build when running from repo root.
		cand := filepath.Join("runtime", "target", "release")
		if _, err := os.Stat(filepath.Join(cand, "forge_runtime.dll")); err == nil {
			_ = os.Setenv("FORGE_RUNTIME_DIR", cand)
		} else if executable, err := os.Executable(); err == nil {
			cand = filepath.Join(filepath.Dir(executable), "..", "runtime", "target", "release")
			if _, err := os.Stat(filepath.Join(cand, "forge_runtime.dll")); err == nil {
				_ = os.Setenv("FORGE_RUNTIME_DIR", cand)
			}
		}
	}

	log.Printf("forge-runtime %s", forge.Version())
	dev := forge.DeviceCPU
	if *device == "cuda" {
		switch *dtype {
		case "bf16":
			dev = forge.DeviceCUDABF16
		case "fp32":
			dev = forge.DeviceCUDA
		default:
			log.Fatalf("unknown --dtype %q (use fp32 or bf16)", *dtype)
		}
	} else if *device != "cpu" {
		log.Fatalf("unknown --device %q (use cpu or cuda)", *device)
	}
	log.Printf("loading model from %s (device=%s dtype=%s) ...", *modelDir, *device, *dtype)
	rt, err := forge.LoadDevice(*modelDir, *maxSeq, dev)
	if err != nil {
		log.Fatalf("load model: %v", err)
	}
	defer rt.Close()
	cap := rt.BatchCapacity()
	log.Printf("model loaded (batch_capacity=%d free_slots=%d)", cap, rt.BatchFreeSlots())
	if *device == "cuda" && *queueCap < cap {
		log.Printf("note: queue=%d < CUDA batch capacity=%d; raising queue to capacity", *queueCap, cap)
		*queueCap = cap
	}

	m := metrics.New()
	q := queue.New(*queueCap)
	sched := scheduler.New(rt, q, m)
	srv := server.New(server.Config{
		Addr:          *addr,
		QueueCap:      *queueCap,
		DefaultMaxTok: 64,
	}, q, sched, m)

	if err := srv.Start(); err != nil {
		log.Fatalf("start: %v", err)
	}

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt, syscall.SIGTERM)
	<-sig
	log.Printf("shutting down...")
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		log.Printf("shutdown: %v", err)
	}
}
