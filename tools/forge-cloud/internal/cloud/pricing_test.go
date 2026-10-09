package cloud

import (
	"testing"
	"time"
)

func TestEstimateL4Session(t *testing.T) {
	rate, est, err := EstimateSession("l4", 20*time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	if rate <= 0 || est <= 0 {
		t.Fatalf("expected positive rate/est, got %v %v", rate, est)
	}
	// 20/60 * rate
	want := rate * (20.0 / 60.0)
	if est < want*0.99 || est > want*1.01 {
		t.Fatalf("est=%v want≈%v", est, want)
	}
}

func TestResolveMachineL4(t *testing.T) {
	m, err := ResolveMachine("l4")
	if err != nil {
		t.Fatal(err)
	}
	if string(m) != "lit-l4-1" {
		t.Fatalf("got %q", m)
	}
}

func TestRefuseUnknownGPU(t *testing.T) {
	_, err := ResolveMachine("rtx-imaginary")
	if err == nil {
		t.Fatal("expected error")
	}
}
