package forge

import (
	"math"
	"testing"
)

func TestVersion(t *testing.T) {
	v := Version()
	if v == "" {
		t.Fatal("empty version")
	}
	t.Logf("runtime version: %s", v)
}

func TestMatmulHandChecked(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6}
	b := []float32{7, 8, 9, 10, 11, 12}
	c := make([]float32, 4)
	if err := MatmulF32(2, 2, 3, a, b, c); err != nil {
		t.Fatal(err)
	}
	want := []float32{58, 64, 139, 154}
	for i := range want {
		if math.Abs(float64(c[i]-want[i])) > 1e-4 {
			t.Fatalf("c[%d]=%v want %v", i, c[i], want[i])
		}
	}
}

func TestMatvecHandChecked(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6}
	x := []float32{1, 0, -1}
	y := make([]float32, 2)
	if err := MatvecF32(2, 3, a, x, y); err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(y[0]+2)) > 1e-4 || math.Abs(float64(y[1]+2)) > 1e-4 {
		t.Fatalf("got %v want [-2,-2]", y)
	}
}

func TestMatmulRejectsShortSlice(t *testing.T) {
	err := MatmulF32(2, 2, 2, []float32{1}, []float32{1, 2, 3, 4}, make([]float32, 4))
	if err == nil {
		t.Fatal("expected error")
	}
}
