package nn

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func readF32(t *testing.T, path string) []float32 {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	f := make([]float32, len(b)/4)
	for i := range f {
		f[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return f
}

// TestModel_Forward_MatchesTrainingReference runs prepared crops through the
// Go network and compares the logits with those PyTorch produced at export
// (train/export.py), so a kernel change that alters what the model reads
// fails here.
func TestModel_Forward_MatchesTrainingReference(t *testing.T) {
	m, err := Load(os.DirFS("../../models/latin"))
	if err != nil {
		t.Fatal(err)
	}
	metas, err := filepath.Glob("testdata/*.json")
	if err != nil || len(metas) == 0 {
		t.Fatalf("no reference crops: %v", err)
	}
	for _, meta := range metas {
		t.Run(filepath.Base(meta), func(t *testing.T) {
			var d struct{ H, W, T, C int }
			raw, err := os.ReadFile(meta)
			if err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(raw, &d); err != nil {
				t.Fatal(err)
			}
			base := meta[:len(meta)-len(".json")]
			got, steps, classes := m.Forward(readF32(t, base+".in"), d.W, &Scratch{})
			want := readF32(t, base+".logits")
			if steps != d.T || classes != d.C {
				t.Fatalf("shape %dx%d, want %dx%d", steps, classes, d.T, d.C)
			}
			for j := range steps {
				ga, wa := 0, 0
				for k := range classes {
					if diff := math.Abs(float64(got[j*classes+k] - want[j*classes+k])); diff > 1e-3 {
						t.Fatalf("step %d class %d: %v, want %v", j, k, got[j*classes+k], want[j*classes+k])
					}
					if got[j*classes+k] > got[j*classes+ga] {
						ga = k
					}
					if want[j*classes+k] > want[j*classes+wa] {
						wa = k
					}
				}
				if ga != wa {
					t.Fatalf("step %d reads class %d, want %d", j, ga, wa)
				}
			}
		})
	}
}
