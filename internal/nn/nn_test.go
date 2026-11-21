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

// TestModel_Forward_MatchesTrainingReference runs each model package's
// reference crops (models/*/testdata, written by train/export.py) through the
// Go network. Full heads must give PyTorch's logits, gated heads its most
// likely class per step, so a kernel change that alters what a model reads
// fails here.
func TestModel_Forward_MatchesTrainingReference(t *testing.T) {
	dirs, err := filepath.Glob("../../models/*/testdata")
	if err != nil || len(dirs) == 0 {
		t.Fatalf("no model reference crops: %v", err)
	}
	for _, dir := range dirs {
		model := filepath.Dir(dir)
		m, err := Load(os.DirFS(model))
		if err != nil {
			t.Fatalf("%s: %v", model, err)
		}
		metas, _ := filepath.Glob(filepath.Join(dir, "*.json"))
		for _, meta := range metas {
			t.Run(filepath.Base(model)+"/"+filepath.Base(meta), func(t *testing.T) {
				var d struct{ H, W, T, C int }
				raw, err := os.ReadFile(meta)
				if err != nil {
					t.Fatal(err)
				}
				if err := json.Unmarshal(raw, &d); err != nil {
					t.Fatal(err)
				}
				base := meta[:len(meta)-len(".json")]
				in := readF32(t, base+".in")
				if _, err := os.Stat(base + ".classes"); err == nil {
					checkSteps(t, m.Steps(in, d.W, &Scratch{}), base, d.T)
					return
				}
				got, steps, classes := m.Forward(in, d.W, &Scratch{})
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
}

func checkSteps(t *testing.T, got []Step, base string, n int) {
	t.Helper()
	raw, err := os.ReadFile(base + ".classes")
	if err != nil {
		t.Fatal(err)
	}
	probs := readF32(t, base+".probs")
	if len(got) != n {
		t.Fatalf("%d steps, want %d", len(got), n)
	}
	for j, s := range got {
		want := int(int32(binary.LittleEndian.Uint32(raw[j*4:])))
		if s.Class != want {
			t.Fatalf("step %d reads class %d, want %d", j, s.Class, want)
		}
		if diff := math.Abs(float64(s.Prob - probs[j])); diff > 1e-3 {
			t.Fatalf("step %d probability %v, want %v", j, s.Prob, probs[j])
		}
	}
}
