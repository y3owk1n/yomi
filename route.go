package yomi

import (
	"math"
	"sort"

	"github.com/y3owk1n/yomi/internal/nn"
)

// WithModel adds a recognizer for more scripts. With a router (WithRouter),
// the recognizer of the family the router names reads each line. Without
// one, the model New was given reads every line.
func WithModel(m *Model) Option {
	return func(e *Engine) { e.byFamily[m.net.Family] = m }
}

// WithRouter makes r, a router model, choose each line's recognizer. A line
// the router finds no text in reads as nothing, and a family no recognizer
// was added for falls back to the model New was given.
func WithRouter(r *Model) Option {
	return func(e *Engine) { e.router = r }
}

// secondReadMargin is how close, in mean log-probability per step, the
// router's runner-up family must come to its first choice for the line to be
// read by both recognizers, keeping the read the recognizer is surer of. At
// 2, about 4% of Western UI lines and 14% of held-out Southeast Asian lines
// take a second read.
const secondReadMargin = 2

// pick returns the recognizers to read a prepared line with: the router's
// family, then the runner-up when it comes within secondReadMargin. It
// returns nil when the router finds no text in the line.
func (e *Engine) pick(x []float32, w int, s *nn.Scratch) []*nn.Model {
	if e.router == nil {
		return []*nn.Model{e.model.net}
	}
	r := e.router.net
	logits, t, c := r.Forward(x, w, s)
	// Each class's log-probability, averaged over the line's steps. Softmax
	// per step first, so a step's scale does not weigh it.
	mean := make([]float64, c)
	for j := range t {
		row := logits[j*c : (j+1)*c]
		maxv := row[0]
		for _, v := range row {
			maxv = max(maxv, v)
		}
		var sum float64
		for _, v := range row {
			sum += math.Exp(float64(v - maxv))
		}
		lse := float64(maxv) + math.Log(sum)
		for k, v := range row {
			mean[k] += (float64(v) - lse) / float64(t)
		}
	}
	order := make([]int, c)
	for k := range order {
		order[k] = k
	}
	sort.Slice(order, func(a, b int) bool { return mean[order[a]] > mean[order[b]] })
	if r.Labels[order[0]] == "none" {
		return nil
	}
	nets := []*nn.Model{e.recognizer(r.Labels[order[0]])}
	if second := r.Labels[order[1]]; second != "none" && mean[order[0]]-mean[order[1]] < secondReadMargin {
		if n := e.recognizer(second); n != nets[0] {
			nets = append(nets, n)
		}
	}
	return nets
}

// recognizer returns the recognizer added for family, or the model New was
// given when none was.
func (e *Engine) recognizer(family string) *nn.Model {
	if m, ok := e.byFamily[family]; ok {
		return m.net
	}
	return e.model.net
}

// sureness is the mean probability of the characters read in steps. Of two
// recognizers' reads of a line, readBox keeps the surer one.
func sureness(steps []nn.Step) float64 {
	var sum float64
	n, prev := 0, 0
	for _, st := range steps {
		if st.Class != prev && st.Class != 0 {
			sum += float64(st.Prob)
			n++
		}
		prev = st.Class
	}
	if n == 0 {
		return 0
	}
	return sum / float64(n)
}
