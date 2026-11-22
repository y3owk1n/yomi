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

// route is the router's view of a line: each class's mean log-probability
// per step, best first.
type route struct {
	labels []string
	scores []float64
}

// routeLine runs the router over a prepared line.
func (e *Engine) routeLine(x []float32, w int, s *nn.Scratch) route {
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
	out := route{labels: make([]string, c), scores: make([]float64, c)}
	for i, k := range order {
		out.labels[i], out.scores[i] = r.Labels[k], mean[k]
	}
	return out
}

// pageMargin and pageWidth decide which families a page is written in: a
// family is on the page once the router picks it for a line at least
// pageWidth wide (prepared pixels, two em) ahead of its runner-up by at least
// pageMargin. Lines in a script the router knows lead by 20 to 35, while icons
// and the short tokens beside them lead by under 16, which is how an English
// page sent a third of its lines to CJK, Arabic and Indic recognizers.
const (
	pageMargin = 12
	pageWidth  = 64
)

// pageFamilies returns the families the lines of one page are written in.
func pageFamilies(lines []prepared) map[string]bool {
	page := map[string]bool{}
	for _, l := range lines {
		r := l.route
		if len(r.labels) > 1 && l.w >= pageWidth && r.scores[0]-r.scores[1] >= pageMargin {
			page[r.labels[0]] = true
		}
	}
	return page
}

// pick returns the recognizers to read a line the router saw as r: the
// router's family, then the runner-up when it comes within secondReadMargin.
// A family not on page reads with the model New was given instead, and a nil
// page holds every family. It returns nil when the router finds no text in
// the line.
func (e *Engine) pick(r route, page map[string]bool) []*nn.Model {
	if e.router == nil {
		return []*nn.Model{e.model.net}
	}
	if r.labels[0] == "none" {
		return nil
	}
	onPage := func(family string) bool {
		return page == nil || page[family] || e.recognizer(family) == e.model.net
	}
	if !onPage(r.labels[0]) {
		return []*nn.Model{e.model.net}
	}
	nets := []*nn.Model{e.recognizer(r.labels[0])}
	if second := r.labels[1]; second != "none" && onPage(second) && r.scores[0]-r.scores[1] < secondReadMargin {
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

// roundIcon is what the CJK recognizer reads a round icon as. Nearly every 〇
// standing alone on a screen is an icon rather than the ideographic zero, and
// on the layout scenes they are half of the stray words routing adds. Other
// symbols stay, since a ☆ or ✓ is as often text as an icon.
const roundIcon = "〇"
