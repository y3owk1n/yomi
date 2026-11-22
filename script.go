package yomi

import (
	"strings"
	"unicode"
)

// Latin, Cyrillic and Greek share letters that draw the same in most fonts
// (o, о and ο), so no recognizer can tell them apart from pixels. A word on
// screen is written in one script, so script.go settles each word's
// lookalikes on the script of its unambiguous letters, or failing that on the
// line's.

type script int

const (
	latinScript script = iota
	cyrillicScript
	greekScript
)

// lookalikes lists each group of letters that draw alike: Latin, Cyrillic,
// Greek, with 0 where a script has no such letter.
var lookalikes = [][3]rune{
	{'a', 'а', 0},
	{'c', 'с', 0},
	{'e', 'е', 0},
	{'o', 'о', 'ο'},
	{'p', 'р', 'ρ'},
	{'x', 'х', 'χ'},
	{'y', 'у', 0},
	{'i', 'і', 0},
	{'j', 'ј', 0},
	{'s', 'ѕ', 0},
	{'v', 0, 'ν'},
	{'A', 'А', 'Α'},
	{'B', 'В', 'Β'},
	{'C', 'С', 0},
	{'E', 'Е', 'Ε'},
	{'H', 'Н', 'Η'},
	{'I', 'І', 'Ι'},
	{'J', 'Ј', 0},
	{'K', 'К', 'Κ'},
	{'M', 'М', 'Μ'},
	{'N', 0, 'Ν'},
	{'O', 'О', 'Ο'},
	{'P', 'Р', 'Ρ'},
	{'S', 'Ѕ', 0},
	{'T', 'Т', 'Τ'},
	{'X', 'Х', 'Χ'},
	{'Y', 0, 'Υ'},
	{'Z', 0, 'Ζ'},
}

var lookalikeOf = func() map[rune][3]rune {
	m := map[rune][3]rune{}
	for _, g := range lookalikes {
		for _, r := range g {
			if r != 0 {
				m[r] = g
			}
		}
	}
	return m
}()

func scriptOfLetter(r rune) (script, bool) {
	if _, ambiguous := lookalikeOf[r]; ambiguous {
		return 0, false
	}
	switch {
	case unicode.Is(unicode.Cyrillic, r):
		return cyrillicScript, true
	case unicode.Is(unicode.Greek, r):
		return greekScript, true
	case unicode.Is(unicode.Latin, r):
		return latinScript, true
	}
	return 0, false
}

// dominant returns the script most of text's unambiguous letters belong to.
func dominant(text string) (script, bool) {
	var n [3]int
	for _, r := range text {
		if s, ok := scriptOfLetter(r); ok {
			n[s]++
		}
	}
	best := latinScript
	for s := range n {
		if n[s] > n[best] {
			best = script(s)
		}
	}
	return best, n[best] > 0
}

// settleScripts rewrites the lookalike letters of each word in the script of
// the word's other letters, or of the line's when the word has none.
func settleScripts(words []Word) {
	var all []rune
	for _, w := range words {
		all = append(all, []rune(w.Text)...)
	}
	line, _ := dominant(string(all))
	for i := range words {
		s, ok := dominant(words[i].Text)
		if !ok {
			s = line
		}
		rs := []rune(words[i].Text)
		changed := false
		for j, r := range rs {
			if g, ambiguous := lookalikeOf[r]; ambiguous && g[s] != 0 && g[s] != r {
				rs[j], changed = g[s], true
			}
		}
		if changed {
			words[i].Text = string(rs)
		}
	}
}

// Arabic and Persian write yeh and kaf with different letters, ي and ی, ك and
// ک, that draw alike at the start and in the middle of a word and differ only
// at its end. A recognizer reads the end, so settleYehKaf settles the rest on
// the page's word endings: Persian and Urdu when more words end in ی or ک
// than in ي or ك, Arabic otherwise. Letters only Persian and Urdu use are no
// evidence, since the recognizer reads Arabic's dots as them now and then.
var (
	toArabic  = map[rune]rune{'ی': 'ي', 'ک': 'ك'}
	toPersian = map[rune]rune{'ي': 'ی', 'ك': 'ک'}
)

func settleYehKaf(lines []Line) {
	votes := 0
	for _, l := range lines {
		for _, w := range l.Words {
			// The word's last Arabic letter, past any punctuation after it.
			rs := []rune(w.Text)
			end := len(rs) - 1
			for end >= 0 && !(unicode.Is(unicode.Arabic, rs[end]) && unicode.IsLetter(rs[end])) {
				end--
			}
			if end < 0 {
				continue
			}
			switch rs[end] {
			case 'ی', 'ک':
				votes++
			case 'ي', 'ك':
				votes--
			}
		}
	}
	to := toArabic
	if votes > 0 {
		to = toPersian
	}
	for i := range lines {
		for j := range lines[i].Words {
			w := &lines[i].Words[j]
			if !strings.ContainsAny(w.Text, "یکيك") {
				continue
			}
			rs := []rune(w.Text)
			for k := 0; k+1 < len(rs); k++ {
				if c, ok := to[rs[k]]; ok && unicode.Is(unicode.Arabic, rs[k+1]) {
					rs[k] = c
				}
			}
			w.Text = string(rs)
		}
	}
}
