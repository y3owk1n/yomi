package yomi

import (
	"image"
	"unicode"
)

// Right-to-left scripts read off the screen in visual order, left to right,
// the way the recognizer scans. logicalOrder puts a word's characters, and
// their boxes with them, in the order they were written: an Arabic or Hebrew
// word reverses, and the digits or Latin letters inside it keep their own
// left-to-right order.

func rightToLeft(r rune) bool {
	return unicode.In(r, unicode.Arabic, unicode.Hebrew, unicode.Syriac, unicode.Thaana, unicode.Nko)
}

func leftToRight(r rune) bool {
	return unicode.IsDigit(r) || unicode.Is(unicode.Latin, r)
}

// logicalOrder rewrites w, read in visual order, in logical order. A word
// without a right-to-left letter stays as it is.
func logicalOrder(w *Word) {
	rs := []rune(w.Text)
	rtl := false
	for _, r := range rs {
		if rightToLeft(r) {
			rtl = true
			break
		}
	}
	if !rtl {
		return
	}
	// order[i] is the visual position of the i-th character in logical order:
	// the word reversed, then each left-to-right run flipped back.
	n := len(rs)
	order := make([]int, n)
	for i := range order {
		order[i] = n - 1 - i
	}
	for i := 0; i < n; {
		if !leftToRight(rs[order[i]]) {
			i++
			continue
		}
		j := i
		for j < n && (leftToRight(rs[order[j]]) || rs[order[j]] == '.' || rs[order[j]] == ':' || rs[order[j]] == ',') {
			j++
		}
		for a, b := i, j-1; a < b; a, b = a+1, b-1 {
			order[a], order[b] = order[b], order[a]
		}
		i = j
	}
	text := make([]rune, n)
	chars := make([]image.Rectangle, n)
	for i, v := range order {
		text[i], chars[i] = rs[v], w.Chars[v]
	}
	w.Text, w.Chars = string(text), chars
}
