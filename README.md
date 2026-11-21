# yomi

yomi reads text off screen frames. It is pure Go with no cgo, ships its
weights embedded, downloads nothing, and runs the same engine on macOS, Linux
and Windows. It is built for [Neru](https://github.com/y3owk1n/neru).

```go
m, err := latin.Model() // github.com/y3owk1n/yomi/models/latin
if err != nil {
	return err
}
e := yomi.New(m)
lines, err := e.Read(ctx, frame, 2) // frame is an *image.RGBA, 2 on a HiDPI display
for _, l := range lines {
	for _, w := range l.Words {
		fmt.Println(w.Text, w.Box, w.Confidence)
	}
}
```

It reads printable ASCII today. [docs/design.md](docs/design.md) plans the
other scripts, Western first and then CJK, and holds the measurements against
Vision, tesseract and Windows.Media.Ocr.

The training pipeline lives in `train/`.

MIT licensed.
