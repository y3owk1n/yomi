# yomi

OCR for screen frames in pure Go. No cgo, no downloads, weights embedded,
the same engine on macOS, Linux and Windows. Built for
[Neru](https://github.com/y3owk1n/neru).

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

Status: printable ASCII. More scripts, CJK first after Western, are planned
in [docs/design.md](docs/design.md), with the measurements against Vision,
tesseract and Windows.Media.Ocr that motivate it.

The training pipeline lives in `train/`.

MIT licensed.
