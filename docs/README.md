# yomi documentation

- [Guide](guide.md): installing, reading a frame, choosing models, scale,
  results, single lines, concurrency, the command line, and what yomi does
  not read.
- [How it works](how-it-works.md): the five stages of a read, from finding
  lines to words and character boxes.
- [Benchmarks](benchmarks.md): yomi against Vision, PaddleOCR, tesseract and
  Windows.Media.Ocr, on interface text, the open set, real websites and live
  captures, with how to run them yourself.
- [Training](training.md): what training uses, rendering lines and pages,
  training and fine-tuning, scoring and exporting a model.
- [Design decisions](decisions.md): what yomi chose, what it tried and
  dropped, and the measurement behind each.

For contributors, [CONTRIBUTING.md](../CONTRIBUTING.md) covers reporting
misreads, adding fonts and words, sending a model, and changing the engine.
The API reference is on
[pkg.go.dev](https://pkg.go.dev/github.com/y3owk1n/yomi).
