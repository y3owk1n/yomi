// Command crops prints the line finder's boxes for frames, one JSON line per
// frame: {"page","scale","boxes":[[crop x0,y0,x1,y1, ink x0,y0,x1,y1]...]}
// in logical pixels.
//
//	crops DIR SCALE < names   (reads DIR/NAME@{SCALE}x.png)
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"image"
	"image/draw"
	"image/png"
	"os"
	"strconv"

	"github.com/y3owk1n/yomi/internal/linefind"
)

func main() {
	dir := os.Args[1]
	scale, _ := strconv.ParseFloat(os.Args[2], 64)
	enc := json.NewEncoder(os.Stdout)
	sc := bufio.NewScanner(os.Stdin)
	for sc.Scan() {
		name := sc.Text()
		f, err := os.Open(fmt.Sprintf("%s/%s@%gx.png", dir, name, scale))
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			continue
		}
		src, err := png.Decode(f)
		_ = f.Close()
		if err != nil {
			fmt.Fprintln(os.Stderr, name, err)
			continue
		}
		img := image.NewRGBA(src.Bounds())
		draw.Draw(img, img.Rect, src, image.Point{}, draw.Src)
		var boxes [][8]int
		for _, b := range linefind.Find(img, scale, linefind.DefaultParams()) {
			c, k := b.Crop, b.Ink
			boxes = append(boxes, [8]int{c.Min.X, c.Min.Y, c.Max.X, c.Max.Y, k.Min.X, k.Min.Y, k.Max.X, k.Max.Y})
		}
		_ = enc.Encode(map[string]any{"page": name, "scale": scale, "boxes": boxes})
	}
}
