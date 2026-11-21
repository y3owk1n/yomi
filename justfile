# List the recipes.
default:
    @just --list

# Run the tests with the race detector.
test:
    go test -race ./...

# Vet, and build with cgo off for every target yomi promises.
check:
    go vet ./...
    for t in darwin/arm64 darwin/amd64 linux/amd64 linux/arm64 windows/amd64 windows/arm64; do CGO_ENABLED=0 GOOS=${t%/*} GOARCH=${t#*/} go build ./... || exit 1; done

# Format the Go files.
fmt:
    gofumpt -w .
