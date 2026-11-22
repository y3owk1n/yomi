// Reads evaluation pages with tesseract, the engine Neru uses on Linux, for
// the baseline there. It sets tesseract up the way Neru does (LSTM only,
// sparse text) and reads each page with the language data for the page's
// language plus English, which UI text mixes in. Writes the same JSON lines as
// `yomi bench`, so pagescore.py scores both.
//
// Build: cc -O2 -o tesseract-bench tesseract.c -ltesseract -lleptonica
// Usage: tesseract-bench TESSDATA EVALDIR [REPS] > out.jsonl
#include <leptonica/allheaders.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tesseract/capi.h>
#include <time.h>

static const char *langs[][2] = {
    {"cs", "ces"},     {"de", "deu"},     {"el", "ell"},      {"en", "eng"}, {"es", "spa"}, {"fr", "fra"},
    {"hu", "hun"},     {"it", "ita"},     {"nl", "nld"},      {"pl", "pol"}, {"pt_BR", "por"}, {"ro", "ron"},
    {"ru", "rus"},     {"sv", "swe"},     {"tr", "tur"},      {"uk", "ukr"}, {"vi", "vie"}, {"ja", "jpn"},
    {"ko", "kor"},     {"zh_CN", "chi_sim"}, {"zh_TW", "chi_tra"}, {"ar", "ara"}, {"he", "heb"}, {"hi", "hin"},
    {"th", "tha"},     {"lo", "lao"},     {"km", "khm"},      {"my", "mya"}, {"am", "amh"}, {"hy", "hye"},
    {"ka", "kat"},
};

// tessLang gives the tesseract languages for a page, from the prefix of its
// name. Pages with no language prefix are English.
static void tessLang(const char *page, char *out, size_t n) {
	size_t k = strcspn(page, "-");
	for (size_t i = 0; i < sizeof langs / sizeof langs[0]; i++) {
		if (strlen(langs[i][0]) == k && strncmp(page, langs[i][0], k) == 0) {
			if (strcmp(langs[i][1], "eng") == 0) {
				snprintf(out, n, "eng");
			} else {
				snprintf(out, n, "%s+eng", langs[i][1]);
			}
			return;
		}
	}
	snprintf(out, n, "eng");
}

static double now_ms(void) {
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static int cmp(const void *a, const void *b) {
	double x = *(const double *)a, y = *(const double *)b;
	return (x > y) - (x < y);
}

// jsonString writes s as a JSON string.
static void jsonString(const char *s) {
	putchar('"');
	for (; *s; s++) {
		if (*s == '"' || *s == '\\') {
			printf("\\%c", *s);
		} else if ((unsigned char)*s < 0x20) {
			printf("\\u%04x", *s);
		} else {
			putchar(*s);
		}
	}
	putchar('"');
}

int main(int argc, char **argv) {
	if (argc < 3) {
		fprintf(stderr, "usage: tesseract-bench TESSDATA EVALDIR [REPS]\n");
		return 2;
	}
	const char *data = argv[1], *dir = argv[2];
	int reps = argc > 3 ? atoi(argv[3]) : 3;
	char path[4096], page[256], lang[64], loaded[64] = "";
	snprintf(path, sizeof path, "%s/pages.txt", dir);
	FILE *pages = fopen(path, "r");
	if (pages == NULL) {
		perror(path);
		return 1;
	}
	TessBaseAPI *api = TessBaseAPICreate();
	while (fscanf(pages, "%255s", page) == 1) {
		tessLang(page, lang, sizeof lang);
		if (strcmp(lang, loaded) != 0) {
			TessBaseAPIEnd(api);
			if (TessBaseAPIInit2(api, data, lang, OEM_LSTM_ONLY) != 0) {
				fprintf(stderr, "no language data for %s\n", lang);
				return 1;
			}
			TessBaseAPISetPageSegMode(api, PSM_SPARSE_TEXT);
			TessBaseAPISetVariable(api, "debug_file", "/dev/null");
			snprintf(loaded, sizeof loaded, "%s", lang);
		}
		for (int scale = 1; scale <= 2; scale++) {
			snprintf(path, sizeof path, "%s/%s@%dx.png", dir, page, scale);
			PIX *pix = pixRead(path);
			if (pix == NULL) {
				continue;
			}
			double times[64];
			int n = reps + 1 < 64 ? reps + 1 : 64;
			for (int r = 0; r < n; r++) {
				TessBaseAPIClear(api);
				TessBaseAPISetImage2(api, pix);
				double t = now_ms();
				TessBaseAPIRecognize(api, NULL);
				times[r] = now_ms() - t;
			}
			double first = times[0];
			qsort(times + 1, n - 1, sizeof times[0], cmp);
			printf("{\"page\":\"%s\",\"scale\":%d,\"first_ms\":%.3f,\"median_ms\":%.3f,\"words\":[", page, scale,
			       first, n > 1 ? times[1 + (n - 1) / 2] : first);
			TessResultIterator *it = TessBaseAPIGetIterator(api);
			int sep = 0;
			if (it != NULL) {
				do {
					char *text = TessResultIteratorGetUTF8Text(it, RIL_WORD);
					int l, t, r, b;
					const TessPageIterator *pi = TessResultIteratorGetPageIteratorConst(it);
					if (text != NULL && *text && TessPageIteratorBoundingBox(pi, RIL_WORD, &l, &t, &r, &b)) {
						printf("%s[", sep++ ? "," : "");
						jsonString(text);
						printf(",%.1f,%.1f,%.1f,%.1f,%.3f]", (double)l / scale, (double)t / scale,
						       (double)(r - l) / scale, (double)(b - t) / scale,
						       TessResultIteratorConfidence(it, RIL_WORD) / 100);
					}
					TessDeleteText(text);
				} while (TessResultIteratorNext(it, RIL_WORD));
				TessResultIteratorDelete(it);
			}
			printf("]}\n");
			fflush(stdout);
			pixDestroy(&pix);
		}
	}
	TessBaseAPIDelete(api);
	return 0;
}
