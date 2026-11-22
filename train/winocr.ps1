# Reads evaluation pages with Windows.Media.Ocr, the engine Neru uses on
# Windows, for the baseline there. It reads each page with the engine for the
# page's language, its best case, and leaves out pages in a language Windows
# has no OCR for. Writes the same JSON lines as `yomi bench`, so pagescore.py
# scores both.
#
# Needs Windows PowerShell 5.1 (WinRT projection) and the OCR capability for
# each language (Language.OCR~~~<tag>~0.0.1.0).
#
# Usage: powershell -File winocr.ps1 EVALDIR [REPS] > out.jsonl
param([string]$Dir, [int]$Reps = 3)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]
$null = [Windows.Globalization.Language, Windows.Globalization, ContentType = WindowsRuntime]

$asTask = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
	$_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
	$_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
} | Select-Object -First 1
function Await($op, [type]$type) {
	$task = $asTask.MakeGenericMethod($type).Invoke($null, @($op))
	$task.Wait(-1) | Out-Null
	$task.Result
}

$tags = @{
	cs = 'cs-CZ'; de = 'de-DE'; el = 'el-GR'; en = 'en-US'; es = 'es-ES'; fr = 'fr-FR'; hu = 'hu-HU'
	it = 'it-IT'; nl = 'nl-NL'; pl = 'pl-PL'; pt_BR = 'pt-BR'; ro = 'ro-RO'; ru = 'ru-RU'; sv = 'sv-SE'
	tr = 'tr-TR'; ja = 'ja-JP'; ko = 'ko-KR'; zh_CN = 'zh-Hans-CN'; zh_TW = 'zh-Hant-TW'; ar = 'ar-SA'
}
$engines = @{}
$out = [Console]::OpenStandardOutput()
$utf8 = New-Object System.Text.UTF8Encoding($false)

foreach ($page in (Get-Content (Join-Path $Dir 'pages.txt') -Raw).Split(" `t`r`n", [StringSplitOptions]::RemoveEmptyEntries)) {
	$lang = $page.Split('-')[0]
	# Pages with no language prefix are English.
	$tag = if ($tags.ContainsKey($lang)) { $tags[$lang] } elseif ($lang.Length -gt 2) { 'en-US' } else { $null }
	if (-not $tag) { continue }
	if (-not $engines.ContainsKey($tag)) {
		$engines[$tag] = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage([Windows.Globalization.Language]::new($tag))
	}
	$engine = $engines[$tag]
	if (-not $engine) { [Console]::Error.WriteLine("no OCR for $tag"); continue }
	foreach ($scale in 1, 2) {
		$path = Join-Path (Resolve-Path $Dir) "$page@${scale}x.png"
		if (-not (Test-Path $path)) { continue }
		$file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($path)) ([Windows.Storage.StorageFile])
		$stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
		$decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
		$bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
		$times = @()
		$result = $null
		for ($r = 0; $r -le $Reps; $r++) {
			$sw = [Diagnostics.Stopwatch]::StartNew()
			$result = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
			$times += $sw.Elapsed.TotalMilliseconds
		}
		$stream.Dispose()
		$rest = @($times | Select-Object -Skip 1 | Sort-Object)
		$median = if ($rest.Count) { $rest[[int][math]::Floor($rest.Count / 2)] } else { $times[0] }
		$words = foreach ($line in $result.Lines) {
			foreach ($w in $line.Words) {
				$b = $w.BoundingRect
				, @($w.Text, [math]::Round($b.X / $scale, 1), [math]::Round($b.Y / $scale, 1),
					[math]::Round($b.Width / $scale, 1), [math]::Round($b.Height / $scale, 1), 1)
			}
		}
		$rec = [ordered]@{ page = $page; scale = $scale; first_ms = $times[0]; median_ms = $median; words = @($words) }
		$bytes = $utf8.GetBytes((ConvertTo-Json $rec -Compress -Depth 4) + "`n")
		$out.Write($bytes, 0, $bytes.Length)
		$out.Flush()
	}
}
