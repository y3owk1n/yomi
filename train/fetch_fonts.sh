#!/bin/sh
# Downloads the training and held-out fonts into fonts/. All OFL-1.1 unless the
# line says otherwise. fonts/LICENSES.md records name, licence and source URL.
set -eu
cd "$(dirname "$0")"
mkdir -p fonts/train fonts/heldout fonts/fallback fonts/cjk-train fonts/cjk-heldout fonts/rtl-train fonts/rtl-heldout
G=https://raw.githubusercontent.com/google/fonts/main
: > fonts/LICENSES.md
echo "| File | Split | Licence | Source |" >> fonts/LICENSES.md
echo "|---|---|---|---|" >> fonts/LICENSES.md

get() { # split path licence
  name=$(basename "$2")
  out="fonts/$1/$(printf %s "$name" | tr -d '[]' | tr ',' '_')"
  url="$G/$(printf %s "$2" | sed 's/\[/%5B/g; s/\]/%5D/g')"
  [ -s "$out" ] || curl -fsSL --retry 3 -o "$out" "$url"
  echo "| $(basename "$out") | $1 | $3 | $url |" >> fonts/LICENSES.md
}

for f in \
  "ofl/inter/Inter[opsz,wght].ttf" "ofl/notosans/NotoSans[wdth,wght].ttf" \
  "ofl/roboto/Roboto[wdth,wght].ttf" "ofl/opensans/OpenSans[wdth,wght].ttf" \
  "ofl/sourcesans3/SourceSans3[wght].ttf" "ofl/lato/Lato-Regular.ttf" "ofl/lato/Lato-Bold.ttf" \
  "ofl/firasans/FiraSans-Regular.ttf" "ofl/firasans/FiraSans-Medium.ttf" "ofl/firasans/FiraSans-Bold.ttf" \
  "ofl/cantarell/Cantarell-Regular.ttf" "ofl/cantarell/Cantarell-Bold.ttf" \
  "ofl/worksans/WorkSans[wght].ttf" "ofl/ptsans/PT_Sans-Web-Regular.ttf" "ofl/ptsans/PT_Sans-Web-Bold.ttf" \
  "ofl/publicsans/PublicSans[wght].ttf" "ofl/arimo/Arimo[wght].ttf" "ofl/nunitosans/NunitoSans[YTLC,opsz,wdth,wght].ttf" \
  "ofl/dmsans/DMSans[opsz,wght].ttf" "ofl/librefranklin/LibreFranklin[wght].ttf" "ofl/karla/Karla[wght].ttf" \
  "ofl/notoserif/NotoSerif[wdth,wght].ttf" "ofl/sourceserif4/SourceSerif4[opsz,wght].ttf" "ofl/tinos/Tinos-Regular.ttf" \
  "ofl/notosansmono/NotoSansMono[wdth,wght].ttf" "ofl/jetbrainsmono/JetBrainsMono[wght].ttf" \
  "ofl/sourcecodepro/SourceCodePro[wght].ttf" "ofl/robotomono/RobotoMono[wght].ttf" "ofl/firacode/FiraCode[wght].ttf" \
  "ofl/firamono/FiraMono-Regular.ttf" "ofl/firamono/FiraMono-Medium.ttf" "ofl/ptmono/PTM55FT.ttf" \
  "ofl/inter/Inter-Italic[opsz,wght].ttf" "ofl/notosans/NotoSans-Italic[wdth,wght].ttf" \
  "ofl/bevietnampro/BeVietnamPro-Regular.ttf" "ofl/bevietnampro/BeVietnamPro-SemiBold.ttf" \
  "ofl/ptserif/PT_Serif-Web-Regular.ttf" "ofl/notosansdisplay/NotoSansDisplay[wdth,wght].ttf"; do
  get train "$f" OFL-1.1
done
for f in "ufl/ubuntu/Ubuntu-Regular.ttf" "ufl/ubuntu/Ubuntu-Medium.ttf" "ufl/ubuntu/Ubuntu-Bold.ttf" \
  "ufl/ubuntumono/UbuntuMono-Regular.ttf" "ufl/ubuntumono/UbuntuMono-Bold.ttf"; do
  get train "$f" UFL-1.0
done
for f in "ofl/ibmplexsans/IBMPlexSans[wdth,wght].ttf" "ofl/ibmplexmono/IBMPlexMono-Regular.ttf" \
  "ofl/ibmplexmono/IBMPlexMono-Medium.ttf"; do
  get heldout "$f" OFL-1.1
done

# Fallback fonts: what a line's own font lacks (arrows, Mac modifier keys,
# check marks) renders in these, as an OS falls back.
for f in "ofl/notosanssymbols2/NotoSansSymbols2-Regular.ttf" "ofl/notosanssymbols/NotoSansSymbols[wght].ttf" \
  "ofl/notosansmath/NotoSansMath-Regular.ttf"; do
  get fallback "$f" OFL-1.1
done

# CJK: Simplified and Traditional Chinese, Japanese and Korean, in sans,
# serif, rounded and handwriting-like styles. IBM Plex JP and KR and Noto Sans
# HK are held out for validation.
for f in "ofl/notosanssc/NotoSansSC[wght].ttf" "ofl/notosanstc/NotoSansTC[wght].ttf" \
  "ofl/notosansjp/NotoSansJP[wght].ttf" "ofl/notosanskr/NotoSansKR[wght].ttf" \
  "ofl/notoserifsc/NotoSerifSC[wght].ttf" "ofl/notoseriftc/NotoSerifTC[wght].ttf" \
  "ofl/notoserifjp/NotoSerifJP[wght].ttf" "ofl/notoserifkr/NotoSerifKR[wght].ttf" \
  "ofl/mplus1p/MPLUS1p-Regular.ttf" "ofl/mplus1p/MPLUS1p-Bold.ttf" "ofl/mplusrounded1c/MPLUSRounded1c-Regular.ttf" \
  "ofl/zenkakugothicnew/ZenKakuGothicNew-Regular.ttf" "ofl/zenkakugothicnew/ZenKakuGothicNew-Medium.ttf" \
  "ofl/bizudpgothic/BIZUDPGothic-Regular.ttf" "ofl/bizudpgothic/BIZUDPGothic-Bold.ttf" \
  "ofl/sawarabigothic/SawarabiGothic-Regular.ttf" "ofl/kleeone/KleeOne-Regular.ttf" \
  "ofl/nanumgothic/NanumGothic-Regular.ttf" "ofl/nanumgothic/NanumGothic-Bold.ttf" \
  "ofl/nanummyeongjo/NanumMyeongjo-Regular.ttf" "ofl/gothica1/GothicA1-Regular.ttf" "ofl/gothica1/GothicA1-Medium.ttf" \
  "ofl/lxgwwenkaitc/LXGWWenKaiTC-Regular.ttf"; do
  get cjk-train "$f" OFL-1.1
done
for f in "ofl/ibmplexsansjp/IBMPlexSansJP-Regular.ttf" "ofl/ibmplexsanskr/IBMPlexSansKR-Regular.ttf" \
  "ofl/notosanshk/NotoSansHK[wght].ttf"; do
  get cjk-heldout "$f" OFL-1.1
done

# Right to left: Arabic, Persian and Urdu (naskh, kufi and sans styles) and
# Hebrew. IBM Plex Sans Arabic and Hebrew are held out.
for f in "ofl/notosansarabic/NotoSansArabic[wdth,wght].ttf" "ofl/notonaskharabic/NotoNaskhArabic[wght].ttf" \
  "ofl/notokufiarabic/NotoKufiArabic[wght].ttf" "ofl/vazirmatn/Vazirmatn[wght].ttf" "ofl/cairo/Cairo[slnt,wght].ttf" \
  "ofl/tajawal/Tajawal-Regular.ttf" "ofl/tajawal/Tajawal-Bold.ttf" "ofl/almarai/Almarai-Regular.ttf" \
  "ofl/changa/Changa[wght].ttf" "ofl/readexpro/ReadexPro[HEXP,wght].ttf" "ofl/amiri/Amiri-Regular.ttf" \
  "ofl/notosanshebrew/NotoSansHebrew[wdth,wght].ttf" "ofl/notoserifhebrew/NotoSerifHebrew[wdth,wght].ttf" \
  "ofl/rubik/Rubik[wght].ttf" "ofl/heebo/Heebo[wght].ttf" "ofl/assistant/Assistant[wght].ttf" \
  "ofl/frankruhllibre/FrankRuhlLibre[wght].ttf"; do
  get rtl-train "$f" OFL-1.1
done
for f in "ofl/ibmplexsansarabic/IBMPlexSansArabic-Regular.ttf" "ofl/ibmplexsanshebrew/IBMPlexSansHebrew-Regular.ttf"; do
  get rtl-heldout "$f" OFL-1.1
done

# DejaVu Sans and Sans Mono (Bitstream Vera licence plus public-domain changes).
D=https://github.com/dejavu-fonts/dejavu-fonts/releases/download/version_2_37/dejavu-fonts-ttf-2.37.zip
if [ ! -s fonts/train/DejaVuSans.ttf ]; then
  mkdir -p fonts/dl && curl -fsSL --retry 3 -o fonts/dl/dejavu.zip "$D"
  unzip -o -q -j fonts/dl/dejavu.zip 'dejavu-fonts-ttf-2.37/ttf/DejaVuSans.ttf' \
    'dejavu-fonts-ttf-2.37/ttf/DejaVuSans-Bold.ttf' 'dejavu-fonts-ttf-2.37/ttf/DejaVuSansMono.ttf' -d fonts/train
fi
for f in DejaVuSans.ttf DejaVuSans-Bold.ttf DejaVuSansMono.ttf; do
  echo "| $f | train | Bitstream-Vera | $D |" >> fonts/LICENSES.md
done
ls fonts/train fonts/heldout | wc -l
