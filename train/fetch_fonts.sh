#!/bin/sh
# Downloads the training and held-out fonts into fonts/. All OFL-1.1 unless the
# line says otherwise. fonts/LICENSES.md records name, licence and source URL.
set -eu
cd "$(dirname "$0")"
mkdir -p fonts/train fonts/heldout fonts/fallback fonts/cjk-train fonts/cjk-heldout fonts/rtl-train fonts/rtl-heldout fonts/indic-train fonts/indic-heldout fonts/sea-train fonts/sea-heldout fonts/cau-train fonts/cau-heldout
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

# Monospace Arabic and Hebrew, as terminals and code editors draw them. Each
# letter takes one cell, which spreads Hebrew letters apart and stretches
# Arabic joins.
for f in "ofl/cousine/Cousine-Regular.ttf" "ofl/cousine/Cousine-Bold.ttf"; do
  get rtl-train "$f" OFL-1.1
done
fetch() { # url member... copies each OFL-1.1 member of the archive into fonts/rtl-train
  url=$1; shift
  [ -s "fonts/dl/$(basename "$url")" ] || curl -fsSL --retry 3 -o "fonts/dl/$(basename "$url")" "$url"
  for m in "$@"; do
    case "$url" in
      *.zip) unzip -o -q -j "fonts/dl/$(basename "$url")" "$m" -d fonts/rtl-train ;;
      *.tar.gz) tar xzf "fonts/dl/$(basename "$url")" -C fonts/rtl-train --strip-components 1 "$m" ;;
    esac
    echo "| $(basename "$m") | rtl-train | OFL-1.1 | $url |" >> fonts/LICENSES.md
  done
}
mkdir -p fonts/dl
fetch https://github.com/aiaf/kawkab-mono/releases/download/v0.501/kawkab-mono-0.501.zip \
  kawkab-mono-0.501/KawkabMono-Regular.ttf kawkab-mono-0.501/KawkabMono-Bold.ttf
fetch https://github.com/aliftype/amiri-typewriter/releases/download/v1.1/AmiriTypewriter-1.1.zip \
  AmiriTypewriter-1.1/AmiriTypewriter-Regular.ttf AmiriTypewriter-1.1/AmiriTypewriter-Bold.ttf
fetch https://github.com/rastikerdar/vazir-code-font/releases/download/v1.1.2/vazir-code-font-v1.1.2.zip Vazir-Code.ttf
fetch https://github.com/liberationfonts/liberation-fonts/files/7261482/liberation-fonts-ttf-2.1.5.tar.gz \
  liberation-fonts-ttf-2.1.5/LiberationMono-Regular.ttf liberation-fonts-ttf-2.1.5/LiberationMono-Bold.ttf

# Indic: Devanagari, Bengali, Gurmukhi, Gujarati, Tamil, Telugu, Kannada,
# Malayalam and Sinhala. The serif Notos and IBM Plex Devanagari are held out.
for f in "ofl/notosansdevanagari/NotoSansDevanagari[wdth,wght].ttf" "ofl/hind/Hind-Regular.ttf" "ofl/hind/Hind-SemiBold.ttf" \
  "ofl/mukta/Mukta-Regular.ttf" "ofl/baloo2/Baloo2[wght].ttf" "ofl/tirodevanagarihindi/TiroDevanagariHindi-Regular.ttf" \
  "ofl/notosansbengali/NotoSansBengali[wdth,wght].ttf" "ofl/hindsiliguri/HindSiliguri-Regular.ttf" \
  "ofl/notosansgurmukhi/NotoSansGurmukhi[wdth,wght].ttf" "ofl/notosansgujarati/NotoSansGujarati[wdth,wght].ttf" \
  "ofl/hindvadodara/HindVadodara-Regular.ttf" "ofl/notosanstamil/NotoSansTamil[wdth,wght].ttf" \
  "ofl/hindmadurai/HindMadurai-Regular.ttf" "ofl/catamaran/Catamaran[wght].ttf" \
  "ofl/notosanstelugu/NotoSansTelugu[wdth,wght].ttf" "ofl/hindguntur/HindGuntur-Regular.ttf" \
  "ofl/notosanskannada/NotoSansKannada[wdth,wght].ttf" "ofl/balootamma2/BalooTamma2[wght].ttf" \
  "ofl/notosansmalayalam/NotoSansMalayalam[wdth,wght].ttf" "ofl/manjari/Manjari-Regular.ttf" \
  "ofl/notosanssinhala/NotoSansSinhala[wdth,wght].ttf" \
  "ofl/googlesans/GoogleSans[GRAD,opsz,wght].ttf" "ofl/anekdevanagari/AnekDevanagari[wdth,wght].ttf" \
  "ofl/poppins/Poppins-Regular.ttf" "ofl/poppins/Poppins-Bold.ttf" "ofl/martelsans/MartelSans-Regular.ttf" \
  "ofl/khula/Khula-Regular.ttf" "ofl/palanquin/Palanquin-Regular.ttf" "ofl/yantramanav/Yantramanav-Regular.ttf" \
  "ofl/gotu/Gotu-Regular.ttf" "ofl/kadwa/Kadwa-Regular.ttf" "ofl/eczar/Eczar[wght].ttf" \
  "ofl/tirodevanagarimarathi/TiroDevanagariMarathi-Regular.ttf" "ofl/annapurnasil/AnnapurnaSIL-Regular.ttf" \
  "ofl/anekbangla/AnekBangla[wdth,wght].ttf" "ofl/mina/Mina-Regular.ttf" "ofl/tirobangla/TiroBangla-Regular.ttf" \
  "ofl/balooda2/BalooDa2[wght].ttf" "ofl/atma/Atma-Regular.ttf" \
  "ofl/anekgurmukhi/AnekGurmukhi[wdth,wght].ttf" "ofl/muktamahee/MuktaMahee-Regular.ttf" \
  "ofl/tirogurmukhi/TiroGurmukhi-Regular.ttf" "ofl/baloopaaji2/BalooPaaji2[wght].ttf" "ofl/braahone/BraahOne-Regular.ttf" \
  "ofl/anekgujarati/AnekGujarati[wdth,wght].ttf" "ofl/muktavaani/MuktaVaani-Regular.ttf" "ofl/muktavaani/MuktaVaani-Bold.ttf" \
  "ofl/rasa/Rasa[wght].ttf" "ofl/baloobhai2/BalooBhai2[wght].ttf" "ofl/farsan/Farsan-Regular.ttf" \
  "ofl/anektamil/AnekTamil[wdth,wght].ttf" "ofl/muktamalar/MuktaMalar-Regular.ttf" "ofl/meerainimai/MeeraInimai-Regular.ttf" \
  "ofl/pavanam/Pavanam-Regular.ttf" "ofl/tirotamil/TiroTamil-Regular.ttf" "ofl/arima/Arima[wght].ttf" \
  "ofl/baloothambi2/BalooThambi2[wght].ttf" "ofl/karlatamilupright/KarlaTamilUpright-Regular.ttf" \
  "ofl/anektelugu/AnekTelugu[wdth,wght].ttf" "ofl/mandali/Mandali-Regular.ttf" "ofl/ntr/NTR-Regular.ttf" \
  "ofl/ramaraja/Ramaraja-Regular.ttf" "ofl/suranna/Suranna-Regular.ttf" "ofl/gurajada/Gurajada-Regular.ttf" \
  "ofl/mallanna/Mallanna-Regular.ttf" "ofl/ramabhadra/Ramabhadra-Regular.ttf" "ofl/tirotelugu/TiroTelugu-Regular.ttf" \
  "ofl/balootammudu2/BalooTammudu2[wght].ttf" "ofl/timmana/Timmana-Regular.ttf" \
  "ofl/anekkannada/AnekKannada[wdth,wght].ttf" "ofl/hindmysuru/HindMysuru-Regular.ttf" "ofl/hubballi/Hubballi-Regular.ttf" \
  "ofl/benne/Benne-Regular.ttf" "ofl/tirokannada/TiroKannada-Regular.ttf" \
  "ofl/anekmalayalam/AnekMalayalam[wdth,wght].ttf" "ofl/gayathri/Gayathri-Regular.ttf" "ofl/gayathri/Gayathri-Bold.ttf" \
  "ofl/baloochettan2/BalooChettan2[wght].ttf" \
  "ofl/abhayalibre/AbhayaLibre-Regular.ttf" "ofl/abhayalibre/AbhayaLibre-Bold.ttf" "ofl/gemunulibre/GemunuLibre[wght].ttf" \
  "ofl/maname/Maname-Regular.ttf" "ofl/yaldevi/Yaldevi[wght].ttf"; do
  get indic-train "$f" OFL-1.1
done
for f in "ofl/ibmplexsansdevanagari/IBMPlexSansDevanagari-Regular.ttf" "ofl/notoserifbengali/NotoSerifBengali[wdth,wght].ttf" \
  "ofl/notoserifgurmukhi/NotoSerifGurmukhi[wght].ttf" "ofl/notoserifgujarati/NotoSerifGujarati[wght].ttf" \
  "ofl/notoseriftamil/NotoSerifTamil[wdth,wght].ttf" "ofl/notoseriftelugu/NotoSerifTelugu[wght].ttf" \
  "ofl/notoserifkannada/NotoSerifKannada[wght].ttf" "ofl/notoserifmalayalam/NotoSerifMalayalam[wght].ttf" \
  "ofl/notoserifsinhala/NotoSerifSinhala[wdth,wght].ttf"; do
  get indic-heldout "$f" OFL-1.1
done

# Southeast Asian: Thai, Lao, Khmer and Myanmar. IBM Plex Thai and the serif
# Notos are held out.
for f in "ofl/notosansthai/NotoSansThai[wdth,wght].ttf" "ofl/notoserifthai/NotoSerifThai[wdth,wght].ttf" \
  "ofl/sarabun/Sarabun-Regular.ttf" "ofl/prompt/Prompt-Regular.ttf" "ofl/kanit/Kanit-Regular.ttf" \
  "ofl/notosanslao/NotoSansLao[wdth,wght].ttf" "ofl/notosanskhmer/NotoSansKhmer[wdth,wght].ttf" \
  "ofl/battambang/Battambang-Regular.ttf" "ofl/hanuman/Hanuman[wght].ttf" "ofl/kantumruypro/KantumruyPro[wght].ttf" \
  "ofl/notosansmyanmar/NotoSansMyanmar[wdth,wght].ttf" "ofl/padauk/Padauk-Regular.ttf" "ofl/padauk/Padauk-Bold.ttf" \
  "ofl/googlesans/GoogleSans[GRAD,opsz,wght].ttf" "ofl/notosansthailooped/NotoSansThaiLooped[wdth,wght].ttf" \
  "ofl/anuphan/Anuphan[wght].ttf" "ofl/baijamjuree/BaiJamjuree-Regular.ttf" "ofl/mitr/Mitr-Regular.ttf" \
  "ofl/niramit/Niramit-Regular.ttf" "ofl/taviraj/Taviraj-Regular.ttf" "ofl/trirong/Trirong-Regular.ttf" \
  "ofl/maitree/Maitree-Regular.ttf" "ofl/athiti/Athiti-Regular.ttf" "ofl/k2d/K2D-Regular.ttf" \
  "ofl/chakrapetch/ChakraPetch-Regular.ttf" "ofl/notosanslaolooped/NotoSansLaoLooped[wdth,wght].ttf" \
  "ofl/phetsarath/Phetsarath-Regular.ttf" "ofl/phetsarath/Phetsarath-Bold.ttf" \
  "ofl/kdamthmorpro/KdamThmorPro-Regular.ttf" "ofl/nokora/Nokora[wght].ttf" "ofl/kohsantepheap/KohSantepheap-Regular.ttf" \
  "ofl/suwannaphum/Suwannaphum-Regular.ttf" "ofl/bayon/Bayon-Regular.ttf" "ofl/moulpali/Moulpali-Regular.ttf" \
  "ofl/preahvihear/Preahvihear-Regular.ttf"; do
  get sea-train "$f" OFL-1.1
done
for f in "ofl/ibmplexsansthai/IBMPlexSansThai-Regular.ttf" "ofl/notoseriflao/NotoSerifLao[wdth,wght].ttf" \
  "ofl/notoserifkhmer/NotoSerifKhmer[wdth,wght].ttf" "ofl/notoserifmyanmar/NotoSerifMyanmar-Regular.ttf"; do
  get sea-heldout "$f" OFL-1.1
done

# Georgian, Armenian and Ethiopic. The serif Notos are held out.
for f in "ofl/notosansgeorgian/NotoSansGeorgian[wdth,wght].ttf" "ofl/notosansarmenian/NotoSansArmenian[wdth,wght].ttf" \
  "ofl/notosansethiopic/NotoSansEthiopic[wdth,wght].ttf" "ofl/abyssinicasil/AbyssinicaSIL-Regular.ttf" \
  "ofl/googlesans/GoogleSans[GRAD,opsz,wght].ttf" "ofl/menbere/Menbere[wght].ttf"; do
  get cau-train "$f" OFL-1.1
done
for f in "ofl/notoserifgeorgian/NotoSerifGeorgian[wdth,wght].ttf" "ofl/notoserifarmenian/NotoSerifArmenian[wdth,wght].ttf" \
  "ofl/notoserifethiopic/NotoSerifEthiopic[wdth,wght].ttf"; do
  get cau-heldout "$f" OFL-1.1
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
# DejaVu Sans Mono draws Arabic in monospace.
cp fonts/train/DejaVuSansMono.ttf fonts/rtl-train/DejaVuSansMono.ttf
echo "| DejaVuSansMono.ttf | rtl-train | Bitstream-Vera | $D |" >> fonts/LICENSES.md
# DejaVu Sans also draws Armenian and Georgian.
for f in DejaVuSans.ttf DejaVuSans-Bold.ttf; do
  cp fonts/train/$f fonts/cau-train/$f
  echo "| $f | cau-train | Bitstream-Vera | $D |" >> fonts/LICENSES.md
done

# GNU FreeFont (GPL-3.0 with the font exception) draws Armenian, Georgian and
# Ethiopic. Only training uses them, and no font file or rendering ships.
F=https://ftp.gnu.org/gnu/freefont/freefont-ttf-20120503.zip
if [ ! -s fonts/cau-train/FreeSerif.ttf ]; then
  mkdir -p fonts/dl && curl -fsSL --retry 3 -o fonts/dl/freefont.zip "$F"
  unzip -o -q -j fonts/dl/freefont.zip 'freefont-20120503/FreeSans.ttf' 'freefont-20120503/FreeSansBold.ttf' \
    'freefont-20120503/FreeSerif.ttf' 'freefont-20120503/FreeSerifBold.ttf' -d fonts/cau-train
fi
for f in FreeSans.ttf FreeSansBold.ttf FreeSerif.ttf FreeSerifBold.ttf; do
  echo "| $f | cau-train | GPL-3.0-with-font-exception | $F |" >> fonts/LICENSES.md
done
# FreeMono draws both Hebrew and Arabic in monospace.
unzip -o -q -j fonts/dl/freefont.zip 'freefont-20120503/FreeMono.ttf' 'freefont-20120503/FreeMonoBold.ttf' -d fonts/rtl-train
for f in FreeMono.ttf FreeMonoBold.ttf; do
  echo "| $f | rtl-train | GPL-3.0-with-font-exception | $F |" >> fonts/LICENSES.md
done
ls fonts/train fonts/heldout | wc -l
