# Arabic thesis typography

`NotoNaskhArabic-Regular.ttf` and `NotoNaskhArabic-Bold.ttf` are unmodified Noto
Naskh Arabic version 2.021 font files from the installed project authoring environment.
Their embedded copyright is: Copyright 2022 The Noto Project Authors.

The [upstream project](https://github.com/notofonts/arabic) distributes these fonts
under the SIL Open Font License 1.1. The accompanying
[license](NotoNaskhArabic-OFL.txt) was obtained from the
[Google Fonts upstream copy](https://github.com/google/fonts/blob/main/ofl/notonaskharabic/OFL.txt).
The font names and bytes are unchanged. The fonts are included so that thesis builds
do not depend on a Windows font installation or network connection.

The thesis renderer shapes Arabic and applies bidirectional ordering after wrapping
logical lines, using the document-only `arabic-reshaper` and `python-bidi` packages.
The editable manuscript retains logical Unicode Arabic text.
