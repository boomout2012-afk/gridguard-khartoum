# Bundled presentation fonts

GridGuard Khartoum registers these fonts from this directory at application startup so the
presentation remains visually deterministic and fully offline. The binary files are unmodified
TrueType-outline copies from the upstream projects at the pinned revisions below. TrueType is
used so the same files register in both Qt and ReportLab and are embedded in the academic PDF.

| Family | Bundled files | Upstream revision | License |
| --- | --- | --- | --- |
| Source Serif 4 | `SourceSerif4-Regular.ttf`, `SourceSerif4-Semibold.ttf` | [`adobe-fonts/source-serif@5f220b1`](https://github.com/adobe-fonts/source-serif/tree/5f220b17d27ed64873f22cde0dd593685387bd19) | SIL Open Font License 1.1; see `licenses/SourceSerif4-LICENSE.md` |
| Atkinson Hyperlegible | `AtkinsonHyperlegible-Regular.ttf`, `AtkinsonHyperlegible-Bold.ttf` | [`googlefonts/atkinson-hyperlegible@1cb3116`](https://github.com/googlefonts/atkinson-hyperlegible/tree/1cb311624b2ddf88e9e37873999d165a8cd28b46) | SIL Open Font License 1.1; see `licenses/AtkinsonHyperlegible-OFL.txt` |
| IBM Plex Mono | `IBMPlexMono-Regular.ttf`, `IBMPlexMono-SemiBold.ttf` | [`IBM/plex@bf26009`](https://github.com/IBM/plex/tree/bf260093582f04622aacc1e9f9ca604d7ccd0c42) | SIL Open Font License 1.1; see `licenses/IBMPlexMono-license.txt` |

## SHA-256 checksums

- `SourceSerif4-Regular.ttf`: `e5a4ee6a3d87bb9024796be390c6771e2a0eb1883dae25effaf57ca01668e24b`
- `SourceSerif4-Semibold.ttf`: `36db62940cb5728b12b1802476dc7fcf4c6c519a7bdd476ba23a4e555fc4655f`
- `AtkinsonHyperlegible-Regular.ttf`: `7fb917c89019896d0b52ee84b7cbb3304c18cb90b19a62f5e32712bd23e97669`
- `AtkinsonHyperlegible-Bold.ttf`: `5a3b0c8cc8ca545155150b4512a1fa248298df121c50d6557e651e61fbdab92f`
- `IBMPlexMono-Regular.ttf`: `7c6fbddca4b700be918f5f6183d9bd4464fa427fe435f0b480d77fe2bb8c5a43`
- `IBMPlexMono-SemiBold.ttf`: `f04d7c488ddf7d1fa99f2574efc3406ea4cbe17bb1af3a1ab960f84d0c96a172`

The Windows fallbacks in `gridguard.ui.theme` are used only if Qt reports that one of these
bundled files cannot be loaded. A runtime warning names the missing or invalid asset rather than
silently changing the typography.
