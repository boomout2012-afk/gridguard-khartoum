# Third-party map data

GridGuard Khartoum bundles a small, offline derivative of two public geographic datasets.
The boundaries provide administrative context only. They do not represent an electrical network,
and the application overlays synthetic decision-support results rather than measured utility data.

## Sudan administrative boundaries (COD-AB)

- Provider: United Nations Office for the Coordination of Humanitarian Affairs (OCHA), distributed
  through the Humanitarian Data Exchange (HDX)
- Dataset: [Sudan administrative boundaries (COD-AB)](https://data.humdata.org/dataset/cod-ab-sdn)
- Source revision: `v03`, valid from `2020-08-31`
- License: [Creative Commons Attribution 3.0 IGO](https://creativecommons.org/licenses/by/3.0/igo/)
- Downloaded archive: `sdn_admin_boundaries.geojson.zip`, 1,200,093 bytes,
  SHA-256 `75330d4e12e53f966df0e5683e04f0919e9964952fbbf827f3729038e603c95e`
- Selected member: `sdn_admin2.geojson`, 1,621,077 bytes,
  SHA-256 `c13cf1e8515730267d41bd560a1f1a0489552ab2900abb070a9aececacca1bb9`

The derived file selects `ADM1_PCODE=SD01` and these exact second-level units:

| P-code | COD-AB English name | GridGuard display name |
|---|---|---|
| `SD01001` | Jebel Awlia | Jebel Awlia |
| `SD01002` | Um Bada | Ombada |
| `SD01003` | Bahri | Bahri |
| `SD01004` | Sharg An Neel | East Nile |
| `SD01005` | Karrari | Karari |
| `SD01006` | Um Durman | Omdurman |
| `SD01007` | Khartoum | Khartoum |

The source English and Arabic names, P-codes, raw WGS84 Polygon/MultiPolygon geometry, and source
center coordinates used as label anchors are retained. Only the product-facing English names are
normalized as shown above.

## Natural Earth rivers

- Provider: [Natural Earth](https://www.naturalearthdata.com/)
- Dataset: 1:10m rivers and lake centerlines, version `5.0.0`
- License: [public domain](https://www.naturalearthdata.com/about/terms-of-use/)
- Immutable source revision: `b70b7b8766c3c83d5f5145cb6f1c93921d383d3f`
- Source file: `ne_10m_rivers_lake_centerlines.geojson`, 7,307,743 bytes,
  SHA-256 `bb854a900ecbd3b408df46d5e16e3e0f974ba55993f9d8b5c26e855273c0905a`

Three `featurecla=River` components are selected exactly:

| Output ID | `rivernum` | `name_en` | Feature | Component |
|---|---:|---|---:|---:|
| `white_nile` | 17 | White Nile | 278 | 0 |
| `blue_nile` | 73 | Blue Nile | 1075 | 0 |
| `nile` | 4 | Nile | 732 | 4 |

Natural Earth's Blue Nile component ends at `[32.488862, 15.632595]`. The build appends the shared
confluence coordinate `[32.490164, 15.634548]`, a 258.07 m adjustment, and then clips all selected
linework against the dissolved seven-locality boundary mask. The result is stored in the order
White Nile, Blue Nile, Nile.

## Reproducibility and runtime use

`tools/build_map_asset.py` verifies every source byte count and SHA-256 digest before transforming
data. `map-provenance.json` records the source pins, exact selectors and transformations, and the
hashes of both derived GeoJSON files. The application verifies that manifest before parsing either
file and fails closed on drift.

The desktop runtime performs no download and requires no GIS package. It validates the closed
GeoJSON schemas, projects WGS84 coordinates to EPSG:3857 with fixed formulas, aspect-fits both
layers into the declared scene, and passes only projected scene coordinates to Qt.
