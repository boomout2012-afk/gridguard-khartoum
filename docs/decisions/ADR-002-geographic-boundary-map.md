# ADR-002: Use pinned geographic locality boundaries with an offline projection

- Status: Accepted
- Date: 2026-08-15
- Decision owners: GridGuard Khartoum project team

## Context

The original desktop application used a hand-authored schematic to make the seven localities
interactive without a network or GIS dependency. That was operationally safe but could not provide
recognizable administrative geography during the academic presentation. Replacing it with a live
web map would make the primary demonstration depend on connectivity, remote service availability,
and mutable upstream data. Adding a desktop GIS stack would also enlarge and complicate the Windows
package for a display-only concern.

The geographic layer must therefore be real, attributable, reproducible, packaged with the
application, and incapable of being mistaken for electrical-network topology or measured system
data.

## Decision

Use OCHA/HDX Sudan Common Operational Dataset administrative boundaries, revision `v03` and valid
from 2020-08-31, as the locality authority. Select only `ADM1_PCODE=SD01` and the exact ADM2 P-codes
`SD01001` through `SD01007`. Preserve the source English and Arabic names and P-codes while mapping
three public display names to the established GridGuard vocabulary: Um Bada to Ombada, Sharg An
Neel to East Nile, and Karrari to Karari; Um Durman is displayed as Omdurman.

Use Natural Earth 1:10m rivers and lake centerlines version 5.0.0 for visual orientation. Select the
White Nile feature 278 component 0, Blue Nile feature 1075 component 0, and Nile feature 732
component 4 after asserting each feature's class, number, and English name. Connect Natural Earth's
Blue Nile endpoint to the shared confluence coordinate by the recorded 258.07 m adjustment, then
clip all three lines against the dissolved seven-locality mask.

Perform acquisition and transformation only in `tools/build_map_asset.py`. The builder verifies the
pinned source sizes and SHA-256 hashes and writes:

- `khartoum-localities.geojson`, retaining raw WGS84 Polygon/MultiPolygon rings and holes;
- `khartoum-nile.geojson`, retaining the clipped WGS84 linework; and
- `map-provenance.json`, binding source pins, selectors, transformations, and derived-file hashes.

At runtime, use a closed schema-version-2 loader implemented with the Python standard library. It
verifies provenance before semantic parsing, rejects unknown or duplicate fields, checks the exact
seven source identities and keyboard graph, and validates finite geographic geometry. It supports
Polygon and MultiPolygon features with interior rings. One fixed transform converts all locality,
anchor, and river coordinates from WGS84 to EPSG:3857, aspect-fits them into the declared 1000 by
720 scene with 36 units of padding, and inverts the vertical axis for Qt. Raw longitude and latitude
are never exposed as scene coordinates.

The permanent disclaimer states that boundaries are administrative context, results are synthetic,
and the drawing is not electrical-network topology. Full attribution and transformation details are
bundled in `src/gridguard/assets/map/THIRD_PARTY_DATA.md`.

## Source pins and licensing

The OCHA archive is pinned at SHA-256
`75330d4e12e53f966df0e5683e04f0919e9964952fbbf827f3729038e603c95e`; its
`sdn_admin2.geojson` member is pinned at
`c13cf1e8515730267d41bd560a1f1a0489552ab2900abb070a9aececacca1bb9`. The data is licensed under
CC BY 3.0 IGO.

The Natural Earth GeoJSON is pinned to commit
`b70b7b8766c3c83d5f5145cb6f1c93921d383d3f` at SHA-256
`bb854a900ecbd3b408df46d5e16e3e0f974ba55993f9d8b5c26e855273c0905a`. Natural Earth releases
these data into the public domain.

## Alternatives considered

### Keep the schematic only

This remains fully offline and simple, but it does not meet the presentation need for recognizable
administrative boundaries. The old asset was removed after the geographic renderer, loader, and
package-data tests passed; it is not retained as a silent runtime fallback.

### Load a live web basemap

Rejected because it introduces an internet requirement, remote tracking and service dependencies,
mutable visual output, and additional licensing obligations.

### Bundle a GIS runtime

Rejected because a display-only map does not justify a second geometry engine, native GIS binaries,
or their packaging and security surface. Build-time clipping plus small standard-library runtime
validation is sufficient.

### Hand-trace geographic-looking polygons

Rejected because imitating geography without an authoritative source would be less defensible than
the explicit schematic and would have no reproducible provenance.

## Consequences

- The presentation gains recognizable, source-attributed administrative geography while remaining
  fully offline.
- Map-source changes become deliberate migrations: update pins, rebuild, review the derived diff,
  and update this decision record.
- Runtime tampering or schema drift blocks application startup instead of silently displaying a
  different map.
- The builder contains modest line-clipping logic, while the shipped runtime remains free of GIS
  dependencies.
- Administrative boundaries and river centerlines must not be used to infer feeders, substations,
  service territories, measured demand, or engineering topology.
