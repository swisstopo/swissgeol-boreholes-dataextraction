# `ground_truth.json` Input Structure

The `ground_truth.json` file provides annotated reference data used to compute extraction metrics and train classification models. See [train_BERT.md](train_BERT.md) for details on model training.

## Schema Overview

We denote with `*` all elements that are mandatory. All other elements can be omitted.

```text
ground_truth.json
└── "<filename>.pdf"                        # One entry per PDF file
    └── []                                  # List of boreholes in that PDF
        ├── borehole_index*                 # Zero-based index (0, 1, …)
        ├── metadata*
        │   ├── coordinates                 # Borehole location
        │   │   ├── E*                      # Easting
        │   │   └── N*                      # Northing
        │   ├── drilling_date               # Date in YYYY-MM-DD format
        │   ├── drilling_methods[]          # List of drilling method identifiers
        │   ├── original_name               # Borehole identifier in the source document
        │   ├── project_name                # Project / report name
        │   ├── reference_elevation         # Surface elevation (m above sea level)
        │   └── total_depth                 # Total borehole depth (m)
        │
        ├── layers*[]
        │   ├── depth_interval*
        │   │   ├── start                   # Start depth (m)
        │   │   └── end                     # End depth (m)
        │   ├── material_description        # Free-text lithology description
        │   ├── consolidated                # Classification for rock layers
        │   │   ├── lithology               # Rock or sediment type
        │   │   ├── uscs[]                  # USCS (Unified Soil Classification System) class codes
        │   │   ├── primary_color           # Dominant colour
        │   │   ├── cementation             # Cementation degree or type
        │   │   ├── accessory_components[]  # Minor mineral or clast components
        │   │   ├── mineral_components[]    # Primary mineral components
        │   │   └── alteration_degree       # Weathering / alteration degree
        │   └── unconsolidated              # Classification for loose sediment layers
        │       ├── main                    # Dominant grain type (EN two-level code, e.g. "Ba")
        │       ├── other[]                 # Secondary grain types
        │       ├── uscs[]                  # USCS class codes
        │       ├── primary_color           # Dominant colour
        │       ├── grain_shape[]           # Grain shape descriptors
        │       ├── grain_angularity[]      # Grain angularity descriptors
        │       ├── debris[]                # Debris or clast types
        │       ├── organic_components[]    # Organic material components
        │       └── alteration_degree       # Weathering / alteration degree
        │
        └── groundwater[]                   # Groundwater measurements
            ├── date                        # Measurement date (YYYY-MM-DD)
            ├── depth*                      # Groundwater depth (m)
            └── elevation*                  # Elevation (m above sea level)
```

All depth and elevation values are in **meters**. Layer depths must be provided in increasing order, with `start` ≤ `end` when both are present.

`material_description` may be absent from the JSON file and populated later using `generate_material_description_gt.py`. All classification fields (`consolidated`, `unconsolidated`, and their sub-fields) are optional and omitted from serialisation when `null`.


## Example

```jsonc
{
  "example.pdf": [
    {
      "borehole_index": 0,
      "metadata": {
        "coordinates": { "E": 499936.0, "N": 116004.0 },
        "drilling_date": "1961-06-08",
        "drilling_methods": null,
        "original_name": "Forage Nº 5",
        "project_name": "Pont de Carouge - Genève",
        "reference_elevation": 380.0,
        "total_depth": 40.32
      },
      "layers": [
        {
          // Unconsolidated layer
          "depth_interval": { "start": 0.05, "end": 0.3 },
          "material_description": "Gravier sableux, légèrement limoneux, galets toutes formes, dm. 10 cm, avec débris de construction, compact, sec.",
          "unconsolidated": {
            "main": "Ba",
            "other": ["gr", "si", "sa", "co"]
          }
        },
        {
          // Consolidated layer
          "depth_interval": { "start": 0.3, "end": 2.5 },
          "material_description": "Roche en place: phyllade argileux noir avec des plans de strastification dépolis et brillants.",
          "consolidated": {
            "alteration_degree": "fresh",
            "lithology": "phyllite"
          }
        }
      ],
      "groundwater": [
        { "date": "1961-07-14", "depth": 3.2, "elevation": 376.8 }
      ]
    }
  ]
}
```
