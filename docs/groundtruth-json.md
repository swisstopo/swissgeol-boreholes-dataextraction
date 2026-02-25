# `ground_truth.json` input structure

This repository expects **ground-truth training/evaluation data** to be provided as a single JSON file.

This document describes the expected format of the ground-truth file.

---

## Top-level structure

The JSON file is a single object (dictionary / map):

- **Key**: `"<pdf_filename>.pdf"` (string) — the name of the PDF file.
- **Value**: `boreholes` (array) — the list of boreholes contained in that PDF.

```json
{
  "<pdf_filename>.pdf": [
    { "borehole_index": 0, "metadata": { ... }, "layers": [ ... ], "groundwater": [ ... ] }
  ]
}
```

Note:

A single PDF can contain **multiple boreholes** (e.g. `borehole_index: 0`, `borehole_index: 1`, …). 


---

## Borehole object

Each item in the per-PDF list is a **borehole object** with the following keys:

- `borehole_index` *(integer)*  
  Index to differentiate boreholes inside a single PDF. Starts at `0`. 

- `metadata` *(object)*  
  Borehole-level metadata (see below). 

- `layers` *(array)*  
  List of lithological layers (depth intervals + material descriptions). 

- `groundwater` *(array)*  
  List of groundwater measurements (date + depth + elevation). 

---

## `metadata` object

The `metadata` object stores borehole-level information. In the Zurich example, it contains:

- `coordinates` *(object, required)*  
  - `E` *(number, required)* — Easting  
  - `N` *(number, required)* — Northing 

- `drilling_date` *(string, optional)*  
  Date in **`YYYY-MM-DD`** format.

- `drilling_methods` *(any / null, optional)*  
  May be `null` if unknown. 

- `original_name` *(string, optional)*  
  Original borehole identifier/name in the source document. 

- `project_name` *(string, optional)*  
  Project/report name.

- `reference_elevation` *(number, optional)*  
  Reference elevation in meters above sea level. 

- `total_depth` *(number, optional)*  
  Total borehole depth in meters. 

---

## `layers` array

`layers` is a list of layer objects. Each layer object contains:

- `depth_interval` *(object, required)*  
  - `start` *(number | null, required)* — start depth in meters  
  - `end` *(number | null, required)* — end depth in meters 

- `material_description` *(string, required)*  
  Free-text lithology/material description for the interval. 

### Depth interval conventions

- Depths are in **meters**.
- `start` should be **<=** `end` when both are present
- Provide layers in **increasing depth order**.

### `classification`attributes
For classififcation of the material descriptions from the ground truth or from the predictions, 
following optional attributes can be added to the `layers`array. 
- `lithology` *(string, optional)* 
  Describes the rock or sediment type.
- `uscs_1` *(string, optional)* 
  Key used to retrieve the ground truth USCS (Unified Soil Classification System) class from a layer dictionary.
- `uscs_2`*(string, optional)* 
  Optional secondary classification.
- `unconsolidated` *(object, optional)* 
  Contains the EN two-level geological classification of loose sediments.
  - `main` *(string, optional)* 
    The dominant grain type. 
  - `other` *(array, optional)* 
    Lists secondary grain types present in smaller proportions.

---

## `groundwater` array 

`groundwater` is a list of groundwater measurement objects:

- `date` *(string)* — date in **`YYYY-MM-DD`** format  
- `depth` *(number)* — measured groundwater depth in meters  
- `elevation` *(number)* — elevation in meters above sea level 
---

## Example

Below is a condensed example showing all main fields (one PDF with one borehole).

```json
{
  "680248008-bp.pdf": [
    {
      "borehole_index": 0,
      "metadata": {
        "coordinates": { "E": 680995, "N": 248040 },
        "drilling_date": "1972-01-01",
        "drilling_methods": null,
        "original_name": "75",
        "project_name": "Oelunfall Hardstrasse-Albisriederplatz",
        "reference_elevation": 411.83,
        "total_depth": 20.0
      },
      "layers": [
        {
          "depth_interval": { "start": 0.0, "end": 0.2 },
          "material_description": "Betonbelag"
        },
        {
          "depth_interval": { "start": 0.2, "end": 0.6 },
          "material_description": "Kies mit sandigem Lehm"
        }
      ],
      "groundwater": [
        { "date": "1972-06-26", "depth": 14.13, "elevation": 397.7 }
      ]
    }
  ]
}
```

Below is a condensed example showing the structure of a ground truth file with classification attributes. 
```json 
    {"680.pdf": [
        {
            "borehole_index": 0,
            "groundwater": null,
            "layers": [
                {
                    "depth_interval": {
                        "end": 0.3,
                        "start": 0.05
                    },
                    "lithology": "unconsolidated deposits",
                    "material_description": "Gravier sableux, légèrement limoneux, galets toutes formes, dm. 10 cm, avec débris de construction, compact, sec.",
                    "unconsolidated": {
                        "main": "Ba",
                        "other": [
                            "gr",
                            "si",
                            "sa",
                            "co"
                        ]
                    },
                    "uscs_1": null,
                    "uscs_2": null
                
                }
            ]            
                "metadata": {
                "coordinates": {
                    "E": 499936.0,
                    "N": 116004.0
                },
                "drilling_date": "1961-06-08",
                "drilling_methods": null,
                "original_name": "Forage Nº 5",
                "project_name": "Pont de Carouge - Genève",
                "reference_elevation": 380.0,
                "total_depth": 40.32
            }
        }
    ]
}
