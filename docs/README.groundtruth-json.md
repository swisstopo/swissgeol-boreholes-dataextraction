# `ground_truth.json` input structure

This repository expects **ground-truth training/evaluation data** to be provided as a single JSON file.

This document describes the expected format based on the example Zurich ground-truth file.

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

- `coordinates` *(object)*  
  - `E` *(number)* — Easting  
  - `N` *(number)* — Northing 

- `drilling_date` *(string)*  
  Date in **`YYYY-MM-DD`** format.

- `drilling_methods` *(any / null)*  
  May be `null` if unknown. 

- `original_name` *(string)*  
  Original borehole identifier/name in the source document. 

- `project_name` *(string)*  
  Project/report name.

- `reference_elevation` *(number)*  
  Reference elevation in meters above sea level. 

- `total_depth` *(number)*  
  Total borehole depth in meters. 

---

## `layers` array

`layers` is a list of layer objects. Each layer object contains:

- `depth_interval` *(object)*  
  - `start` *(number | null)* — start depth in meters  
  - `end` *(number | null)* — end depth in meters 

- `material_description` *(string)*  
  Free-text lithology/material description for the interval. 

### Depth interval conventions

- Depths are in **meters**.
- `start` should be **<=** `end` when both are present
- Provide layers in **increasing depth order**.

---

## `groundwater` array 

`groundwater` is a list of groundwater measurement objects:

- `date` *(string)* — date in **`YYYY-MM-DD`** format  
- `depth` *(number)* — measured groundwater depth in meters  
- `elevation` *(number)* — elevation in meters above sea level 
---

## Example

Below is a condensed example showing all main fields (one PDF with one borehole).
Values are illustrative but follow the structure used in the Zurich ground-truth file. 

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

