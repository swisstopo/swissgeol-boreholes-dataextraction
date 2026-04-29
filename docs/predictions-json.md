# `predictions.json` Output Structure

The `predictions.json` file contains the results of a data extraction process in a machine-readable format. By default, the file is written to `data/output/predictions.json`.


## Schema Overview

```text
predictions.json
└── "<filename>.pdf"                      # One entry per processed PDF file
    ├── file_metadata
    │   ├── language                      # Detected language
    │   └── page_dimensions[]             # Dimensions of each page, in PDF points
    │
    ├── boreholes[]                       # All boreholes identified in the document
    │   ├── borehole_index                # Zero-based index for extracted boreholes
    │   ├── metadata
    │   │   ├── elevation                 # Borehole surface elevation, if found
    │   │   └── coordinates               # Borehole coordinates, if found
    │   ├── layers[]                      # Detected layers of the borehole profil
    │   │   ├── material_description      # Material text description
    │   │   └── depth_interval            # Measured depth of the layer's upper and lower limits
    │   ├── bounding_boxes[]              # List of bounding boxes, one for each part of a borehole profile, that can be used for visualizations
    │   │   ├── page                      # Page number for this segment
    │   │   ├── sidebar_rect              # Area of the depth sidebar, if found
    │   │   ├── depth_column_entries[]    # Bounding boxes of depth entries
    │   │   └── material_description_rect # Bounding boxes of material descriptions
    │   └── groundwater[]                 # Groundwater measurements extracted from the PDF
    │       ├── date                      # Measurement date (YYYY-MM-DD), if found
    │       ├── depth                     # Depth of the measurement (m), if found
    │       ├── elevation                 # Elevation (m above sea level), if found
    │       ├── page                      # Page number of detected groundwater
    │       └── rect                      # Bounding box of detected groundwater
    │
    └── metrics                           # Evaluation scores with tp, fp, fn, precision, recall, and f1
        ├── language                      # Ground truth language for metric computation
        ├── layer_metrics                 # Layer detection
        ├── depth_interval_metrics        # Depth layer detection
        ├── material_description_metrics  # Material description detection
        ├── gw_metrics
        │   ├── metrics                   # Groundwater overall detection
        │   ├── depth_metrics             # Groundwater depth detection
        │   ├── elevation_metrics         # Groundwater elevation detection
        │   └── date_metrics              # Groundwater date detection
        └── metadata_metrics
            ├── elevation                 # Borehole elevation detection
            ├── coordinates               # Borehole coordinates detection
            └── name                      # Borehole name detection
```

All page numbers are counted starting at 1. All bounding boxes are measured with PDF points as the unit, and with the top-left of the page as the origin.


## Example output
```jsonc
{
  // filename
  "B366.pdf": {
    "file_metadata": {
      "language": "de",
      "page_dimensions": [
        {
          "width": 591.956787109375,
          "height": 1030.426025390625
        },
        {
          "width": 588.009521484375,
          "height": 792.114990234375
        }
      ],
    },
    "boreholes": [
      {
        "borehole_index": 0,
        "metadata": {
          "elevation": {
            "elevation": 355.35,
            "page": 1,
            "rect": [27.49843978881836, 150.2817840576172, 159.42971801757812, 160.76754760742188]
          },
          "coordinates": {
            "E": 659490.0,
            "N": 257200.0,
            "rect": [28.263830184936523, 179.63882446289062, 150.3379364013672, 188.7487335205078],
            "page": 1
          }
        },
        "layers": [
          {
            "material_description": {
              "text": "beiger, massig-dichter, stark dolomitisierter Kalk, mit Muschelresten",
              "lines": [
                {
                  "text": "beiger, massig-dichter, stark",
                  "page": 1,
                  "rect": [258.5303039550781, 345.9997253417969, 379.9410705566406, 356.1011657714844]
                },
                {
                  "text": "dolomitisierter Kalk, mit",
                  "page": 1,
                  "rect": [258.2362060546875, 354.4559326171875, 363.0706787109375, 364.295654296875]
                },
                {
                  "text": "Muschelresten",
                  "page": 1,
                  "rect": [258.48748779296875, 363.6712341308594, 313.03204345703125, 371.3343505859375]
                }
              ],
              "page": 1,
              "rect": [258.2362060546875, 345.9997253417969, 379.9410705566406, 371.3343505859375]
            },
            "depth_interval": {
              "start": {
                "value": 1.5,
                "rect": [200.63790893554688, 331.3035888671875, 207.83108520507812, 338.30450439453125]
              },
              "end": {
                "value": 6.0,
                "rect": [201.62551879882812, 374.30560302734375, 210.0361328125, 380.828857421875]
              }
            }
          }
          // ... (more layers)
        ],
        "bounding_boxes": [
          {
            "sidebar_rect": [198.11251831054688, 321.8956298828125, 210.75906372070312, 702.2628173828125],
            "depth_column_entries": [
              [200.1201171875, 321.8956298828125, 208.59901428222656, 328.6802062988281],
              [200.63790893554688, 331.3035888671875, 207.83108520507812, 338.30450439453125],
              [201.62551879882812, 374.30560302734375, 210.0361328125, 380.828857421875],
              [199.86251831054688, 434.51556396484375, 210.10894775390625, 441.4538879394531],
              [198.11251831054688, 557.5472412109375, 210.35877990722656, 563.9244995117188],
              [198.28451538085938, 582.0216674804688, 209.76953125, 588.7603759765625],
              [198.7814178466797, 616.177001953125, 209.50042724609375, 622.502197265625],
              [198.6378173828125, 663.2830810546875, 210.75906372070312, 669.5428466796875],
              [198.26901245117188, 695.974609375, 209.12693786621094, 702.2628173828125]
            ],
            "material_description_rect": [256.777099609375, 345.9997253417969, 392.46051025390625, 728.2700805664062],
            "page": 1
          },
          {
            "sidebar_rect": null,
            "depth_column_entries": [],
            "material_description_rect": [192.3216094970703, 337.677978515625, 291.1827392578125, 633.6331176757812],
            "page": 2
          }
        ],
        "groundwater": [
          {
            "date": "1979-11-29",
            "depth": 19.28,
            "elevation": 336.07,
            "page": 1,
            "rect": [61.23963928222656, 489.3185119628906, 94.0096435546875, 513.6478881835938]
          }
        ]
      }
      // ... (more boreholes)
    ],
    "metrics": {
      "layer_metrics": {
        "tp": 2,
        "fp": 18,
        "fn": 1,
        "precision": 0.1,
        "recall": 0.6666666666666666,
        "f1": 0.1739130434782609
      },
      "depth_interval_metrics": {
        "tp": 2,
        "fp": 18,
        "fn": 1,
        "precision": 0.1,
        "recall": 0.6666666666666666,
        "f1": 0.1739130434782609
      },
      "material_description_metrics": {
        "tp": 2,
        "fp": 18,
        "fn": 1,
        "precision": 0.1,
        "recall": 0.6666666666666666,
        "f1": 0.1739130434782609
      },
      "gw_metrics": {
        "metrics": {
          "tp": 0,
          "fp": 1,
          "fn": 0,
          "precision": 0.0,
          "recall": 0,
          "f1": 0
        },
        "depth_metrics": {
          "tp": 0,
          "fp": 1,
          "fn": 0,
          "precision": 0.0,
          "recall": 0,
          "f1": 0
        },
        "elevation_metrics": {
          "tp": 0,
          "fp": 1,
          "fn": 0,
          "precision": 0.0,
          "recall": 0,
          "f1": 0
        },
        "date_metrics": {
          "tp": 0,
          "fp": 1,
          "fn": 0,
          "precision": 0.0,
          "recall": 0,
          "f1": 0
        }
      },
      "metadata_metrics": {
        "elevation": {
          "tp": 0,
          "fp": 1,
          "fn": 1,
          "precision": 0.0,
          "recall": 0.0,
          "f1": 0
        },
        "coordinates": {
          "tp": 1,
          "fp": 0,
          "fn": 0,
          "precision": 1.0,
          "recall": 1.0,
          "f1": 1.0
        },
        "name": {
          "tp": 1,
          "fp": 0,
          "fn": 0,
          "precision": 1.0,
          "recall": 1.0,
          "f1": 1.0
        }
      }
    }
  }
}
```
