# `predictions.json` output Structure
The `predictions.json` file contains the results of a data extraction process in a machine-readable format. By default, the file is written to `data/output/predictions.json`.

Each key in the JSON object is the name of a PDF file. The extracted data is listed as an object with the following keys:
- `metadata`
  - `elevation`: the detected elevation (if any) and the location in the PDF where they were extraction from.
  - `coordinates`: the detected coordinates (if any) and the location in the PDF where they were extraction from.
  - `language`: language that was detected for the document.
  - `page_dimensions`: dimensions of each page in the PDF, measured in PDF points
- `layers`: a list of objects, where each object represents a layer of the borehole profile, using the following keys:
  - `material_description`: the text of the material description, both as a single value as well as line-by-line, and the location in the PDF where the text resp. the lines where extracted from.
  - `depth_interval`: the measured depth of the upper and lower limits of the layer, and the location in the PDF where they were extracted from.
- `bounding_boxes`: a list of objects, one for each (part of a) borehole profile in the PDF, that list some bounding boxes that can be used for visualizations. Each object has the following keys:
  - `sidebar_rect`: the area of the page the contains a "sidebar" (if any), which contains depths or other data displayed to the side of material descriptions.
  - `depth_column_entries`: list of locations of the entries in the depth column (if any).
  - `material_description_rect`: the area of the page that contains all material descriptions.
  - `page`: the number of the page of the PDF.
- `groundwater`: a list of objects, one for each groundwater measurement that was extracted from the PDF. Each object has the following keys.
  - `date`: extracted date for the groundwater measurement (if any) as a string in YYYY-MM-DD format.
  - `depth`: the measured depth (in m) of the groundwater measurement.
  - `elevation`: the elevation (in m above sea level) of the groundwater measurement.
  - `page` and `rect`: the location in the PDF where the groundwater measurement was extracted from.

All page numbers are counted starting at 1.

All bounding boxes are measured with PDF points as the unit, and with the top-left of the page as the origin.

## Example output
```yaml
{
  "B366.pdf": {  # file name
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
      },
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
      ]
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
      },
      # ... (more layers)
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
}
```