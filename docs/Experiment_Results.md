# Classification

## Overview (BERT Only)

| Dataset                            | Enhanced | Support (num classes) | Target | F1-macro | F1-micro | Kendall's Tau |
|------------------------------------|----------|-----------------------|--------|----------|----------|---------------|
| `accessory_components`             | x        |          152,083 (47) |  Multi |    0.842 |    0.972 |             - |
| `alteration_degree_consolidated`   |          |             2,716 (8) | Single |    0.392 |    0.782 |             - |
| `alteration_degree_unconsolidated` |          |                41 (8) | Single |        - |        - |             - |
| `cementation`                      | x        |           119,404 (7) | Single |    0.851 |    0.959|             - |
| `color_consolidated`*              |          |           16,143 (91) | Single |    0.487 |    0.752 |             - |
| `color_unconsolidated`*            |          |           20,121 (91) | Single |    0.502 |    0.803 |             - |
| `debris`                           |          |           70,084, (7) |  Multi |    0.797 |    0.983 |             - |
| `en_main`                          |          |           89,842 (34) | Single |    0.859 |    0.905 |             - |
| `en_secondary`                     |          |           89,842 (34) |   Rank |    0.808 |    0.945 |         0.744 |
| `grain_angularity`                 |          |            70,377 (8) |  Multi |    0.831 |    0.974 |             - |
| `grain_shape`                      |          |            70,087 (5) |  Multi |    0.560 |    0.998 |             - |
| `lithology`                        |          |           45,323 (61) | Single |    0.848 |    0.942 |             - |
| `mineral_components`               | x        |         141,825 (111) |  Multi |    0.794 |    0.984 |             - |
| `organic_components`               |          |           70,102 (11) |  Multi |    0.839 |    0.983 |             - |
| `uscs`                             |          |            9,917 (38) | Single |    0.329 |    0.602 |             - |

* Model jointly trained, same for both tasks.

## Accessory Components - Enhanced

| Test set  | Bedrock F1-macro | Bedrock F1-micro | BERT F1-macro | BERT F1-micro |
|-----------|------------------|------------------|---------------|---------------|
| Deepwells | -                | -                | 0.721         | 0.914         |
| Geoquat   | -                | -                | 0.560         | 0.980         |
| Nagra     | -                | -                | 0.924         | 0.978         |
| Zurich    | -                | -                | 0.600         | 0.963         |
| Overall   | -                | -                | 0.842         | 0.972         |


## Alteration Degree (Consolidated)

| Test set  | Train set    | Bedrock F1-macro | Bedrock F1-micro | BERT F1-macro | BERT F1-micro |
|-----------|--------------|------------------|------------------|---------------|---------------|
| Geoquat-C | Consolidated | 0.179            | 0.536            | 0.392         | 0.782         |


## Cementation - Enhanced

| Test set  | Bedrock F1-macro | Bedrock F1-micro | BERT F1-macro | BERT F1-micro |
|-----------|------------------|------------------|---------------|---------------|
| Deepwells | -                | -                | 0.774         | 0.891         |
| Geoquat   | -                | -                | 0.785         | 0.956         |
| Nagra     | -                | -                | 0.982         | 0.997         |
| Zurich    | -                | -                | 0.725         | 0.949         |
| Overall   | -                | -                | 0.851         | 0.959         |


## Color (Consolidated + Unconsolidated)

| Test set   | Train set      | Bedrock F1-macro | Bedrock F1-micro | BERT F1-macro | BERT F1-micro |
|------------|----------------|------------------|------------------|---------------|---------------|
| Thurgau-C  | Consolidated   | 0.255            | 0.655            | 0.360         | 0.753         |
| Thurgau-C  | Unconsolidated | -                | -                | 0.409         | 0.723         |
| Thurgau-C  | All            | -                | -                | 0.487         | 0.752         |
| Thurgau-U  | Consolidated   | -                | -                | 0.193         | 0.687         |
| Thurgau-U  | Unconsolidated | 0.335            | 0.703            | 0.442         | 0.797         |
| Thurgau-U  | All            | -                | -                | 0.502         | 0.803         |


## Debris

| Test set | Bedrock F1-macro | Bedrock F1-micro | BERT F1-macro | BERT F1-micro |
|----------|------------------|------------------|---------------|---------------|
| Geoquat  | 0.595            | 0.926            | 0.797         | 0.983         |


## EN Main

| Test set  | Bedrock F1-macro | Bedrock F1-micro | BERT F1-macro | BERT F1-micro |
|-----------|------------------|------------------|---------------|---------------|
| Deepwells | 0.400            | 0.920            | 1.000         | 1.000         |
| Geoquat   | 0.676            | 0.861            | 0.847         | 0.914         |
| Nagra     | 0.911            | 0.966            | 1.000         | 1.000         |
| Thurgau   | 0.515            | 0.812            | 0.595         | 0.868         |
| Overall   | -                | -                | 0.859         | 0.905         |


## EN Secondary

|           | Bedrock (v1) |          |               | BERT (Rank) |          |               |
|-----------|--------------|----------|---------------|-------------|----------|---------------|
| Test set  | F1-macro     | F1-micro | Kendall's Tau | F1-macro    | F1-micro | Kendall's Tau |
| Deepwells | 0.xxx        | 0.xxx    | 0.xxx         | 0.629       | 0.966    | 0.965         |
| Geoquat   | 0.xxx        | 0.xxx    | 0.xxx         | 0.829       | 0.948    | 0.778         |
| Nagra     | 0.xxx        | 0.xxx    | 0.xxx         | 0.880       | 0.991    | 0.798         |
| Thurgau   | 0.xxx        | 0.xxx    | 0.xxx         | 0.646       | 0.932    | 0.617         |
| Overall   | -            | -        | -             | 0.808       | 0.945    | 0.744         |


| Prompt   | Test set  | Bedrock F1-macro | Bedrock F1-micro | Bedrock Kendall's Tau |
|----------|-----------|------------------|------------------|-----------------------|
| Baseline | Deepwells | 0.768            | 0.970            | -                     |
|          | Geoquat   | 0.649            | 0.933            | -                     |
|          | Nagra     | 0.514            | 0.954            | -                     |
|          | Thurgau   | 0.575            | 0.895            | -                     |
|          | Overall   | -                | -                | -                     |
| v1       | Deepwells | 0.xxx            | 0.xxx            | 0.xxx                 |
|          | Geoquat   | 0.xxx            | 0.xxx            | 0.xxx                 |
|          | Nagra     | 0.xxx            | 0.xxx            | 0.xxx                 |
|          | Thurgau   | 0.xxx            | 0.xxx            | 0.xxx                 |
|          | Overall   | -                | -                | -                     |

| Loss | Test set  | BERT F1-macro | BERT F1-micro | BERT Kendall's Tau |
|------|-----------|---------------|---------------|--------------------|
| XEnt | Deepwells | 0.696         | 0.979         | -0.096             |
|      | Geoquat   | 0.840         | 0.951         |  0.244             |
|      | Nagra     | 1.000         | 1.000         |  0.195             |
|      | Thurgau   | 0.650         | 0.933         |  0.202             |
|      | Overall   | 0.854         | 0.947         |  -                 |
| Rank | Deepwells | 0.629         | 0.966         |  0.965             |
|      | Geoquat   | 0.829         | 0.948         |  0.778             |
|      | Nagra     | 0.880         | 0.991         |  0.798             |
|      | Thurgau   | 0.646         | 0.932         |  0.617             |
|      | Overall   | 0.808         | 0.945         |  0.744             |


## Grain Angularity

| Test set | Bedrock F1-macro | Bedrock F1-micro | BERT F1-macro | BERT F1-micro |
|----------|------------------|------------------|---------------|---------------|
| Geoquat  | 0.821            | 0.966            | 0.823         | 0.974         |
| Nagra    | 0.660            | 0.623            | 0.963         | 0.986         |
| Overall  | -                | -                | 0.831         | 0.974         |


## Grain Shape

| Test set | Bedrock F1-macro | Bedrock F1-micro | BERT F1-macro | BERT F1-micro |
|----------|------------------|------------------|---------------|---------------|
| Geoquat  | 0.629            | 0.990            | 0.560         | 0.998         |


## Lithology

| Test set  | Bedrock F1-macro | Bedrock F1-micro | BERT F1-macro | BERT F1-micro |
|-----------|------------------|------------------|---------------|---------------|
| Deepwells | 0.469            | 0.374            | 0.808         | 0.947         |
| Geoquat   | 0.456            | 0.745            | 0.529         | 0.872         |
| Lithology | 0.735            | 0.925            | 0.798         | 0.942         |
| Nagra     | 0.504            | 0.968            | 0.842         | 0.992         |
| Thurgau   | 0.363            | 0.864            | 0.576         | 0.916         |
| Overall   | -                | -                | 0.848         | 0.942         |


## Mineral Components - Enhanced

| Test set  | Bedrock F1-macro | Bedrock F1-micro | BERT F1-macro | BERT F1-micro |
|-----------|------------------|------------------|---------------|---------------|
| Deepwells | -                | -                | 0.78          | 0.942         |
| Geoquat   | -                | -                | 0.739         | 0.997         |
| Nagra     | -                | -                | 0.924         | 0.984         |
| Zurich    | -                | -                | 0.369         | 0.984         |
| Overall   | -                | -                | 0.794         | 0.984         |

## Organic Components

| Test set | Bedrock F1-macro | Bedrock F1-micro | BERT F1-macro | BERT F1-micro |
|----------|------------------|------------------|---------------|---------------|
| Geoquat  | 0.841            | 0.976            | 0.839         | 0.983         |


## USCS

| Test set | Bedrock F1-macro | Bedrock F1-micro | BERT F1-macro | BERT F1-micro |
|----------|------------------|------------------|---------------|---------------|
| Geoquat  | 0.251            | 0.538            | 0.329         | 0.602         |

