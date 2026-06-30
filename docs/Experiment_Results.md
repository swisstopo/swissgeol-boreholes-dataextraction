
# Classification

## Overview (BERT Only)

| Dataset                            | Support (num classes) | Target | F1-macro | F1-micro |
|------------------------------------|-----------------------|--------|----------|----------|
| `accessory_components`             |               75 (66) |  Multi |        - |        - |
| `alteration_degree_consolidated`   |             2,716 (8) | Single |        - |        - |
| `alteration_degree_unconsolidated` |                41 (8) | Single |        - |        - |
| `cementation`                      |                78 (7) | Single |        - |        - |
| `color_consolidated`*              |           16,143 (91) | Single |    0.487 |    0.752 |
| `color_unconsolidated`*            |           20,121 (91) | Single |    0.502 |    0.803 |
| `debris`                           |           70,084, (7) |  Multi |        - |        - |
| `en_main`                          |           89,842 (34) | Single |    0.859 |    0.905 |
| `grain_angularity`                 |            70,377 (8) |  Multi |    0.831 |    0.974 |
| `grain_shape`                      |            70,087 (5) |  Multi |        - |        - |
| `lithology`                        |           45,323 (61) | Single |    0.848 |    0.942 |
| `mineral_components`               |              63 (111) |  Multi |        - |        - |
| `organic_components`               |           70,102 (11) |  Multi |        - |        - |
| `uscs`                             |            9,917 (38) | Single |    0.329 |    0.602 |

* Model jointly trained, same for both tasks.


## Color (Consolidated + Unconsolidated)

| Test set   | Train set      | Bedrock F1-macro | Bedrock F1-micro | BERT F1-macro | BERT F1-micro |
|------------|----------------|------------------|------------------|---------------|---------------|
| Thurgau-C  | Consolidated   | 0.255            | 0.655            | 0.360         | 0.753         |
| Thurgau-C  | Unconsolidated | -                | -                | 0.409         | 0.723         |
| Thurgau-C  | All            | -                | -                | 0.487         | 0.752         |
| Thurgau-U  | Consolidated   | -                | -                | 0.193         | 0.687         |
| Thurgau-U  | Unconsolidated | 0.335            | 0.703            | 0.442         | 0.797         |
| Thurgau-U  | All            | -                | -                | 0.502         | 0.803         |


## EN Main

| Test set  | Bedrock F1-macro | Bedrock F1-micro | BERT F1-macro | BERT F1-micro |
|-----------|------------------|------------------|---------------|---------------|
| Deepwells | -                | -                | 1.000         | 1.000         |
| Geoquat   | 0.676            | 0.861            | 0.847         | 0.914         |
| Nagra     | 0.911            | 0.966            | 1.000         | 1.000         |
| Thurgau   | 0.515            | 0.812            | 0.595         | 0.868         |
| Overall   | -                | -                | 0.859         | 0.905         |


## Grain angularity

| Test set | Bedrock F1-macro | Bedrock F1-micro | BERT F1-macro | BERT F1-micro |
|----------|------------------|------------------|---------------|---------------|
| Geoquat  | 0.821            | 0.966            | 0.823         | 0.974         |
| Nagra    | 0.660            | 0.623            | 0.963         | 0.986         |
| Overall  | -                | -                | 0.831         | 0.974         |


## Lithology

| Test set  | Bedrock F1-macro | Bedrock F1-micro | BERT F1-macro | BERT F1-micro |
|-----------|------------------|------------------|---------------|---------------|
| Deepwells | 0.469            | 0.374            | 0.808         | 0.947         |
| Geoquat   | 0.456            | 0.745            | 0.529         | 0.872         |
| Lithology | 0.735            | 0.925            | 0.798         | 0.942         |
| Nagra     | 0.504            | 0.968            | 0.842         | 0.992         |
| Thurgau   | 0.363            | 0.864            | 0.576         | 0.916         |
| Overall   | -                | -                | 0.848         | 0.942         |

## USCS

| Test set | Bedrock F1-macro | Bedrock F1-micro | BERT F1-macro | BERT F1-micro |
|----------|------------------|------------------|---------------|---------------|
| Geoquat  | 0.251            | 0.538            | 0.329         | 0.602         |


