# Classification

## Overview (BERT Only)

All figures in the overview reflect the actual task performance, with the change relative to the previous model shown in parentheses.

| Dataset                            | Enhanced | Support (num classes) | Target | F1-macro       | F1-micro       | Kendall's Tau |
|------------------------------------|----------|-----------------------|--------|----------------|----------------|---------------|
| `accessory_components`             | x        |          152,083 (47) |  Multi |          0.842 |          0.972 |             - |
| `alteration_degree_consolidated`   |          |             2,716 (8) | Single | 0.211 (+0.051) | 0.333 (−0.105) |             - |
| `alteration_degree_unconsolidated` |          |                41 (8) | Single |              - |              - |             - |
| `cementation`                      | x        |           119,404 (7) | Single |          0.851 |          0.959 |             - |
| `color_consolidated`*              |          |           16,143 (91) | Single |          0.487 |          0.752 |             - |
| `color_unconsolidated`*            |          |           20,121 (91) | Single |          0.502 |          0.803 |             - |
| `debris`                           |          |           70,084, (7) |  Multi |          0.797 |          0.983 |             - |
| `en_main`                          |          |           89,842 (34) | Single |          0.859 |          0.905 |             - |
| `en_secondary`                     |          |           89,842 (34) |   Rank |          0.808 |          0.945 |         0.744 |
| `grain_angularity`                 |          |            70,377 (8) |  Multi |          0.831 |          0.974 |             - |
| `grain_shape`                      |          |            70,087 (5) |  Multi |          0.560 |          0.998 |             - |
| `lithology`                        |          |           45,323 (61) | Single |          0.848 |          0.942 |             - |
| `mineral_components`               | x        |         141,825 (111) |  Multi |          0.794 |          0.984 |             - |
| `organic_components`               |          |           70,102 (11) |  Multi |          0.839 |          0.983 |             - |
| `uscs`                             |          |            9,917 (38) | Single |          0.329 |          0.602 |             - |

* Model jointly trained, same for both tasks.


## Accessory Components - Enhanced

| Test set  | Bedrock F1-macro | Bedrock F1-micro | BERT F1-macro | BERT F1-micro |
|-----------|------------------|------------------|---------------|---------------|
| Deepwells | -                | -                | 0.721         | 0.914         |
| Geoquat   | -                | -                | 0.560         | 0.980         |
| Nagra     | -                | -                | 0.924         | 0.978         |
| Zurich    | -                | -                | 0.600         | 0.963         |
| Extra-NS  | -                | -                | 1.000         | 1.000         |
| Extra-KW  | -                | -                | 0.318         | 0.298         |
| Overall   | -                | -                | 0.765         | 0.966         |


## Alteration Degree (Consolidated)

| Test set   | Train set    | Bedrock F1-macro | Bedrock F1-micro | BERT F1-macro    | BERT F1-micro    |
|------------|--------------|------------------|------------------|------------------|------------------|
| Geoquat-C  | Consolidated | 0.179 (+0.000)   | 0.536 (+0.000)   | 0.296 (−0.096)   | 0.724 (−0.058)   |
| Extra-NS-C | Consolidated | -                | -                | 1.000 (+1.000)   | 1.000 (+1.000)   |
| Extra-KW-C | Consolidated | -                | -                | 0.436 (+0.365)   | 0.472 (+0.321)   |
| Extra-CF-C | Consolidated | -                | -                | 0.786 (+0.692)   | 0.803 (+0.656)   |
| Global-C   | Consolidated | -                | -                | 0.774 (+0.614)   | 0.788 (+0.350)   |


## Cementation - Enhanced

| Test set  | Bedrock F1-macro | Bedrock F1-micro | BERT F1-macro | BERT F1-micro |
|-----------|------------------|------------------|---------------|---------------|
| Deepwells | -                | -                | 0.774         | 0.891         |
| Geoquat   | -                | -                | 0.785         | 0.956         |
| Nagra     | -                | -                | 0.982         | 0.997         |
| Zurich    | -                | -                | 0.725         | 0.949         |
| Extra-NS  | -                | -                | 1.000         | 1.000         |
| Extra-KW  | -                | -                | 0.271         | 0.250         |
| Extra-CF  | -                | -                | 0.179         | 0.260         |
| Overall   | -                | -                | 0.745         | 0.933         |


## Color (Consolidated + Unconsolidated)

| Test set   | Train set      | Bedrock F1-macro | Bedrock F1-micro | BERT F1-macro | BERT F1-micro |
|------------|----------------|------------------|------------------|---------------|---------------|
| Thurgau-C  | Consolidated   | 0.255            | 0.655            | 0.360         | 0.753         |
| Thurgau-U  | Consolidated   | -                | -                | 0.193         | 0.687         |


| Test set   | Train set      | Bedrock F1-macro | Bedrock F1-micro | BERT F1-macro | BERT F1-micro |
|------------|----------------|------------------|------------------|---------------|---------------|
| Thurgau-C  | Unconsolidated | -                | -                | 0.409         | 0.723         |
| Thurgau-U  | Unconsolidated | 0.335            | 0.703            | 0.442         | 0.797         |


| Test set   | Train set      | Bedrock F1-macro | Bedrock F1-micro | BERT F1-macro | BERT F1-micro |
|------------|----------------|------------------|------------------|---------------|---------------|
| Thurgau-C  | All            | -                | -                | 0.487         | 0.752         |
| Thurgau-U  | All            | -                | -                | 0.502         | 0.803         |
| Extra-NS   | All            | -                | -                | 0.099         | 0.425         |
| Extra-KW   | All            | -                | -                | 0.090         | 0.130         |
| Extra-CF   | All            | -                | -                | 0.214         | 0.271         |
| Overall-C  | All            | -                | -                | 0.207         | 0.667         |
| Overall-U  | All            | -                | -                | 0.323         | 0.728         |


## Debris

| Test set | Bedrock F1-macro | Bedrock F1-micro | BERT F1-macro | BERT F1-micro |
|----------|------------------|------------------|---------------|---------------|
| Geoquat  | 0.595            | 0.926            | 0.797         | 0.983         |
| Extra-NS | -                | -                | 0.499         | 0.995         |
| Extra-KW | -                | -                | 0.183         | 0.135         |
| Overall  | -                | -                | 0.690         | 0.979         |



## EN Main

| Test set  | Bedrock F1-macro | Bedrock F1-micro | BERT F1-macro | BERT F1-micro |
|-----------|------------------|------------------|---------------|---------------|
| Deepwells | 0.400            | 0.920            | 1.000         | 1.000         |
| Geoquat   | 0.676            | 0.861            | 0.847         | 0.914         |
| Nagra     | 0.911            | 0.966            | 1.000         | 1.000         |
| Thurgau   | 0.515            | 0.812            | 0.595         | 0.868         |
| Extra-NS  | -                | -                | 0.034         | 0.285         |
| Extra-KW  | -                | -                | 0.509         | 0.535         |
| Extra-CF  | -                | -                | 0.537         | 0.566         |
| Overall   | -                | -                | 0.722         | 0.877         |


## EN Secondary

|           | Bedrock (v1) |          |               | BERT (Rank) |          |               |
|-----------|--------------|----------|---------------|-------------|----------|---------------|
| Test set  | F1-macro     | F1-micro | Kendall's Tau | F1-macro    | F1-micro | Kendall's Tau |
| Deepwells | 0.766        | 0.966    | 0.832         | 0.629       | 0.966    |  0.965        |
| Geoquat   | 0.736        | 0.9196   | 0.621         | 0.829       | 0.948    |  0.780        |
| Nagra     | 0.668	       | 0.963	  | 0.866         | 0.880       | 0.991    |  0.798        |
| Thurgau   | 0.581	       | 0.885	  | 0.595         | 0.646       | 0.932    |  0.616        |
| Extra-NS  | -            | -        | -             | 0.017       | 0.087    | -0.660        |
| Extra-KW  | -            | -        | -             | 0.481       | 0.512    |  0.104        |
| Extra-CF  | -            | -        | -             | 0.422       | 0.324    |  0.121        |
| Overall   | -            | -        | -             | 0.733       | 0.919    |  0.691        |


## Grain Angularity

| Test set | Bedrock F1-macro | Bedrock F1-micro | BERT F1-macro | BERT F1-micro |
|----------|------------------|------------------|---------------|---------------|
| Geoquat  | 0.821            | 0.966            | 0.823         | 0.974         |
| Nagra    | 0.660            | 0.623            | 0.963         | 0.986         |
| Extra-NS | -                | -                | 0.249         | 0.988         |
| Extra-KW | -                | -                | 0.418         | 0.330         |
| Overall  | -                | -                | 0.814         | 0.972         |


## Grain Shape

| Test set | Bedrock F1-macro | Bedrock F1-micro | BERT F1-macro | BERT F1-micro |
|----------|------------------|------------------|---------------|---------------|
| Geoquat  | 0.629            | 0.990            | 0.560         | 0.998         |
| Extra-NS | -                | -                | 1.000         | 1.000         |
| Extra-KW | -                | -                | 0.146         | 0.118         |
| Overall  | -                | -                | 0.548         | 0.997         |


## Lithology

| Test set  | Bedrock F1-macro | Bedrock F1-micro | BERT F1-macro | BERT F1-micro |
|-----------|------------------|------------------|---------------|---------------|
| Deepwells | 0.490            | 0.384            | 0.784         | 0.938         |
| Geoquat   | 0.548            | 0.747            | 0.585         | 0.865         |
| Lithology | 0.767            | 0.931            | 0.858         | 0.960         |
| Nagra     | 0.579            | 0.973            | 0.912         | 0.992         |
| Thurgau   | 0.406            | 0.855            | 0.537         | 0.912         |
| Extra-NS  | -                | -                | 1.000         | 1.000         |
| Extra-KW  | -                | -                | 0.988         | 0.984         |
| Extra-CF  | -                | -                | 0.608         | 0.738         |
| Overall   | -                | -                | 0.929         | 0.941         |


## Mineral Components - Enhanced

| Test set  | Bedrock F1-macro | Bedrock F1-micro | BERT F1-macro | BERT F1-micro |
|-----------|------------------|------------------|---------------|---------------|
| Deepwells | -                | -                | 0.780         | 0.942         |
| Geoquat   | -                | -                | 0.739         | 0.997         |
| Nagra     | -                | -                | 0.924         | 0.984         |
| Zurich    | -                | -                | 0.369         | 0.975         |
| Extra-NS  | -                | -                | 1.000         | 1.000         |
| Extra-KW  | -                | -                | 0.346         | 0.348         |
| Overall   | -                | -                | 0.563         | 0.973         |


## Organic Components

| Test set | Bedrock F1-macro | Bedrock F1-micro | BERT F1-macro | BERT F1-micro |
|----------|------------------|------------------|---------------|---------------|
| Geoquat  | 0.841            | 0.976            | 0.839         | 0.983         |
| Extra-NS | -                | -                | 1.000         | 1.000         |
| Extra-KW | -                | -                | 0.631         | 0.596         |
| Overall  | -                | -                | 0.833         | 0.981         |


## USCS

| Test set | Bedrock F1-macro | Bedrock F1-micro | BERT F1-macro | BERT F1-micro |
|----------|------------------|------------------|---------------|---------------|
| Geoquat  | 0.251            | 0.538            | 0.329         | 0.602         |
| Extra-NS | -                | -                | 0.000         | 0.xxx         |
| Extra-KW | -                | -                | 0.161         | 0.204         |
| Extra-CF | -                | -                | 0.115         | 0.229         |
| Overall  | -                | -                | 0.245         | 0.483         |
