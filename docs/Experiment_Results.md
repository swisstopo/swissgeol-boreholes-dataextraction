# Classification

## Overview (BERT Only)

All figures in the overview reflect the actual task performance, with the change relative to the previous model shown in parentheses.

| Dataset                            | Enhanced | Support (num classes) | Target | F1-macro       | F1-micro       | Kendall's Tau |
|------------------------------------|----------|-----------------------|--------|----------------|----------------|---------------|
| `accessory_components`             | x        |          152,083 (47) |  Multi |          0.842 |          0.972 |             - |
| `alteration_degree_consolidated`   |          |             2,716 (8) | Single |          0.324 |          0.663 |             - |
| `alteration_degree_unconsolidated` |          |                41 (8) | Single |              - |              - |             - |
| `cementation`                      | x        |           119,404 (7) | Single |          0.851 |          0.959 |             - |
| `color_consolidated`*              |          |           16,143 (91) | Single |          0.228 |          0.716 |             - |
| `color_unconsolidated`*            |          |           20,121 (91) | Single |          0.353 |          0.757 |             - |
| `debris`                           |          |           70,084, (7) |  Multi |          0.694 |          0.980 |             - |
| `en_main`                          |          |           89,842 (34) | Single |          0.744 |          0.889 |             - |
| `en_secondary`                     |          |           89,842 (34) |   Rank |          0.775 |          0.926 |         0.714 |
| `grain_angularity`                 |          |            70,377 (8) |  Multi |          0.831 |          0.974 |             - |
| `grain_shape`                      |          |            70,087 (5) |  Multi |          0.560 |          0.998 |             - |
| `lithology`                        |          |           45,323 (61) | Single |          0.xxx |          0.xxx |             - |
| `mineral_components`               | x        |         141,825 (111) |  Multi |          0.794 |          0.984 |             - |
| `organic_components`               |          |           70,102 (11) |  Multi |          0.839 |          0.983 |             - |
| `uscs`                             |          |            9,917 (38) | Single |          0.300 |          0.597 |             - |


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
| Geoquat-C  | Consolidated | 0.179            | 0.536            | 0.305 (−0.087)   | 0.763 (−0.019)   |
| Extra-NS-C | Consolidated | -                | -                | 1.000 (+1.000)   | 1.000 (+1.000)   |
| Extra-KW-C | Consolidated | -                | -                | 0.079 (+0.008)   | 0.132 (−0.019)   |
| Extra-CF-C | Consolidated | -                | -                | 0.098 (+0.004)   | 0.140 (−0.007)   |
| Global-C   | Consolidated | -                | -                | 0.324 (+0.164)   | 0.663 (+0.225)   |


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

| Test set   | Train set | Bedrock F1-macro | Bedrock F1-micro | BERT F1-macro    | BERT F1-micro    |
|------------|-----------|------------------|------------------|------------------|------------------|
| Thurgau-C  | All       | -                | -                | 0.503 (+0.016)   | 0.759 (+0.007)   |
| Thurgau-U  | All       | -                | -                | 0.514 (+0.012)   | 0.794 (−0.009)   |
| Extra-NS   | All       | -                | -                | 0.499 (+0.400)   | 0.995 (+0.570)   |
| Extra-KW   | All       | -                | -                | 0.118 (+0.028)   | 0.130 (+0.000)   |
| Extra-CF-C | All       | -                | -                | 0.153 (−0.061)   | 0.188 (−0.083)   |
| Extra-CF-U | All       | -                | -                | 0.238 (+0.011)   | 0.297 (+0.016)   |
| Overall-C  | All       | -                | -                | 0.228 (+0.021)   | 0.716 (+0.049)   |
| Overall-U  | All       | -                | -                | 0.353 (+0.030)   | 0.757 (+0.029)   |


## Debris

| Test set | Bedrock F1-macro | Bedrock F1-micro | BERT F1-macro    | BERT F1-micro    |
|----------|------------------|------------------|------------------|------------------|
| Geoquat  | 0.595 (+0.000)   | 0.926 (+0.000)   | 0.824 (+0.027)   | 0.983 (+0.000)   |
| Extra-NS | -                | -                | 1.000 (+0.501)   | 1.000 (+0.005)   |
| Extra-KW | -                | -                | 0.149 (−0.034)   | 0.114 (−0.021)   |
| Overall  | -                | -                | 0.694 (+0.004)   | 0.980 (+0.001)   |


## EN Main

| Test set  | Bedrock F1-macro | Bedrock F1-micro | BERT F1-macro    | BERT F1-micro    |
|-----------|------------------|------------------|------------------|------------------|
| Deepwells | 0.400            | 0.920            | 1.000 (+0.000)   | 1.000 (+0.000)   |
| Geoquat   | 0.676            | 0.861            | 0.851 (+0.004)   | 0.916 (+0.002)   |
| Nagra     | 0.911            | 0.966            | 1.000 (+0.000)   | 1.000 (+0.000)   |
| Thurgau   | 0.515            | 0.812            | 0.573 (−0.022)   | 0.866 (−0.002)   |
| Extra-NS  | -                | -                | 0.497 (+0.463)   | 0.990 (+0.705)   |
| Extra-KW  | -                | -                | 0.573 (+0.064)   | 0.596 (+0.061)   |
| Extra-CF  | -                | -                | 0.529 (−0.008)   | 0.556 (−0.010)   |
| Overall   | -                | -                | 0.744 (+0.022)   | 0.889 (+0.012)   |


## EN Secondary
|           | Bedrock        |                 |                 | BERT           |                |                 |
|-----------|----------------|-----------------|-----------------|----------------|----------------|-----------------|
| Test set  | F1-macro       | F1-micro        | Kendall's Tau   | F1-macro       | F1-micro       | Kendall's Tau   |
| Deepwells | 0.766          | 0.966           | 0.832           | 0.693 (+0.064) | 0.975 (+0.009) | 0.965 (+0.000)  |
| Geoquat   | 0.736          | 0.920           | 0.621           | 0.837 (+0.008) | 0.949 (+0.001) | 0.773 (−0.007)  |
| Nagra     | 0.668          | 0.963           | 0.866           | 0.991 (+0.111) | 0.994 (+0.003) | 0.831 (+0.033)  |
| Thurgau   | 0.581          | 0.885           | 0.595           | 0.646 (+0.000) | 0.930 (−0.002) | 0.620 (+0.004)  |
| Extra-NS  | -              | -               | -               | 1.000 (+0.983) | 1.000 (+0.913) | 1.000 (+1.660)  |
| Extra-KW  | -              | -               | -               | 0.530 (+0.049) | 0.529 (+0.017) | 0.156 (+0.052)  |
| Extra-CF  | -              | -               | -               | 0.424 (+0.002) | 0.329 (+0.005) | 0.173 (+0.052)  |
| Overall   | -              | -               | -               | 0.775 (+0.042) | 0.926 (+0.007) | 0.714 (+0.023)  |


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

| Test set | Bedrock F1-macro | Bedrock F1-micro | BERT F1-macro    | BERT F1-micro    |
|----------|------------------|------------------|------------------|------------------|
| Geoquat  | 0.251            | 0.538            | 0.316 (−0.013)   | 0.600 (−0.002)   |
| Extra-NS | -                | -                | 1.000 (+1.000)   | 1.000 (+1.000)   |
| Extra-KW | -                | -                | 0.171 (+0.010)   | 0.213 (+0.009)   |
| Extra-CF | -                | -                | 0.076 (−0.039)   | 0.202 (−0.027)   |
| Overall  | -                | -                | 0.300 (+0.055)   | 0.597 (+0.114)   |

