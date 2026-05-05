# Human Study Aggregation Report

## 1) 审计范围与规则

- 扫描目录: `/data/xueyanz/code/PDI-Bench/humanstudy`
- 发现 xlsx 文件数: `5`
- 评分方向: `1 为最好，10 为最差（越低越好）`
- 案例行识别规则: 在某个 section 下，单行至少 3 个模型存在数值
- 有效分数规则: 数值且落在 `[1,10]`
- 缺失值处理: 不填补，按有效样本统计

## 2) 文件级审计结果

| rater | file | models | sections | valid_cases | valid_scores | missing_cells | non_numeric | out_of_range |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| lyh | lyh.xlsx | 7 | 5 | 15 | 104 | 1 | 0 | 0 |
| mfy | mfy.xlsx | 7 | 5 | 15 | 105 | 0 | 0 | 0 |
| pyh | pyh.xlsx | 7 | 5 | 15 | 105 | 0 | 0 | 0 |
| tcc | tcc.xlsx | 7 | 5 | 15 | 104 | 1 | 0 | 0 |
| wjx | wjx.xlsx | 7 | 5 | 15 | 105 | 0 | 0 | 0 |

### 审计警告

- 无结构性警告。

## 3) 总体模型排名（全量汇总）

| rank | model | mean | std | n |
|---:|---|---:|---:|---:|
| 1 | GT | 1.068 | 0.253 | 73 |
| 2 | seedance | 2.640 | 1.881 | 75 |
| 3 | cogvideoX | 2.693 | 1.356 | 75 |
| 4 | wan22 | 2.800 | 1.833 | 75 |
| 5 | Flow | 3.187 | 1.555 | 75 |
| 6 | Sora | 3.227 | 1.678 | 75 |
| 7 | hunyuan | 3.640 | 2.544 | 75 |

## 4) 分维度排名（Section-wise）

### Biological Motion

| rank | model | mean | n |
|---:|---|---:|---:|
| 1 | GT | 1.000 | 15 |
| 2 | hunyuan | 2.467 | 15 |
| 3 | seedance | 2.533 | 15 |
| 4 | wan22 | 2.600 | 15 |
| 5 | Sora | 2.867 | 15 |
| 6 | cogvideoX | 3.200 | 15 |
| 7 | Flow | 3.867 | 15 |

### Curved Motion

| rank | model | mean | n |
|---:|---|---:|---:|
| 1 | GT | 1.154 | 13 |
| 2 | cogvideoX | 2.400 | 15 |
| 3 | wan22 | 2.467 | 15 |
| 4 | seedance | 2.800 | 15 |
| 5 | Flow | 3.267 | 15 |
| 6 | Sora | 3.267 | 15 |
| 7 | hunyuan | 7.133 | 15 |

### Dynamic Tracking

| rank | model | mean | n |
|---:|---|---:|---:|
| 1 | GT | 1.067 | 15 |
| 2 | seedance | 1.867 | 15 |
| 3 | cogvideoX | 2.667 | 15 |
| 4 | Sora | 3.133 | 15 |
| 5 | hunyuan | 3.200 | 15 |
| 6 | Flow | 3.400 | 15 |
| 7 | wan22 | 4.467 | 15 |

### Longitudinal Convergence

| rank | model | mean | n |
|---:|---|---:|---:|
| 1 | GT | 1.067 | 15 |
| 2 | Flow | 1.667 | 15 |
| 3 | wan22 | 2.267 | 15 |
| 4 | hunyuan | 2.333 | 15 |
| 5 | cogvideoX | 2.400 | 15 |
| 6 | Sora | 3.400 | 15 |
| 7 | seedance | 4.267 | 15 |

### Partial Occlusion

| rank | model | mean | n |
|---:|---|---:|---:|
| 1 | GT | 1.067 | 15 |
| 2 | seedance | 1.733 | 15 |
| 3 | wan22 | 2.200 | 15 |
| 4 | cogvideoX | 2.800 | 15 |
| 5 | hunyuan | 3.067 | 15 |
| 6 | Sora | 3.467 | 15 |
| 7 | Flow | 3.733 | 15 |

## 5) 评分者一致性参考（每位评分者的模型均值）

### lyh

| rank | model | mean | n |
|---:|---|---:|---:|
| 1 | GT | 1.071 | 14 |
| 2 | seedance | 3.667 | 15 |
| 3 | Sora | 3.733 | 15 |
| 4 | wan22 | 3.800 | 15 |
| 5 | hunyuan | 3.867 | 15 |
| 6 | cogvideoX | 3.933 | 15 |
| 7 | Flow | 4.267 | 15 |

### mfy

| rank | model | mean | n |
|---:|---|---:|---:|
| 1 | GT | 1.200 | 15 |
| 2 | wan22 | 1.867 | 15 |
| 3 | hunyuan | 2.600 | 15 |
| 4 | seedance | 2.800 | 15 |
| 5 | cogvideoX | 2.867 | 15 |
| 6 | Flow | 3.067 | 15 |
| 7 | Sora | 3.467 | 15 |

### pyh

| rank | model | mean | n |
|---:|---|---:|---:|
| 1 | GT | 1.000 | 15 |
| 2 | cogvideoX | 1.733 | 15 |
| 3 | Sora | 2.000 | 15 |
| 4 | seedance | 2.000 | 15 |
| 5 | Flow | 2.533 | 15 |
| 6 | wan22 | 2.800 | 15 |
| 7 | hunyuan | 3.333 | 15 |

### tcc

| rank | model | mean | n |
|---:|---|---:|---:|
| 1 | GT | 1.000 | 14 |
| 2 | wan22 | 2.267 | 15 |
| 3 | cogvideoX | 3.000 | 15 |
| 4 | Flow | 3.067 | 15 |
| 5 | Sora | 3.067 | 15 |
| 6 | hunyuan | 3.067 | 15 |
| 7 | seedance | 3.067 | 15 |

### wjx

| rank | model | mean | n |
|---:|---|---:|---:|
| 1 | GT | 1.067 | 15 |
| 2 | seedance | 1.667 | 15 |
| 3 | cogvideoX | 1.933 | 15 |
| 4 | Flow | 3.000 | 15 |
| 5 | wan22 | 3.267 | 15 |
| 6 | Sora | 3.867 | 15 |
| 7 | hunyuan | 5.333 | 15 |

## 6) Section 完整性（每位评分者每个 Section 的案例数）

| rater | section | cases |
|---|---|---:|
| lyh | Biological Motion | 3 |
| lyh | Curved Motion | 3 |
| lyh | Dynamic Tracking | 3 |
| lyh | Longitudinal Convergence | 3 |
| lyh | Partial Occlusion | 3 |
| mfy | Biological Motion | 3 |
| mfy | Curved Motion | 3 |
| mfy | Dynamic Tracking | 3 |
| mfy | Longitudinal Convergence | 3 |
| mfy | Partial Occlusion | 3 |
| pyh | Biological Motion | 3 |
| pyh | Curved Motion | 3 |
| pyh | Dynamic Tracking | 3 |
| pyh | Longitudinal Convergence | 3 |
| pyh | Partial Occlusion | 3 |
| tcc | Biological Motion | 3 |
| tcc | Curved Motion | 3 |
| tcc | Dynamic Tracking | 3 |
| tcc | Longitudinal Convergence | 3 |
| tcc | Partial Occlusion | 3 |
| wjx | Biological Motion | 3 |
| wjx | Curved Motion | 3 |
| wjx | Dynamic Tracking | 3 |
| wjx | Longitudinal Convergence | 3 |
| wjx | Partial Occlusion | 3 |
