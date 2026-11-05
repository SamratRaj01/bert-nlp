# Candidate spans for the traced examples

Chosen without looking at any outcome: the first spans, in saved order, that meet each structural criterion. Join with Workstream B's model outcomes before using.

## Substitution that changes the pieces (`sub-p0.10-s11`)

| span_id | clean | noisy | clean pieces | noisy pieces | cosine | distance |
|---|---|---|---|---|---|---|
| wt2-valid-000007:0 | `Homarus` | `Homarsi` | ho ##mar ##us | ho ##mar ##si | 0.8813 | 0.4946 |
| wt2-valid-000007:1 | `gammarus` | `gammprus` | gamma ##rus | ga ##mm ##pr ##us | 0.6540 | 0.9274 |
| wt2-valid-000007:2 | `is` | `ls` | is | l ##s | 0.3628 | 0.9719 |

## Insertion (`ins-p0.10-s11`)

| span_id | clean | noisy | clean pieces | noisy pieces | cosine | distance |
|---|---|---|---|---|---|---|
| wt2-valid-000007:1 | `gammarus` | `gdammarus` | gamma ##rus | g ##dam ##mar ##us | 0.6594 | 0.9188 |
| wt2-valid-000007:2 | `is` | `isj` | is | is ##j | 0.6834 | 0.6972 |
| wt2-valid-000007:5 | `crustacean` | `cruistacean` | crust ##ace ##an | cr ##uis ##ta ##cea ##n | 0.7249 | 0.7876 |

## Fragmentation into more pieces (`ins-p0.20-s11`)

| span_id | clean | noisy | clean pieces | noisy pieces | cosine | distance |
|---|---|---|---|---|---|---|
| wt2-valid-000007:1 | `gammarus` | `gammanrusn` | gamma ##rus | gamma ##nr ##us ##n | 0.7602 | 0.7787 |
| wt2-valid-000007:5 | `crustacean` | `crustaceyasn` | crust ##ace ##an | crust ##ace ##yas ##n | 0.9088 | 0.4715 |
| wt2-valid-000007:9 | `body` | `bqojdye` | body | b ##q ##oj ##dy ##e | 0.3617 | 1.1319 |

## Fully deleted span (undefined pair) (`del-p0.20-s11`)

| span_id | clean | noisy | clean pieces | noisy pieces | cosine | distance |
|---|---|---|---|---|---|---|
| wt2-valid-000007:8 | `a` | `` | a |  |  |  |
| wt2-valid-000008:5 | `is` | `` | is |  |  |  |
| wt2-valid-000008:71 | `the` | `` | the |  |  |  |
