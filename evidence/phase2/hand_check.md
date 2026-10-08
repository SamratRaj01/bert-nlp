# Phase 2 hand check

Toy training records: `a`, `a`, `b`; V = {a, b, </s>}; framing `<s> <s> … </s>`.

| quantity | by hand | expected | program | match |
|---|---|---|---|---|
| q2(a | <s>) | `(2+1)/(3+3) = 3/6` | 0.500000000000 | 0.500000000000 | True |
| q2(b | <s>) | `(1+1)/(3+3) = 2/6` | 0.333333333333 | 0.333333333333 | True |
| q2(</s> | <s>) | `(0+1)/(3+3) = 1/6` | 0.166666666667 | 0.166666666667 | True |
| q1(</s>) | `(3+1)/(6+3) = 4/9` | 0.444444444444 | 0.444444444444 | True |
| q2(b | a) | `(0+1)/(2+3) = 1/5` | 0.200000000000 | 0.200000000000 | True |
| corpus H_bits, records 'a b' + 'b' (T = 5) | `-(log2 3/6 + log2 1/5 + log2 2/4 + log2 2/6 + log2 2/4) / 5` | 1.381378119122 | 1.381378119122 | True |
| H_nats / ln 2 | `equals H_bits` | 1.381378119122 | 1.381378119122 | True |
| H_bits, four equally likely targets | `-log2(1/4) = 2` | 2.000000000000 | 2.000000000000 | True |
| PP, four equally likely targets | `2**2 = 4` | 4.000000000000 | 4.000000000000 | True |
| sum_w P_mix(w | <s> <s>), λ = (.25,.25,.5) | `normalized` | 1.000000000000 | 1.000000000000 | True |
