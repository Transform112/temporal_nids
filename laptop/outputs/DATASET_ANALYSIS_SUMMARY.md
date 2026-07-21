# Dataset Analysis Summary

## All Datasets — Split Verification

| Dataset | Total Flows | Train | Val | Test | Time Span | Attack% (Train) | Attack% (Val) | Attack% (Test) |
|---|---|---|---|---|---|---|---|---|
| NF-CICIDS2018 | 20,115,529 | 14,080,870 | 3,017,329 | 3,017,330 | 391.6h | 15.7% | 4.5% | 8.6% | 
| NF-UNSW-NB15 | 2,365,424 | 1,655,796 | 354,813 | 354,815 | 648.7h | 3.9% | 8.8% | 8.8% | 
| NF-ToN-IoT | 27,520,260 | 19,264,182 | 4,128,039 | 4,128,039 | 144.6h | 29.2% | 78.8% | 45.0% | 
| NF-BoT-IoT | 16,933,808 | 11,853,665 | 2,540,071 | 2,540,072 | 843.8h | 99.6% | 100.0% | 100.0% | 

## Combined Training Distribution (CICIDS2018 + UNSW-NB15)

| Class | Count | % | Minority? | CVAE Needed |
|---|---|---|---|---|
| Benign | 13,466,429 | 85.574% |  | 0 |
| DoS/DDoS | 1,630,554 | 10.361% |  | 3,756,017 |
| Reconnaissance | 8,950 | 0.057% |  | 5,377,621 |
| Exploits | 39,015 | 0.248% |  | 5,347,556 |
| Backdoor | 3,253 | 0.021% | YES | 5,383,318 |
| Bot | 0 | 0.000% | YES | 5,386,571 |
| Brute Force | 575,194 | 3.655% |  | 4,811,377 |
| Web Attack | 2,538 | 0.016% | YES | 5,384,033 |
| Infiltration | 0 | 0.000% | YES | 5,386,571 |
| Generic | 9,243 | 0.059% |  | 5,377,328 |
| Shellcode/Worms | 1,490 | 0.009% | YES | 5,385,081 |

- **Total training flows:** 15,736,666
- **Imbalance ratio:** 13466429.0:1
- **Minority classes (CVAE targets):** ['Backdoor', 'Bot', 'Web Attack', 'Infiltration', 'Shellcode/Worms']
- **Time split gaps:** All < 1 second (essentially contiguous)
- **No time overlap in any dataset** — chronological split is clean
