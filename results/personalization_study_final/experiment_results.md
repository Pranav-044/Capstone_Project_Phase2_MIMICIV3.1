| Experiment | Global AUC | Mean client AUC | Worst client AUC | Client AUC SD | Comm. MB/round | Savings | Time (s) |
|---|---|---|---|---|---|---|---|
| FedAvg baseline | 0.7956 | 0.7958 | 0.7081 | 0.0525 | 0.0865 | 0.0% | 129.1 |
| MCFL personalization | 0.8025 | 0.7952 | 0.7367 | 0.0408 | 0.0865 | 0.0% | 210.0 |
| Full personalization (default) | 0.8176 | 0.7817 | 0.6538 | 0.0680 | 0.0781 | 9.8% | 231.8 |
| Full: Ditto λ=0.001 | 0.8252 | 0.8020 | 0.7340 | 0.0499 | 0.0781 | 9.8% | 243.7 |
| Full: Ditto λ=0.1 | 0.8225 | 0.8209 | 0.7867 | 0.0352 | 0.0781 | 9.8% | 250.6 |
| Full: MCFL lr=0.03 | 0.8209 | 0.7767 | 0.6184 | 0.0864 | 0.0781 | 9.8% | 212.2 |
| Full: MCFL lr=0.30 | 0.8247 | 0.7511 | 0.5318 | 0.1129 | 0.0781 | 9.8% | 160.4 |
