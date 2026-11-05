# Phase 3 gate results (2026-11-05)

Uniform 200,000-row samples of each Kaggle dataset (seed 20261105), made by fetch_samples.sh. Data stays in ~/lh-data, outside the repo.

### Phase 3 gate: four public datasets, no schema code

| dataset | rows | result | role check (correct / soft / wrong-asked / wrong-settled) | seconds |
| --- | --- | --- | --- | --- |
| paysim | 200,000 | PASS | 11 / 0 / 0 / 0 | 4.8 |
| ulb | 200,000 | PASS | 31 / 0 / 0 / 0 | 4.6 |
| ieee | 200,000 | PASS | 394 / 0 / 0 / 0 | 30.5 |
| baf | 200,000 | PASS | 30 / 2 / 0 / 0 | 6.6 |

Label maturity: none of the four has a label-time column, so every sample is built on the assumption route (labels stated final by the person) and the certificate records it as an assumption, not a check.

### paysim

- ok  no role wrong and settled
- ok  loads
- ok  roles confirmed
- ok  sql
- ok  check_feature runs
- ok  isFlaggedFraud refused as a feature
- ok  sample CERTIFIED
- ok  maturity recorded as assumed
- ok  finding recorded
- ok  report saved
- ok  report says maturity assumed

```
sample built -- CERTIFIED (certificate 4ed71251b05022bf)
  split=grouped_temporal, entity=nameOrig, time=step
  train=1,840 rows, test=8,174 rows (125,736 legitimate train rows down-sampled away)
  label maturity: 0 days, assumed
  prevalence: design 0.05, natural 0.0013
  [WARNING     ] label_maturity
```

### ulb

- ok  no role wrong and settled
- ok  loads
- ok  roles confirmed
- ok  sql
- ok  check_feature runs
- ok  sample CERTIFIED
- ok  maturity recorded as assumed
- ok  finding recorded
- ok  report saved
- ok  report says maturity assumed

```
sample built -- CERTIFIED (certificate 8cf77b7bd813dc82)
  split=temporal, time=Time
  train=5,620 rows, test=40,000 rows (154,380 legitimate train rows down-sampled away)
  label maturity: 0 days, assumed
  prevalence: design 0.05, natural 0.001675
  [WARNING     ] label_maturity
```

### ieee

- ok  no role wrong and settled
- ok  loads
- ok  roles confirmed
- ok  sql
- ok  check_feature runs
- ok  TransactionID refused as a feature
- ok  sample CERTIFIED
- ok  maturity recorded as assumed
- ok  finding recorded
- ok  report saved
- ok  report says maturity assumed

```
sample built -- CERTIFIED (certificate 0a92374b921d28ff)
  split=grouped_temporal, entity=card1, time=TransactionDT
  train=132,209 rows, test=7,121 rows
  label maturity: 0 days, assumed
  prevalence: design 0.03529, natural 0.03529
  [WARNING     ] label_maturity
```

### baf

- ok  no role wrong and settled
- ok  loads
- ok  roles confirmed
- ok  sql
- ok  check_feature runs
- ok  sample CERTIFIED
- ok  maturity recorded as assumed
- ok  finding recorded
- ok  report saved
- ok  report says maturity assumed

```
sample built -- CERTIFIED (certificate cfbf2f53d789d5e3)
  split=temporal, time=month
  train=32,560 rows, test=40,963 rows (126,477 legitimate train rows down-sampled away)
  label maturity: 0 days, assumed
  prevalence: design 0.05, natural 0.01087
  [WARNING     ] label_maturity
```

### Phase 3 role check

| dataset | columns | correct | soft | wrong, asked | wrong, settled | asked in confirm_roles |
| --- | --- | --- | --- | --- | --- | --- |
| paysim | 11 | 11 | 0 | 0 | 0 | 4 |
| ulb | 31 | 31 | 0 | 0 | 0 | 2 |
| ieee | 394 | 394 | 0 | 0 | 0 | 4 |
| baf | 32 | 30 | 2 | 0 | 0 | 5 |

### paysim (200,000 sampled rows)

Inferred: step=event_time, type=channel, amount=amount, nameOrig=payer, nameDest=payee, isFraud=label, isFlaggedFraud=rule_output; 4 feature column(s); not in the data: label_time, label_source

| column | inferred | confidence | status | expected | verdict |
| --- | --- | --- | --- | --- | --- |
| step | event_time | low | proposed | event_time | correct |
| type | channel | medium | proposed | channel | correct |
| amount | amount | high | auto | amount | correct |
| nameOrig | payer | high | auto | payer | correct |
| nameDest | payee | high | auto | payee | correct |
| isFraud | label | high | proposed | label | correct |
| isFlaggedFraud | rule_output | medium | proposed | rule_output | correct |

### ulb (200,000 sampled rows)

Inferred: Time=event_time, Amount=amount, Class=label; 28 feature column(s); not in the data: label_time, label_source

| column | inferred | confidence | status | expected | verdict |
| --- | --- | --- | --- | --- | --- |
| Time | event_time | medium | proposed | event_time | correct |
| Amount | amount | high | auto | amount | correct |
| Class | label | high | proposed | label | correct |

### ieee (200,000 sampled rows)

Inferred: TransactionID=transaction_id, isFraud=label, TransactionDT=event_time, TransactionAmt=amount, ProductCD=channel, card1=card; 388 feature column(s); not in the data: label_time, label_source

| column | inferred | confidence | status | expected | verdict |
| --- | --- | --- | --- | --- | --- |
| TransactionID | transaction_id | high | auto | transaction_id | correct |
| isFraud | label | high | proposed | label | correct |
| TransactionDT | event_time | medium | proposed | event_time | correct |
| TransactionAmt | amount | high | auto | amount | correct |
| ProductCD | channel | medium | proposed | channel | correct |
| card1 | card | medium | proposed | card/feature | correct |

### baf (200,000 sampled rows)

Inferred: fraud_bool=label, intended_balcon_amount=amount, payment_type=channel, credit_risk_score=score, month=event_time; 27 feature column(s); not in the data: label_time, label_source

| column | inferred | confidence | status | expected | verdict |
| --- | --- | --- | --- | --- | --- |
| fraud_bool | label | high | proposed | label | correct |
| intended_balcon_amount | amount | high | auto | amount/feature | correct |
| payment_type | channel | medium | proposed | channel/feature | correct |
| credit_risk_score | score | medium | proposed | feature/score | soft |
| source | feature | high | auto | channel/feature | soft |
| month | event_time | low | proposed | event_time | correct |
