# Employee Sentiment Analysis

## Overview
End-to-end Python analysis of the supplied unlabeled `test.csv` dataset for the Employee Sentiment Analysis assessment.

### Required tasks implemented
1. Sentiment labeling: Positive / Neutral / Negative
2. Exploratory Data Analysis and visualizations
3. Monthly employee sentiment scoring
4. Monthly employee ranking
5. Rolling 30-day flight-risk identification
6. Linear regression for monthly sentiment trends

## Dataset
- Records: **2,191**
- Columns: `Subject`, `body`, `date`, `from`
- Senders: **10**
- Date range: **2010-01-01 to 2011-12-31**
- Missing values: **0**
- Duplicate rows: **0**

## Methodology

### Sentiment labeling
The dataset is unlabeled. We combine `Subject + body` and calculate TextBlob polarity.

- Polarity > `0.10` → Positive
- Polarity from `-0.10` through `0.10` → Neutral
- Polarity < `-0.10` → Negative

A small, explicitly documented correction list handles a few high-confidence short conversational phrases. No external API key is required.

### Monthly score
- Positive = `+1`
- Negative = `-1`
- Neutral = `0`

Scores are summed by employee and calendar month, so scores reset each month.

### Ranking
For every month, the top three highest scores and bottom three lowest scores are selected. Ties are resolved alphabetically.

Full results: `output/employee_rankings.csv`

### Flight risk
A flight risk is an employee with **4 or more negative emails in any rolling 30-day period**. The implementation uses actual dates and is independent of calendar-month boundaries.

### Regression
Target: monthly sentiment score.

Features:
- monthly message count
- average word count
- average character count

Model: `sklearn.linear_model.LinearRegression`

80/20 train/test split, `random_state=42`.

Results:
- MAE: **1.6343**
- RMSE: **2.3904**
- R²: **0.3673**

## Key results

### Sentiment distribution
- Positive: **978 (44.6%)**
- Neutral: **1,018 (46.5%)**
- Negative: **195 (8.9%)**

### Latest month — 2011-12

Top 3 Positive:

| Rank | Employee | Score |
|---:|---|---:|
| 1 | kayne.coulter@enron.com | 5 |
| 2 | patti.thompson@enron.com | 5 |
| 3 | don.baughman@enron.com | 4 |

Top 3 Negative:

| Rank | Employee | Score |
|---:|---|---:|
| 1 | bobette.riner@ipgdirect.com | 0 |
| 2 | lydia.delgado@enron.com | 0 |
| 3 | johnny.palmer@enron.com | 1 |

The complete month-by-month ranking is in `output/employee_rankings.csv`.

### Flight-risk result
**6 employees** were flagged. See `output/flight_risk.csv`.

## Repository structure
```text
employee-sentiment-analysis/
├── Employee_Sentiment_Analysis.ipynb
├── README.md
├── requirements.txt
├── data/
│   └── test.csv
├── output/
├── visualization/
└── report/
    └── Final_Report.md
```

## Setup

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install:
```bash
pip install -r requirements.txt
```

Launch:
```bash
jupyter notebook
```

Open `Employee_Sentiment_Analysis.ipynb` and run all cells.

## Reproducibility
The notebook generates all labeled data, scores, rankings, flight-risk results, regression metrics, and visualizations from `data/test.csv`.

No `.env` is required because no external API is used.

## Internal-data handling
This assessment dataset and its results are internal evaluation materials. Keep the GitHub repository private unless the evaluator explicitly authorizes public sharing.

## Limitations
TextBlob is a lightweight reproducible sentiment baseline rather than a domain-specific transformer/LLM. Email quoting, forwarding, sarcasm, and business-specific language can affect classification. The regression analysis is exploratory and does not establish causality.

## Author
Jeevan Jadhav
