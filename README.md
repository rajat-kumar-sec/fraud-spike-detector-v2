# Fraud Spike Detector

**AI Risk Manager — Real-time Transaction Fraud Detection**

Built for the AI Risk Manager hackathon track.

---

## Problem Statement

Payment platforms lose billions annually to fraud. Traditional rule-based systems catch only ~34% of fraud while generating false alarms. This project combines rule-based detection, machine learning, and AI explanations into a single real-time monitoring dashboard.

## Architecture

```
Transaction Data
       │
       ▼
┌─────────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│  Rule Engine    │     │   ML Model          │     │  AI Explainer    │
│  (3 rules)      │────▶│   (Random Forest)   │────▶│  (Gemini/Claude) │
│  - Burst        │     │   - 18 features     │     │  - Natural lang  │
│  - Geo-mismatch │     │   - Class-balanced  │     │  - Risk reasoning│
│  - Odd-hour     │     │   - 80.6% F1        │     │                  │
└─────────────────┘     └─────────────────────┘     └──────────────────┘
       │                         │                          │
       └─────────────────────────┼──────────────────────────┘
                                 ▼
                        ┌─────────────────┐
                        │   Dashboard     │
                        │   (Flask + HTML)│
                        └─────────────────┘
```

**Components:**
1. **Rule Engine** — 3 hand-crafted rules that catch obvious fraud patterns (rapid bursts, geo-mismatches, odd-hour high-value)
2. **ML Model** — Random Forest classifier trained on 18 engineered features with class-weight balancing
3. **AI Explainer** — Gemini/Claude generates natural language explanations for why each transaction is flagged
4. **Dashboard** — Flask backend + HTML/JS frontend with charts, tables, and detail modals

## How to Run Locally

```bash
# Clone the repo
git clone https://github.com/rajat-kumar-sec/fraud-spike-detector-v2.git
cd fraud-spike-detector-v2

# Install dependencies
pip install pandas faker scikit-learn flask google-generativeai

# Generate synthetic data
python generate_data.py

# Train the ML model
python ml_model.py

# Set your API key (optional — mock mode works without it)
set GEMINI_API_KEY=your-key-here

# Start the server
python app.py
```

Open **http://localhost:3000** in your browser.

## Metrics Comparison

| System | Precision | Recall | F1 Score | AUC-ROC |
|---|---|---|---|---|
| Rules Only | 0.868 | 0.341 | 0.489 | — |
| ML + Rules | 0.828 | 0.750 | 0.806 | 0.982 |

The ML model boosts recall from **34% to 75%** while maintaining strong precision — catching fraud patterns that static rules miss. The model was trained with realistic ambiguity (hard negatives/positives) to avoid overfitting.

## Project Structure

```
├── generate_data.py        # Synthetic dataset generator (1900 txns)
├── rule_engine.py          # Rule-based fraud detection (3 rules)
├── ml_model.py             # ML model training + feature engineering
├── explainer.py            # AI explanation layer (Gemini)
├── app.py                  # Flask server + API endpoints
├── run.py                  # Quick start script (sets env vars)
├── static/index.html       # Live dashboard (Flask version)
├── docs/                   # Static demo for GitHub Pages
│   ├── index.html
│   └── data.json
└── transactions.csv        # Generated dataset
```

## Limitations

- **Synthetic data** — This uses generated transactions, not real banking data. Real fraud patterns are more complex and imbalanced.
- **Model performance** — AUC of 0.982 on synthetic data will be lower in production. Real-world performance depends on data quality and feature availability.
- **No real-time streaming** — This batch-processes CSV data, not a live transaction feed.
- **Static explanations** — The AI explainer uses pre-computed text in the static demo. Full AI integration requires a backend API key.
- **No persistence** — Flask serves data from memory. No database, no auth, no user sessions.

## Tech Stack

- **Backend:** Python, Flask, pandas, scikit-learn
- **Frontend:** HTML, CSS, JavaScript, Chart.js
- **AI:** Google Gemini API (or Anthropic Claude)
- **Data:** Faker (synthetic data generation)
