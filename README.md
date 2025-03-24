# 📚 Summary Evaluation Metrics using LLMs & NLP

This project evaluates machine-generated summaries using a combination of traditional NLP metrics (ROUGE, BERTScore) and referenceless Large Language Model (LLM)-powered metrics (e.g., Bias, Toxicity, Hallucination, etc.).

---

## 🚀 Features

- ✅ ROUGE and BERTScore computation
- ✅ DeepEval LLM metrics using Claude-3 Sonnet (anthropic, can be extended for use with AzureOpenAi or others)
- ✅ Visual EDA with grouped bar charts
- ✅ Interactive dashboard with Plotly Dash (TO DO)
- ✅ Exportable plots and scores for analysis/reporting (TO DO)

---

## ⚙️ Setup

### 1. Clone the repo

```bash
git clone https://github.com/vrjymm/LLM-Evaluations.git
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas matplotlib seaborn plotly dash bert_score rouge_score anthropic deepeval
```

> 🔐 You'll need access to the Anthropic Claude API via `anthropic` package for LLM-based metrics.

---

## 🧪 Run Evaluation

```bash
cd LLM\ Evaluations 
python summary_evaluations.py
```

This loads the dataset, runs metrics for each model, and saves the results to `summary_eval_scores.csv`.

---

## 📊 Run EDA & Save Plots

```bash
python eda_summary_metrics.py
```

- Creates charts for each metric  
- Saves them to the `analysis/` folder  
- Example:
![Example Plot](analysis/hallucination_scores.png)


## 📦 Future Ideas

- Add summary length & token-based analysis  
- Integrate with Streamlit or Gradio  
- Support for multiple LLM judges  
- Export HTML/PDF full reports  

---

## 🙌 Credits

- Built by **Vrinda**  
- LLM-based metrics powered by [DeepEval](https://github.com/confident-ai/deepeval)  
