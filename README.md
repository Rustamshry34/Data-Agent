# Data-Agent

An intelligent AI agent that automatically analyzes your dataset, performs EDA, cleans data, runs AutoML, and lets you **download the trained model** — all with a simple file upload.

Built with **Gradio**, **smolagents**, and **Hugging Face / Together AI**, this agent acts like a full-stack data scientist, automating the entire ML pipeline from raw data to deployable model.

## 🔍 Features

✅ **Automatic Data Loading & Cleaning**  
✅ **Smart Target Detection** (no need to label "target" manually)  
✅ **Exploratory Data Analysis (EDA)** with interactive plots:  
- Correlation heatmap  
- Missingness map  
- Distribution & box plots  

✅ **AutoML with Model Selection**  
- Tries multiple models (Random Forest, Logistic/Linear Regression)  
- Picks the best one based on performance

✅ **Visual Insights in Gradio**  
- All plots rendered in-gallery  
- Clean Markdown-style report  

✅ **Download Trained Model**  
- Export model as `.pkl` file  
- Ready for inference or deployment  

✅ **Token-Efficient & Fast**  
- Optimized for cost and speed  
- Uses smart heuristics to avoid loops


## 🛠️ Tech Stack

- 🤖 **smolagents** – For AI agent reasoning and tool use
- 💬 **Qwen/Qwen2.5-Coder-32B-Instruct** (via Together AI) – Powerful code-generation LLM
- 📊 **Pandas, Matplotlib, Seaborn** – Data analysis & visualization
- 🤖 **Scikit-learn** – Machine learning models
- 🎛️ **Gradio** – Beautiful web interface
- ☁️ **Together AI** – Inference backend (alternative: Hugging Face)

## 📦 Downloadable Model

The trained model is saved as a `.pkl` file containing:
```python
{
  "model": trained_sklearn_model,
  "task": "classification" or "regression",
  "target_column": "target",
  "features": [""]
}

