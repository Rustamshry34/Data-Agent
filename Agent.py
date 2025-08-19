import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image
import gradio as gr
from smolagents import tool, CodeAgent, InferenceClientModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import joblib
import tempfile
import os


# 🔑 Set your HF API key
os.environ["HF_TOKEN"] = "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"


# ————————————————————————————————
# 🔍 Heuristic Target Column Detection
# ————————————————————————————————

def detect_target_column(df: pd.DataFrame) -> str:
    """
    Heuristically detect the most likely target column based on naming, cardinality, and type.
    """
    if df.empty or len(df.columns) < 2:
        return None

    scores = {}

    for col in df.columns:
        score = 0.0
        name_lower = col.lower()

        # Rule 1: Name matches common target keywords
        keywords = ["target", "label", "class", "outcome", "result", "y", "output", "flag", "status", "churn", "survived", "price", "sale"]
        if any(kw in name_lower for kw in keywords):
            score += 3.0
        if name_lower in ["target", "label", "class", "y"]:
            score += 2.0

        # Rule 2: Binary or low-cardinality categorical → likely classification
        nunique = df[col].nunique()
        total = len(df)
        unique_ratio = nunique / total

        if nunique == 2 and df[col].dtype in ["int64", "object", "category"]:
            score += 4.0  # Strong signal
        elif nunique <= 20 and df[col].dtype in ["int64", "object", "category"]:
            score += 3.0

        # Rule 3: High unique ratio + numeric → likely regression target
        if unique_ratio > 0.8 and df[col].dtype in ["int64", "float64"]:
            score += 2.5

        # Rule 4: Avoid ID-like or high-cardinality text
        id_keywords = ["id", "name", "email", "phone", "address", "username", "url", "link"]
        if any(kw in name_lower for kw in id_keywords):
            score -= 10.0
        if nunique == total and df[col].dtype == "object":
            score -= 10.0  # Likely unique identifier

        scores[col] = score

    # Return best candidate if score > 0
    best_col = max(scores, key=scores.get)
    return best_col if scores[best_col] > 0 else None


# ————————————————————————————————
# 🛠️ Tool 1: LoadData
# ————————————————————————————————

@tool
def LoadData(filepath: str) -> dict:
    """
    Loads data from a CSV file and returns it as a dictionary.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        dict: Data as dictionary (from DataFrame.to_dict()).
    """
    df = pd.read_csv(filepath)
    return df.to_dict()


# ————————————————————————————————
# 🛠️ Tool 2: CleanData (Enhanced)
# ————————————————————————————————

@tool
def CleanData(data: dict, handle_outliers: bool = True, impute_strategy: str = "median_mode") -> pd.DataFrame:
    """
    Cleans dataset with smart imputation, encoding, and optional outlier removal.

    Args:
        data (dict): Dataset in dictionary format.
        handle_outliers (bool): Whether to remove outliers using IQR.
        impute_strategy (str): "median_mode" or "mean_mode"

    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    df = pd.DataFrame.from_dict(data)

    # Drop duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Handle missing values
    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            if impute_strategy == "median_mode" or df[col].skew() > 1:
                fill_val = df[col].median()
            else:
                fill_val = df[col].mean()
            df[col] = df[col].fillna(fill_val)
        else:
            mode = df[col].mode()
            fill_val = mode[0] if len(mode) > 0 else "Unknown"
            df[col] = df[col].fillna(fill_val)

    # Parse datetime
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
            except:
                pass

    # Encode categorical variables (only if not too many unique values)
    for col in df.select_dtypes(include="object").columns:
        if df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype("category").cat.codes
        # else: leave as object (e.g., free text)

    # Outlier removal (optional)
    if handle_outliers:
        for col in df.select_dtypes(include=["float64", "int64"]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            count_before = len(df)
            df = df[(df[col] >= lower) & (df[col] <= upper)]
            if len(df) == 0:
                # Avoid empty df
                df = pd.DataFrame.from_dict(data)  # Revert
                break

    return df.reset_index(drop=True)


# ————————————————————————————————
# 📊 Tool 3: EDA (Enhanced)
# ————————————————————————————————

@tool
def EDA(data: dict, max_cat_plots: int = 3, max_num_plots: int = 3) -> dict:
    """
    Performs advanced EDA with smart visualizations and insights.

    Args:
        data (dict): Dataset in dictionary format.
        max_cat_plots (int): Max number of categorical distribution plots.
        max_num_plots (int): Max number of numeric vs target plots.

    Returns:
        dict: EDA results including text, plots, and recommendations.
    """
    df = pd.DataFrame.from_dict(data)
    results = {}

    # 1. Summary Stats
    results["summary"] = df.describe(include="all").to_string()

    # 2. Missing Values
    missing = df.isnull().sum()
    results["missing_values"] = missing[missing > 0].to_dict()

    # Missingness heatmap
    if missing.sum() > 0:
        plt.figure(figsize=(8, 4))
        sns.heatmap(df.isnull(), cbar=True, cmap="viridis", yticklabels=False)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
        img = Image.open(buf)
        results["missingness_plot"] = img #buf

    # 3. Correlation Heatmap
    corr = df.corr(numeric_only=True)
    if not corr.empty and len(corr.columns) > 1:
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
        img = Image.open(buf)
        results["correlation_plot"] = img #buf

        # Top 5 absolute correlations
        unstacked = corr.abs().unstack()
        unstacked = unstacked[unstacked < 1.0]
        top_corr = unstacked.sort_values(ascending=False).head(5).to_dict()
        results["top_correlations"] = top_corr

    # 4. Skewness & Kurtosis
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    skew_kurt = {}
    for col in numeric_cols:
        skew_kurt[col] = {"skew": df[col].skew(), "kurtosis": df[col].kurtosis()}
    results["skew_kurtosis"] = skew_kurt

    # 5. Numeric Distributions
    if len(numeric_cols) > 0:
        df[numeric_cols].hist(bins=20, figsize=(12, 8), layout=(2, -1))
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
        img = Image.open(buf)
        results["numeric_distributions"] = img #buf

    # 6. Categorical Distributions
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols[:max_cat_plots]:
        plt.figure(figsize=(6, 4))
        top_vals = df[col].value_counts().head(10)
        sns.barplot(x=top_vals.index, y=top_vals.values)
        plt.xticks(rotation=45)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
        img = Image.open(buf)
        results[f"dist_{col}"] = img #buf

    # 7. Target Relationships
    target_col = detect_target_column(df)
    if target_col:
        results["detected_target"] = target_col
        for col in numeric_cols[:max_num_plots]:
            plt.figure(figsize=(6, 4))
            if df[target_col].nunique() <= 20:
                sns.boxplot(data=df, x=target_col, y=col)
            else:
                sns.scatterplot(data=df, x=col, y=target_col)
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            plt.close()
            buf.seek(0)
            img = Image.open(buf)
            results[f"{col}_vs_{target_col}"] = img #buf

    # 8. Recommendations
    recs = []
    for col, sk in skew_kurt.items():
        if abs(sk["skew"]) > 1:
            recs.append(f"Feature '{col}' is skewed ({sk['skew']:.2f}) → consider log transform.")
    if results["missing_values"]:
        recs.append("Missing data detected → consider KNN or iterative imputation.")
    if results.get("top_correlations"):
        recs.append("High correlations found → consider PCA or feature selection.")
    if target_col:
        recs.append(f"Target variable '{target_col}' detected automatically.")
    results["recommendations"] = recs

    return results


# ————————————————————————————————
# 🤖 Tool 4: AutoML (Enhanced)
# ————————————————————————————————

@tool
def AutoML(data: dict, task_hint: str = None) -> dict:
    """
    Enhanced AutoML with multiple models and robust evaluation.

    Args:
        data (dict): Cleaned dataset.
        task_hint (str): "classification", "regression", or None.

    Returns:
        dict: Model results and metrics.
    """
    df = pd.DataFrame.from_dict(data)
    results = {}

    target_col = detect_target_column(df)
    if not target_col:
        results["note"] = "No target column detected. Check column names and data."
        return results

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # One-hot encode X
    X = pd.get_dummies(X, drop_first=True)

    if X.shape[1] == 0:
        results["error"] = "No valid features after encoding."
        return results

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Detect task
    if task_hint:
        task = task_hint
    elif y.dtype in ["object", "category"] or y.nunique() <= 20:
        task = "classification"
    else:
        task = "regression"

    try:
        if task == "classification":
            models = {
                "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
                "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
            }
            results["task"] = "classification"
            best_acc = 0
            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                if acc > best_acc:
                    best_acc = acc
                    results["accuracy"] = acc
                    results["best_model"] = name
                    results["report"] = classification_report(y_test, preds, zero_division=0)
                    if hasattr(model, "feature_importances_"):
                        results["feature_importance"] = dict(zip(X.columns, model.feature_importances_))

        else:
            models = {
                "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
                "LinearRegression": LinearRegression()
            }
            results["task"] = "regression"
            best_r2 = -float("inf")
            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                r2 = r2_score(y_test, preds)
                if r2 > best_r2:
                    best_r2 = r2
                    results["r2_score"] = r2
                    results["mse"] = mean_squared_error(y_test, preds)
                    results["best_model"] = name
                    best_model = model  # Keep best model
                    if hasattr(model, "feature_importances_"):
                        results["feature_importance"] = dict(zip(X.columns, model.feature_importances_))
        # ✅ Save the best model to a temporary file
        model_dir = tempfile.mkdtemp()
        model_path = os.path.join(model_dir, f"trained_model_{task}.pkl")
        joblib.dump({
            "model": best_model,
            "task": task,
            "target_column": target_col,
            "features": X.columns.tolist()
        }, model_path)

        results["model_download_path"] = model_path
        results["model_info"] = f"Best model: {results['best_model']} | Task: {task} | Target: {target_col}"

    except Exception as e:
        results["error"] = f"Model training failed: {str(e)}"

    return results


# ————————————————————————————————
# 🧠 Initialize the AI Agent
# ————————————————————————————————

agent = CodeAgent(
    tools=[LoadData, CleanData, EDA, AutoML],
    model=InferenceClientModel(
        model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        token=os.environ["HF_TOKEN"], 
        provider="together",
        max_tokens=4048
    ),
    additional_authorized_imports=[
        "pandas", "matplotlib.pyplot", "seaborn", "PIL", "sklearn", "io", "os","joblib","tempfile"
    ],
    max_steps=10,
)


# ————————————————————————————————
# 🖼️ Gradio Interface
# ————————————————————————————————

def analyze_data(file):
    filepath = file.name
    prompt = f"""
    Load the data from '{filepath}', then clean it using CleanData with outlier handling.
    Run EDA to analyze data quality, distributions, and detect the target variable.
    If a target is found, run AutoML to train the best model.
    Return all insights, metrics, and visualizations.
    """
    try:
        results = agent.run(prompt)
    except Exception as e:
        results = {"error": f"Agent failed: {str(e)}"}

    # === Text Report ===
    text_output = ""

    if "error" in results:
        text_output = f"❌ Error: {results['error']}"
    else:
        summary = results.get("summary", "No summary.")
        missing_vals = results.get("missing_values", {})
        top_corr = results.get("top_correlations", {})
        outliers = results.get("outliers", {})
        recs = results.get("recommendations", [])
        detected_target = results.get("detected_target", "Unknown")

        text_output += f"### 📊 Dataset Overview\n"
        text_output += f"**Detected Target:** `{detected_target}`\n\n"
        text_output += f"### Summary Stats\n{summary}\n\n"
        text_output += f"### Missing Values\n{missing_vals}\n\n"
        text_output += f"### Top Correlations\n{top_corr}\n\n"
        text_output += f"### Outliers\n{outliers}\n\n"
        text_output += f"### Recommendations\n" + "\n".join([f"- {r}" for r in recs]) + "\n\n"

        if "task" in results:
            task = results["task"]
            text_output += f"### 🤖 AutoML Results ({task.title()})\n"
            text_output += f"**Best Model:** {results.get('best_model', 'Unknown')}\n"
            if task == "classification":
                text_output += f"**Accuracy:** {results['accuracy']:.3f}\n\n"
                text_output += f"```\n{results['report']}\n```\n"
            else:
                text_output += f"**R²:** {results['r2_score']:.3f}, **MSE:** {results['mse']:.3f}\n"

            feat_imp = sorted(results.get("feature_importance", {}).items(), key=lambda x: x[1], reverse=True)[:5]
            text_output += f"### Top Features\n" + "\n".join([f"- `{f}`: {imp:.3f}" for f, imp in feat_imp])

    # === Collect Plots ===
    plots = []
    for key, value in results.items():
        if isinstance(value, Image.Image):
            plots.append(value)

    model_file = results.get("model_download_path", None)
    if model_file and os.path.exists(model_file):
        model_download_output = model_file
    else:
        model_download_output = None  # No file to download

    return text_output, plots, model_download_output


# ————————————————————————————————
# 🚀 Launch Gradio App
# ————————————————————————————————

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 AI Data Analyst Agent with AutoML & Smart Target Detection")
    gr.Markdown("Upload a CSV file to get **automatic EDA, cleaning, and machine learning insights**.")
    with gr.Row():
        file_input = gr.File(label="📁 Upload CSV")
    with gr.Row():
        text_output = gr.Textbox(label="📝 Analysis Report", lines=24)
    with gr.Row():
        plots_output = gr.Gallery(label="📊 EDA & Model Plots", scale=2)
    with gr.Row():
        model_download = gr.File(label="💾 Download Trained Model (.pkl)")   

    file_input.upload(analyze_data, inputs=file_input, outputs=[text_output, plots_output, model_download])

# Launch
if __name__ == "__main__":
    demo.launch(share=True)  # Use share=True for public link 
    