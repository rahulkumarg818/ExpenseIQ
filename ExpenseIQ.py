import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.ticker as mticker

sns.set_style("whitegrid")
st.set_page_config(page_title="ExpenseIQ", layout="wide")

#-----------------------Example Datasets-------------------------------

@st.cache_data
def load_example_data():
    data = {
        "Date": pd.date_range(end=pd.Timestamp.today(), periods=60).astype(str),
        "Category": np.random.choice(["Food", "Transport", "Shopping", "Entertainment", "Utilities", "Health"], size=60),
        "Amount": np.round(np.random.gamma(2.0, 200.0, 60), 2),
        "Description": ["demo"] * 60
    }
    return pd.DataFrame(data)

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Date' not in df.columns or 'Amount' not in df.columns:
        raise ValueError("CSV must contain at least 'Date' and 'Amount' columns.")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date', 'Amount'])
    df = df[df['Amount'] > 0].reset_index(drop=True)
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Year'] = df['Date'].dt.year
    df['Weekday'] = df['Date'].dt.weekday
    if 'Category' in df.columns:
        df['Category'] = df['Category'].astype(str)
    else:
        df['Category'] = np.nan
    return df

#--------------------------Visualization Functions-------------------------------

def plot_expense_distribution(df):
    expense_distribution = df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="#fafafa")
    colors = sns.color_palette("pastel", len(expense_distribution))
    wedges, texts, autotexts = ax.pie(
        expense_distribution,
        autopct='%1.1f%%',
        startangle=120,
        colors=colors,
        textprops={'fontsize': 10, 'color': 'black'}
    )
    plt.setp(autotexts, size=10, weight="bold")
    ax.set_title('Expense Distribution by Category', fontsize=14, weight='bold', pad=20)
    ax.legend(expense_distribution.index, loc="center left", bbox_to_anchor=(1, 0.5))
    st.pyplot(fig)

def plot_time_series(df):
    ts = df.groupby('Date')['Amount'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#fafafa")
    sns.lineplot(
        data=ts,
        x='Date',
        y='Amount',
        marker='o',
        linewidth=2,
        color=sns.color_palette("deep")[0],
        ax=ax
    )
    ax.fill_between(ts['Date'], ts['Amount'], alpha=0.2, color=sns.color_palette("deep")[0])
    ax.set_title('Total Expenses Over Time', fontsize=14, weight='bold', pad=15)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Amount (Rs)', fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')
    ax.xaxis.set_major_locator(mticker.MaxNLocator(8))
    plt.xticks(rotation=30, ha='right')
    st.pyplot(fig)

def plot_boxplot(df):
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#fafafa")
    sns.boxplot(
        data=df,
        x='Category',
        y='Amount',
        palette='Set2',
        linewidth=1.5,
        ax=ax
    )
    ax.set_title("Category-wise Expense Distribution", fontsize=14, weight='bold', pad=15)
    ax.set_xlabel('Category', fontsize=11)
    ax.set_ylabel('Amount (Rs)', fontsize=11)
    plt.xticks(rotation=30, ha='right')
    st.pyplot(fig)

#--------------------------------Model Functions--------------------------------

def train_model(df, model_type='logreg'):
    df_train = df.dropna(subset=['Category']).copy()
    if df_train.empty:
        st.warning("No labeled rows (Category) to train on.")
        return None, None, None
    X = df_train[['Amount','Month','Day','Year','Weekday']].values
    y = df_train['Category'].astype('category').cat.codes.values
    label_encoder = dict(enumerate(df_train['Category'].astype('category').cat.categories))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000) if model_type=='logreg' else GradientBoostingClassifier(n_estimators=200)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    return model, scaler, (acc, classification_report(y_test, y_pred, zero_division=0), confusion_matrix(y_test, y_pred), label_encoder)

def predict_category(model, scaler, df_row, label_encoder):
    X = np.array([[df_row['Amount'], df_row['Month'], df_row['Day'], df_row['Year'], df_row['Weekday']]])
    X_s = scaler.transform(X)
    pred_code = model.predict(X_s)[0]
    return label_encoder.get(pred_code, str(pred_code))

#-------------------------------------Sidebar----------------------------------------

st.sidebar.header("Upload / Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV (Date, Category, Amount, Description optional)", type=['csv'])
use_example = st.sidebar.checkbox("Use example dataset", value=False)
st.sidebar.markdown("---")
model_choice = st.sidebar.selectbox("Model type for classification", ["Logistic Regression", "Gradient Boosting"], index=0)

#-------------------------------------Load Data---------------------------------------------

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()
elif use_example:
    df = load_example_data()
else:
    st.info("Upload a CSV to begin, or check 'Use example dataset' from sidebar.")
    st.stop()

try:
    df = preprocess_df(df)
except ValueError as e:
    st.error(str(e))
    st.stop()

#----------------------------------------Summary Metrics--------------------------------------

st.title("ExpenseIQ")
st.markdown("Upload your expense CSV, explore visualizations, and train a model to categorize and predict expenses.")

st.markdown("---")

st.subheader("Summary Metrics")
total_spent = df['Amount'].sum()
avg_spent = df['Amount'].mean()
top_cat = df.groupby('Category')['Amount'].sum().idxmax() if 'Category' in df.columns else "N/A"
col_a, col_b, col_c = st.columns(3)
col_a.metric("Total Spent", f"₹{total_spent:,.2f}")
col_b.metric("Average per Transaction", f"₹{avg_spent:,.2f}")
col_c.metric("Top Spending Category", top_cat)

st.markdown("---")

#-----------------------------------Visual Insights--------------------------------------------

st.subheader("Visual Insights")
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("Expense Distribution")
    if 'Category' in df.columns and df['Category'].notna().any():
        plot_expense_distribution(df.fillna('Unknown'))
    else:
        st.info("No category labels available for pie chart.")

with col2:
    st.markdown("Expense Trend Over Time")
    plot_time_series(df)

st.markdown("---")

st.markdown("Category-wise Expense Spread")
if df['Category'].notna().any():
    plot_boxplot(df)
else:
    st.info("Boxplot requires labeled categories.")

st.markdown("---")

#--------------------------------Model Training & Prediction---------------------------------

st.header("Model Training & Prediction")

if "trained_model" not in st.session_state:
    st.session_state.trained_model = None
if "trained_scaler" not in st.session_state:
    st.session_state.trained_scaler = None
if "model_label_map" not in st.session_state:
    st.session_state.model_label_map = None
if "model_metrics" not in st.session_state:
    st.session_state.model_metrics = None

def display_model_results(acc, cls_report, cm):
    st.success(f"Model trained successfully — Accuracy: {acc*100:.2f}%")
    st.text(cls_report)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax_cm)
    ax_cm.set_title("Confusion Matrix (Test Set)", fontsize=13, weight='bold')
    st.pyplot(fig_cm)

if st.button("Train model now"):
    model_type = 'logreg' if model_choice == "Logistic Regression" else 'gb'
    model, scaler, metrics = train_model(df, model_type=model_type)
    if model is not None:
        acc, cls_report, cm, label_map = metrics
        st.session_state.trained_model = model
        st.session_state.trained_scaler = scaler
        st.session_state.model_label_map = {v: k for k, v in label_map.items()}
        st.session_state.model_metrics = metrics
        display_model_results(acc, cls_report, cm)
    else:
        st.warning("Could not train — ensure CSV has a labeled 'Category' column.")

trained_model = st.session_state.trained_model
trained_scaler = st.session_state.trained_scaler
model_label_map = st.session_state.model_label_map

st.markdown("---")

#---------------------------------Prediction Form-------------------------------------------

st.subheader("Predict Category for a New Transaction")
with st.form("predict_form"):
    c_amt = st.number_input("Amount (₹)", min_value=0.01, value=100.0, step=1.0)
    c_date = st.date_input("Date", value=datetime.today())
    submitted = st.form_submit_button("Predict")
    if submitted:
        if trained_model is None:
            st.warning("No trained model found. Please train a model first.")
        else:
            row = {
                'Amount': float(c_amt),
                'Date': pd.to_datetime(c_date),
            }
            row['Month'] = row['Date'].month
            row['Day'] = row['Date'].day
            row['Year'] = row['Date'].year
            row['Weekday'] = row['Date'].weekday()
            pred = predict_category(
                trained_model,
                trained_scaler,
                row,
                {v: k for k, v in model_label_map.items()} if model_label_map else {}
            )
            st.success(f"Predicted Category: **{pred}**")

