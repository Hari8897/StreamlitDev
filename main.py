import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import zscore
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
def loadfile(uploadedFile):
    fileExtension = uploadedFile.name.split('.')[-1]
    try:
        if fileExtension == 'csv':
            try:
                return pd.read_csv(uploadedFile, encoding='utf-8')
            except UnicodeDecodeError:
                return pd.read_csv(uploadedFile, encoding='ISO-8859-1')
        elif fileExtension in ['xlsx', 'xls']:
            return pd.read_excel(uploadedFile)
        else:
            st.error("Unsupported file format.")
            return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# Fill null values
def fill_null_values(df):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    num_cols = df.select_dtypes(include=['number']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in num_cols:
        df[col] = df[col].fillna(df[col].mean())
    return df

# Encode categorical columns
def convert_to_numerical(df):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    return df

# Handle outliers using Z-score
def handle_outliers_zscore(df, threshold=3):
    num_cols = df.select_dtypes(include=['number']).columns
    for col in num_cols:
        z_scores = zscore(df[col])
        outliers = np.abs(z_scores) > threshold
        mean_val = df[col][~outliers].mean()
        df.loc[outliers, col] = mean_val
    return df

# Normalize data
def apply_standard_scaling(df):
    num_cols = df.select_dtypes(include=['number']).columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

# Linear Regression model
def linear_regression_model(df, target_column):
    if target_column not in df.columns:
        st.error(f"Target column '{target_column}' not found.")
        return

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation
    st.subheader("Model Evaluation Metrics")
    st.write(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    
    n = len(y_test)
    p = X_test.shape[1]
    r2_adj = 1 - (1 - r2_score(y_test, y_pred)) * (n - 1) / (n - p - 1)
    st.write(f"Adjusted R² Score: {r2_adj:.4f}")

    st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    st.write(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    st.write(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")

    # Coefficients
    st.subheader("Model Coefficients")
    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_
    })
    st.write(coef_df)

    # Actual vs Predicted Plot
    st.subheader("Actual vs Predicted Plot")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax1)
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    st.pyplot(fig1)

    # Residual Plot
    st.subheader("Residuals Distribution")
    residuals = y_test - y_pred
    fig2, ax2 = plt.subplots()
    sns.histplot(residuals, kde=True, ax=ax2)
    ax2.set_title("Residuals")
    st.pyplot(fig2)

# Streamlit UI
st.title("📊 Machine Learning Linear Regression App")

uploadedFile = st.file_uploader('Upload your dataset', type=['csv', 'xlsx', 'xls'])

if uploadedFile is not None:
    df = loadfile(uploadedFile)

    if isinstance(df, pd.DataFrame):
        st.subheader("Initial Data Preview")
        st.write(df.head())

        # Option to drop columns
        drop_cols = st.multiselect("Select columns to drop (e.g., Loan_ID)", options=df.columns)
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)
            st.success(f"Dropped columns: {', '.join(drop_cols)}")

        # Optional preprocessing steps
        if st.checkbox("Fill Missing Values"):
            df = fill_null_values(df)

        if st.checkbox("Encode Categorical Variables"):
            df = convert_to_numerical(df)

        if st.checkbox("Handle Outliers (Z-score Method)"):
            df = handle_outliers_zscore(df)

        if st.checkbox("Apply Standard Scaling"):
            df = apply_standard_scaling(df)

        st.subheader("Processed Data Preview")
        st.write(df.head())
        st.write("Summary Statistics:")
        st.write(df.describe())

        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        fig_corr, ax_corr = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax_corr)
        st.pyplot(fig_corr)     

        # Model
        target_col = st.selectbox("Select target column for Linear Regression", df.columns)
        if st.button("Run Linear Regression"):
            linear_regression_model(df, target_col)

        # Download processed data
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Processed Data", csv, "processed_data.csv", "text/csv")
    else:
        st.error("Uploaded file is not a valid DataFrame.")

