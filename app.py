# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------------
# Helper: load data
# -------------------------
@st.cache_data
def load_data():
    data_bunch = fetch_california_housing(as_frame=True)
    df = data_bunch.frame.copy()
    df.rename(columns={"MedHouseVal": "MedHouseValue"}, inplace=True)
    return df, data_bunch

# -------------------------
# Page layout
# -------------------------
st.set_page_config(
    page_title="Interactive ML Lab – California Housing",
    layout="wide"
)

st.title("🏡 Interactive ML Lab: California Housing")
st.markdown(
    """
This mini app walks you through a simple **end-to-end ML workflow**:

1. Explore the California Housing dataset  
2. Do some quick EDA  
3. Train a model and inspect performance  

Use this as a teaching or learning tool — tweak things and see what happens.
"""
)

# Load data
df, data_bunch = load_data()

# Sidebar controls
st.sidebar.header("Controls")

target_col = "MedHouseValue"
feature_cols = [c for c in df.columns if c != target_col]

model_type = st.sidebar.selectbox(
    "Choose model",
    ["Linear Regression", "Random Forest Regressor"]
)

test_size = st.sidebar.slider(
    "Test size (fraction of data)",
    min_value=0.1,
    max_value=0.5,
    value=0.2,
    step=0.05
)

random_state = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    max_value=10_000,
    value=42,
    step=1
)

st.sidebar.markdown("---")
train_button = st.sidebar.button("Train / Retrain Model")

# -------------------------
# 1. Data overview
# -------------------------
st.subheader("1. Dataset Overview")

with st.expander("Show data dictionary"):
    st.write(
        """
        **Columns (features)**  
        - `MedInc`: median income in block group  
        - `HouseAge`: median house age in years  
        - `AveRooms`: average number of rooms per household  
        - `AveBedrms`: average number of bedrooms per household  
        - `Population`: block group population  
        - `AveOccup`: average household size  
        - `Latitude`, `Longitude`: geographic coordinates  
        
        **Target**  
        - `MedHouseValue`: median house value (in 100,000s of dollars)
        """
    )

st.write("Preview of the dataset:")
st.dataframe(df.head())

# -------------------------
# 2. Quick EDA
# -------------------------
st.subheader("2. Quick EDA")

col1, col2 = st.columns(2)

with col1:
    numeric_col = st.selectbox("Choose a column to explore", feature_cols + [target_col])
    st.write(f"Summary statistics for `{numeric_col}`:")
    st.write(df[numeric_col].describe())

    fig, ax = plt.subplots()
    ax.hist(df[numeric_col], bins=30)
    ax.set_title(f"Histogram of {numeric_col}")
    ax.set_xlabel(numeric_col)
    ax.set_ylabel("Count")
    st.pyplot(fig)

with col2:
    st.write("Relationship with target (scatter plot)")
    x_for_scatter = st.selectbox(
        "Choose feature for x-axis",
        feature_cols,
        index=feature_cols.index("MedInc") if "MedInc" in feature_cols else 0
    )

    fig2, ax2 = plt.subplots()
    ax2.scatter(df[x_for_scatter], df[target_col], alpha=0.3)
    ax2.set_xlabel(x_for_scatter)
    ax2.set_ylabel(target_col)
    ax2.set_title(f"{x_for_scatter} vs {target_col}")
    st.pyplot(fig2)

st.markdown(
    """
**Teaching note:** Encourage students to *talk about patterns* they see:
- Does the relationship look linear?
- Are there obvious outliers?
- How might this affect model choice?
"""
)

# -------------------------
# 3. Modeling
# -------------------------
st.subheader("3. Train a Model")

X = df[feature_cols].values
y = df[target_col].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

if train_button:
    # Choose model
    if model_type == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"**Model:** {model_type}")
    st.write(f"- RMSE: `{rmse:.3f}`")
    st.write(f"- MAE: `{mae:.3f}`")
    st.write(f"- R²: `{r2:.3f}`")

    # Feature importances if RF
    if model_type == "Random Forest Regressor":
        importances = model.feature_importances_
        imp_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": importances
        }).sort_values("importance", ascending=False)

        st.write("Feature importances (Random Forest):")
        st.dataframe(imp_df)

        fig3, ax3 = plt.subplots()
        ax3.barh(imp_df["feature"], imp_df["importance"])
        ax3.set_xlabel("Importance")
        ax3.set_title("Feature Importances")
        plt.gca().invert_yaxis()
        st.pyplot(fig3)

    st.markdown(
        """
        **Reflection prompts (for students):**
        - How does the performance of Linear Regression compare to Random Forest?  
        - Which features seem most important, and does that match your EDA?  
        - Try changing the test size or random seed — do the metrics change much?
        """
    )
else:
    st.info("👈 Use the sidebar and click **Train / Retrain Model** to fit a model.")