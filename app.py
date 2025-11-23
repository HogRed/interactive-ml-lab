# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score
)

# load data
@st.cache_data # decorator that will cache the loaded data
def load_data():
    data_bunch = fetch_california_housing(as_frame=True) # load in DataFrame format
    df = data_bunch.frame.copy() # make a copy of the DataFrame
    df.rename(columns={"MedHouseVal": "MedHouseValue"}, inplace=True) # rename target for clarity
    return df, data_bunch

# Page layout
st.set_page_config(
    page_title="Interactive ML Lab – California Housing",
    layout="wide"
)

# Title and description
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

# Set target column and feature columns
target_col = "MedHouseValue"
feature_cols = [c for c in df.columns if c != target_col] # list comprehension

# Model selection box
model_type = st.sidebar.selectbox(
    "Choose model",
    ["Linear Regression", "Random Forest Regressor"]
)

# Test size slider
test_size = st.sidebar.slider(
    "Test size (fraction of data)",
    min_value=0.1,
    max_value=0.5,
    value=0.2,
    step=0.05
)

# Random seed input
random_state = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    max_value=10_000,
    value=42,
    step=1
)

st.sidebar.markdown("---") # separator line

# Train button
train_button = st.sidebar.button("Train / Retrain Model")

# Data overview
st.subheader("1. Dataset Overview")

# Data dictionary nested in expander
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

# Show data preview (first few rows)
st.write("Preview of the dataset:")
st.dataframe(df.head())

# Quick EDA
st.subheader("2. Quick EDA")

# create two columns for layout
col1, col2 = st.columns(2)

# in first column: summary stats and histogram
with col1:
    numeric_col = st.selectbox("Choose a column to explore", feature_cols + [target_col])

    # summary stats
    st.write(f"Summary statistics for `{numeric_col}`:")
    st.write(df[numeric_col].describe())

    # histogram
    fig, ax = plt.subplots()
    ax.hist(df[numeric_col], bins=30)
    ax.set_title(f"Histogram of {numeric_col}")
    ax.set_xlabel(numeric_col)
    ax.set_ylabel("Count")
    st.pyplot(fig)

with col2:
    st.write("Relationship with target (scatter plot)")

    # select feature for x-axis (default to MedInc if available)
    x_for_scatter = st.selectbox(
        "Choose feature for x-axis",
        feature_cols,
        index=feature_cols.index("MedInc") if "MedInc" in feature_cols else 0
    )

    # scatter plot
    fig2, ax2 = plt.subplots()
    ax2.scatter(df[x_for_scatter], df[target_col], alpha=0.3)
    ax2.set_xlabel(x_for_scatter)
    ax2.set_ylabel(target_col)
    ax2.set_title(f"{x_for_scatter} vs {target_col}")
    st.pyplot(fig2)

st.markdown(
    """
**Think** about the patterns you're seeing:
- Does the relationship look linear?
- Are there obvious outliers?
- How might this affect model choice?
"""
)

# Modeling
st.subheader("3. Train a Model")

X = df[feature_cols].values # feature matrix
y = df[target_col].values # target vector

# create training and test splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

# Train model on button click
if train_button:
    # Choose model
    if model_type == "Linear Regression":
        model = LinearRegression() # create instance of Linear Regression
    else:
        # create instance of Random Forest Regressor
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1 # use all CPU cores
        )

    # fit and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate metrics
    rmse = mean_squared_error(y_test, y_pred)**0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"**Model:** {model_type}")
    st.write(f"- RMSE: `{rmse:.3f}`")
    st.write(f"- MAE: `{mae:.3f}`")
    st.write(f"- R²: `{r2:.3f}`")

    # Feature importances plot if RF selected
    if model_type == "Random Forest Regressor":
        # get feature importances from the model
        importances = model.feature_importances_
        
        # create DataFrame for better visualization
        imp_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": importances
        }).sort_values("importance", ascending=False)

        st.write("Feature importances (Random Forest):")
        st.dataframe(imp_df)

        # plot feature importances
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

# If train button not clicked yet
else:
    st.info("👈 Use the sidebar and click **Train / Retrain Model** to fit a model.")