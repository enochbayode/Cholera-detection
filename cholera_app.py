import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb # Import xgboost

# Set Streamlit page configuration
st.set_page_config(layout="wide")

st.title("Cholera Outbreak Detection and Analysis")

# Add a section for data loading
st.header("Data Loading")

# Add file uploader to Streamlit
uploaded_file = st.file_uploader("Upload your Cholera Dataset CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the data
    df = pd.read_csv(uploaded_file, encoding='latin1')
    st.write("Data loaded successfully!")
    st.write(df.head())

    # Add a section for data preprocessing
    st.header("Data Preprocessing")

    # Create binary target variable (1 for cholera positive, 0 otherwise)
    df['cholera_positive'] = np.where(df['rdt_cat'] == 'positive', 1, 0)

    # Drop unnecessary columns
    df.drop(['Unnamed: 0', 'EpidNo'], axis=1, inplace=True, errors='ignore') # Use errors='ignore' in case columns are already dropped

    # Create flags for performed tests and drop original columns
    df['rdt_performed'] = df['rdt_bin'].notnull().astype(int)
    df['culture_performed'] = df['cult_bin'].notnull().astype(int)
    df.drop(['rdt_cat', 'rdt_bin', 'cult_cat', 'cult_bin'], axis=1, inplace=True, errors='ignore')

    # Check redundancy before deciding on flood columns
    if 'flood' in df.columns and 'flood_cat' in df.columns and set(df['flood'].unique()) == set(df['flood_cat'].unique()):
        df.drop('flood', axis=1, inplace=True)  # keep flood_cat

    # Create binary features
    if 'flood_cat' in df.columns:
        df['flood_occurred'] = np.where(df['flood_cat'].astype(str).str.contains('no flooding', na=False), 0, 1)
    else:
        df['flood_occurred'] = 0 # Default to 0 if flood_cat is missing

    if 'sec' in df.columns:
        df['insurgency_area'] = np.where(df['sec'] == 'insurgency', 1, 0)
    else:
        df['insurgency_area'] = 0 # Default to 0 if sec is missing

    if 'season' in df.columns:
        df['dry_season'] = np.where(df['season'] == 'dry season', 1, 0)
    else:
        df['dry_season'] = 0 # Default to 0 if season is missing

    if 'setting' in df.columns:
        df['urban_area'] = np.where(df['setting'].isin(['urban', 'peri-urban']), 1, 0)
    else:
        df['urban_area'] = 0 # Default to 0 if setting is missing

    # Handle missing age values (impute with median or mean)
    if 'age' in df.columns:
        df['age'].fillna(df['age'].median(), inplace=True)
    else:
        df['age'] = 0 # Default to 0 if age is missing


    st.write("Data preprocessing complete!")
    st.write(df.head())

    # Convert epiweek to numerical value for time series analysis
    if 'epiweek' in df.columns:
        df['week_num'] = df['epiweek'].astype(str).str.extract('W(\d+)').fillna(0).astype(int)
    else:
        df['week_num'] = 0 # Default to 0 if epiweek is missing


    # Identify columns with object dtype that need encoding
    object_cols_to_encode = df.select_dtypes(include='object').columns.tolist()

    # Remove the target column if it was mistakenly included
    if 'cholera_positive' in object_cols_to_encode:
        object_cols_to_encode.remove('cholera_positive')

    # Remove 'outcome' if you are dropping it later
    if 'outcome' in object_cols_to_encode:
        object_cols_to_encode.remove('outcome')


    # Apply one-hot encoding to the identified object columns
    df_encoded = pd.get_dummies(df, columns=object_cols_to_encode, drop_first=True)

    st.write("One-hot encoding complete!")
    st.write(df_encoded.head())

    # Prepare features and target - use the df_encoded DataFrame
    X = df_encoded.drop(['cholera_positive', 'outcome'], axis=1, errors='ignore') # Use errors='ignore' in case outcome is already dropped
    y = df_encoded['cholera_positive']

    st.write("Features (X) and Target (y) prepared.")
    st.write("Shape of X:", X.shape)
    st.write("Shape of y:", y.shape)

    # Add a section for Outbreak Detection
    st.header("Outbreak Detection")

    if 'week_num' in df_encoded.columns:
        # Weekly cases and positive rate
        weekly_analysis = df_encoded.groupby('week_num').agg({
            'cholera_positive': ['count', 'mean']
        }).sort_index()

        weekly_analysis.columns = ['weekly_cases', 'weekly_positive_rate']

        # Calculate z-scores for anomaly detection
        weekly_analysis['cases_zscore'] = (weekly_analysis['weekly_cases'] - weekly_analysis['weekly_cases'].mean()) / weekly_analysis['weekly_cases'].std()
        weekly_analysis['positive_zscore'] = (weekly_analysis['weekly_positive_rate'] - weekly_analysis['weekly_positive_rate'].mean()) / weekly_analysis['weekly_positive_rate'].std()

        # Flag potential outbreaks (z-score > 2)
        weekly_analysis['outbreak_alert'] = np.where(
            (weekly_analysis['cases_zscore'] > 2) | (weekly_analysis['positive_zscore'] > 2),
            'Potential Outbreak',
            'Normal'
        )

        st.write("Weekly Analysis with Outbreak Alerts:")
        st.write(weekly_analysis)

        # Display outbreak alerts
        st.subheader("Potential Outbreak Alerts")
        outbreaks = weekly_analysis[weekly_analysis['outbreak_alert'] == 'Potential Outbreak']
        if not outbreaks.empty:
            st.warning("Potential outbreaks detected in the following weeks:")
            for index, row in outbreaks.iterrows():
                st.write(f"- Week {index}: Cases Z-score: {row['cases_zscore']:.2f}, Positive Rate Z-score: {row['positive_zscore']:.2f}")
        else:
            st.info("No potential outbreaks detected based on current criteria.")

    else:
        st.warning("Weekly analysis cannot be performed as 'week_num' column is missing.")

    # Add a section for Model Training and Evaluation
    st.header("Model Training and Evaluation")

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    st.write("Data split into training and testing sets.")
    st.write("Training features shape:", X_train.shape)
    st.write("Testing features shape:", X_test.shape)
    st.write("Training target shape:", y_train.shape)
    st.write("Testing target shape:", y_test.shape)


    # Initialize and train Random Forest classifier
    st.subheader("Random Forest Model")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Make predictions and evaluate
    y_pred_rf = rf_model.predict(X_test)
    st.write("Random Forest Model Evaluation:")
    st.text("Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred_rf)))
    st.text("Classification Report:\n" + classification_report(y_test, y_pred_rf))
    st.text("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred_rf)))


    # Initialize and train XGBoost classifier
    st.subheader("XGBoost Model")
    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)

    # Make predictions and evaluate
    y_pred_xgb = xgb_model.predict(X_test)
    st.write("XGBoost Model Evaluation:")
    st.text("Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred_xgb)))
    st.text("Classification Report:\n" + classification_report(y_test, y_pred_xgb))
    st.text("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred_xgb)))

    # Feature importance
    st.subheader("Feature Importance (Random Forest)")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(20), ax=ax)
    ax.set_title('Top 20 Important Features for Cholera Detection (Random Forest)')
    st.pyplot(fig)

else:
    st.info("Please upload a CSV file to proceed.")