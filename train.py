import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier
import plotly.express as px
import plotly.graph_objects as go
import joblib
from fuzzywuzzy import process
import logging
import io

# Logging setup
logging.basicConfig(filename='crime_predictor.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Streamlit app started.")

# Streamlit app title and description
st.title("Crime Location Predictor")
st.markdown("""
This application predicts likely crime locations based on crime data using RandomForest, XGBoost, and an optimized Neural Network. Upload an Excel file, explore visualizations, and predict locations for specific crime types.
""")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
if uploaded_file is None:
    st.info("Please upload an Excel file to begin.")
    st.stop()

try:
    df = pd.read_excel(uploaded_file)
    st.success(f"✅ Data Loaded. Shape: {df.shape}")
    logging.info(f"Data loaded successfully: {df.shape}")
except Exception as e:
    st.error(f"❌ Failed to read uploaded file: {e}")
    logging.exception("Excel Read Error")
    st.stop()

# --- Normalize Column Names ---
df.columns = df.columns.str.strip().str.upper()

# Show raw data option
if st.checkbox("Show first few rows of data"):
    st.dataframe(df.head())

# --- Drop Unwanted Columns ---
drop_cols = ['FIR_NO', 'ACTS_SEC', 'ALTERATION_DT', 'CS_DATE',
             'TAKENONFILE_DATE', 'DISPOSAL_DT', 'BRIEF_FACTS']
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

# --- Clean & Deduplicate ---
df = df.fillna('Unknown')
before = df.shape[0]
df = df.drop_duplicates()
st.info(f"Removed {before - df.shape[0]} duplicate rows. New shape: {df.shape}")
logging.info(f"Duplicates removed: {before - df.shape[0]} rows. New shape: {df.shape}")

# --- Robust Column Detection ---
cols_set = set(df.columns)

def find_col(candidates):
    """Find the first matching column from a list of possible names."""
    for c in candidates:
        if c and c.upper() in cols_set:
            return c.upper()
    return None

# Detect important columns
major_head_col = find_col(['MAJOR_HEAD', 'CRIME TYPE', 'CRIME', 'OFFENCE', 'OFFENCE TYPE'])
district_col = find_col(['DISTRICT'])
minor_head_col = find_col(['MINOR_HEAD'])
ps_col = find_col(['PS', 'POLICE STATION', 'POLICE_STATION', 'POLICE_STN'])

if not major_head_col:
    st.error(f"❌ Could not find a crime type column (e.g., 'MAJOR_HEAD', 'CRIME TYPE'). Columns present: {list(df.columns)}")
    logging.error(f"Could not find crime type column. Columns present: {list(df.columns)}")
    st.stop()

st.write(f"**Crime Type Column:** {major_head_col}")
if district_col:
    st.write(f"**District Column:** {district_col}")
if ps_col:
    st.write(f"**Police Station Column:** {ps_col}")

# --- Clean Text Fields ---
for c in [major_head_col, ps_col, district_col, minor_head_col]:
    if c and c in df.columns:
        df[c] = df[c].astype(str).str.strip().str.title()

# --- Clean Year and Month ---
if 'FIR_YEAR' in df.columns:
    df['FIR_YEAR'] = pd.to_numeric(df['FIR_YEAR'], errors='coerce')
    df = df[df['FIR_YEAR'].between(2000, 2030) | df['FIR_YEAR'].isna()]
    logging.info("FIR_YEAR validated.")
if 'FIR_MONTH' in df.columns:
    df['FIR_MONTH'] = pd.to_numeric(df['FIR_MONTH'], errors='coerce').clip(1, 12)
    logging.info("FIR_MONTH validated.")

# --- Target Selection for ML ---
target_col = None
min_samples_for_district = 5
min_samples_for_ps = 3

if district_col:
    vc = df[district_col].value_counts()
    if vc.size >= 2 and vc.min() >= min_samples_for_district:
        target_col = district_col
        st.write(f"📊 Using {district_col} as ML Target ({vc.size} classes, min {vc.min()} samples).")
    else:
        st.write(f"{district_col} has low variance ({vc.size} classes, min {vc.min()} samples).")

if target_col is None and ps_col:
    vcps = df[ps_col].value_counts()
    if vcps.size >= 2 and vcps.min() >= min_samples_for_ps:
        target_col = ps_col
        st.write(f"📊 Using {ps_col} as ML Target ({vcps.size} classes, min {vcps.min()} samples).")
    else:
        st.write(f"{ps_col} has low variance ({vcps.size if ps_col in df.columns else 0} classes, min {vcps.min() if ps_col in df.columns else 0} samples).")

# --- ML Training ---
model = None
xgb_model = None
nn_model = None
le_dict = {}

if target_col:
    st.subheader("Training Machine Learning Models")
    keep = [target_col, major_head_col, minor_head_col, 'FIR_MONTH', 'FIR_YEAR',
            'CASE_STAGE', 'TYPE_OF_DISP', 'GRAVE', 'OCCURENCE_PLACE']
    keep = [c for c in keep if c in df.columns]
    df_ml = df[keep].copy().dropna(subset=[target_col, major_head_col])

    # Filter classes with minimum samples
    min_samples = min_samples_for_district if target_col == district_col else min_samples_for_ps
    class_counts = df_ml[target_col].value_counts()
    good_classes = class_counts[class_counts >= min_samples].index
    df_ml = df_ml[df_ml[target_col].isin(good_classes)]
    st.info(f"ML dataset shape after filtering: {df_ml.shape} (classes: {len(good_classes)})")

    if df_ml[target_col].nunique() < 2:
        st.warning("⚠️ Not enough distinct classes to train ML model.")
        logging.warning("Not enough classes for ML training.")
    else:
        # --- Initial NaN Check ---
        nan_counts = df_ml.isna().sum()
        if nan_counts.any():
            st.warning(f"NaN counts in df_ml before processing:\n{nan_counts[nan_counts > 0]}")
            logging.info(f"NaN counts in df_ml before processing:\n{nan_counts[nan_counts > 0]}")

        # --- Impute NaN for All Columns ---
        for col in df_ml.columns:
            if df_ml[col].isna().any():
                if df_ml[col].dtype in ['object', 'category']:
                    mode_value = df_ml[col].mode().iloc[0] if not df_ml[col].mode().empty else 'Unknown'
                    df_ml[col] = df_ml[col].fillna(mode_value)
                    logging.info(f"Imputed NaN in categorical column {col} with mode: {mode_value}")
                else:
                    median_value = df_ml[col].median()
                    df_ml[col] = df_ml[col].fillna(median_value)
                    logging.info(f"Imputed NaN in numerical column {col} with median: {median_value}")

        # --- Feature Engineering ---
        season_map = {1:'Winter', 2:'Winter', 3:'Spring', 4:'Spring', 5:'Spring',
                      6:'Summer', 7:'Summer', 8:'Summer', 9:'Monsoon', 10:'Monsoon',
                      11:'Monsoon', 12:'Winter'}
        if 'FIR_MONTH' in df_ml.columns:
            df_ml['SEASON'] = df_ml['FIR_MONTH'].map(season_map).fillna('Unknown')
            st.write("Added SEASON feature.")
            logging.info("Added SEASON feature.")
        if major_head_col in df_ml.columns and 'GRAVE' in df_ml.columns:
            df_ml['CRIME_GRAVE_INTERACT'] = df_ml[major_head_col].astype(str) + '_' + df_ml['GRAVE'].astype(str)
            df_ml['CRIME_GRAVE_INTERACT'] = df_ml['CRIME_GRAVE_INTERACT'].fillna('Unknown_Unknown')
            st.write("Added CRIME_GRAVE_INTERACT feature.")
            logging.info("Added CRIME_GRAVE_INTERACT feature.")

        # --- Check NaN After Feature Engineering ---
        nan_counts = df_ml.isna().sum()
        if nan_counts.any():
            st.error(f"❌ NaN values detected after feature engineering:\n{nan_counts[nan_counts > 0]}")
            logging.error(f"NaN values after feature engineering:\n{nan_counts[nan_counts > 0]}")
            st.stop()

        # --- Label Encoding ---
        cat_cols = df_ml.select_dtypes(include=['object', 'category']).columns.tolist()
        if df_ml[target_col].dtype == 'object' or df_ml[target_col].dtype.name == 'category':
            cat_cols = [c for c in cat_cols if c != target_col]

        for col in cat_cols:
            le = LabelEncoder()
            df_ml[col] = le.fit_transform(df_ml[col].astype(str))
            le_dict[col] = le

        target_le = LabelEncoder()
        df_ml[target_col] = target_le.fit_transform(df_ml[target_col].astype(str))
        le_dict[target_col] = target_le

        # --- Check NaN After Label Encoding ---
        nan_counts = df_ml.isna().sum()
        if nan_counts.any():
            st.error(f"❌ NaN values detected after label encoding:\n{nan_counts[nan_counts > 0]}")
            logging.error(f"NaN values after label encoding:\n{nan_counts[nan_counts > 0]}")
            st.stop()

        # --- Train/Test Split ---
        X = df_ml.drop(columns=[target_col])
        y = df_ml[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # --- Check NaN in X_train and X_test ---
        nan_train = X_train.isna().sum()
        nan_test = X_test.isna().sum()
        if nan_train.any() or nan_test.any():
            st.error(f"❌ NaN values detected in X_train:\n{nan_train[nan_train > 0]}\nX_test:\n{nan_test[nan_test > 0]}")
            logging.error(f"NaN in X_train:\n{nan_train[nan_train > 0]}\nX_test:\n{nan_test[nan_test > 0]}")
            st.stop()

        # --- Standardize Features for Neural Network ---
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        joblib.dump(scaler, 'scaler.pkl')

        # --- Validate Scaled Data ---
        if np.any(np.isnan(X_train_scaled)) or np.any(np.isnan(X_test_scaled)):
            nan_cols_train = np.where(np.any(np.isnan(X_train_scaled), axis=0))[0]
            nan_cols_test = np.where(np.any(np.isnan(X_test_scaled), axis=0))[0]
            st.error(f"❌ NaN values detected in scaled data. Columns with NaN in X_train_scaled: {X.columns[nan_cols_train].tolist()}. Columns with NaN in X_test_scaled: {X.columns[nan_cols_test].tolist()}")
            logging.error(f"NaN in X_train_scaled columns: {X.columns[nan_cols_train].tolist()}. NaN in X_test_scaled columns: {X.columns[nan_cols_test].tolist()}")
            st.stop()
        else:
            st.info("✅ No NaN values in scaled data.")
            logging.info("No NaN values in scaled data.")

        # --- RandomForest Training ---
        model = RandomForestClassifier(n_estimators=200, max_depth=14, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, y_pred)
        st.success(f"✅ RandomForest Accuracy on {target_col}: {rf_accuracy:.3f}")
        logging.info(f"RandomForest trained. Accuracy: {rf_accuracy}")

        # --- XGBoost Training ---
        try:
            xgb_model = XGBClassifier(
                n_estimators=200, max_depth=10, random_state=42, n_jobs=-1,
                objective='multi:softprob', num_class=len(np.unique(y)),
                eval_metric='mlogloss', learning_rate=0.1, subsample=0.8, colsample_bytree=0.8
            )
            xgb_model.fit(X_train, y_train)
            xgb_pred = xgb_model.predict(X_test)
            xgb_accuracy = accuracy_score(y_test, xgb_pred)
            st.success(f"✅ XGBoost Accuracy on {target_col}: {xgb_accuracy:.3f}")
            logging.info(f"XGBoost trained. Accuracy: {xgb_accuracy}")
        except Exception as ex:
            st.warning(f"⚠️ XGBoost training failed: {ex}")
            logging.exception("XGBoost error")
            xgb_model = None

        # --- Neural Network Hyperparameter Optimization ---
        def create_model(neurons=128, layers=2, dropout_rate=0.3, learning_rate=0.001):
            model = Sequential()
            model.add(Dense(neurons, activation='relu', input_shape=(X_train.shape[1],)))
            model.add(Dropout(dropout_rate))
            for _ in range(layers - 1):
                model.add(Dense(neurons // 2, activation='relu'))
                model.add(Dropout(dropout_rate))
            model.add(Dense(len(np.unique(y)), activation='softmax'))
            model.compile(optimizer=Adam(learning_rate=learning_rate),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            return model

        try:
            # Wrap Keras model for GridSearchCV
            nn_model = KerasClassifier(model=create_model, verbose=0)
            param_grid = {
                'model__neurons': [64, 128],
                'model__layers': [2, 3],
                'model__dropout_rate': [0.2, 0.3],
                'model__learning_rate': [0.001, 0.0001],
                'batch_size': [32, 64],
                'epochs': [50, 100]
            }
            grid = GridSearchCV(estimator=nn_model, param_grid=param_grid, cv=3, n_jobs=1)
            y_train_cat = to_categorical(y_train, num_classes=len(np.unique(y)))
            grid_result = grid.fit(X_train_scaled, y_train_cat)
            nn_model = grid_result.best_estimator_
            st.success(f"✅ Neural Network Best Params: {grid_result.best_params_}")
            st.success(f"✅ Neural Network Best CV Score: {grid_result.best_score_:.3f}")
            logging.info(f"Neural Network best params: {grid_result.best_params_}")
            logging.info(f"Neural Network best CV score: {grid_result.best_score_}")

            # Evaluate on test set
            y_test_cat = to_categorical(y_test, num_classes=len(np.unique(y)))
            nn_pred = nn_model.predict(X_test_scaled)
            nn_accuracy = accuracy_score(y_test, nn_pred)
            st.success(f"✅ Neural Network Accuracy on {target_col}: {nn_accuracy:.3f}")
            logging.info(f"Neural Network trained. Accuracy: {nn_accuracy}")
        except Exception as ex:
            st.warning(f"⚠️ Neural Network training failed: {ex}")
            logging.exception("Neural Network error")
            nn_model = None

        # --- Cross-Validation for RandomForest ---
        cv = min(5, max(2, len(y) // 20))
        try:
            scores = cross_val_score(model, X, y, cv=cv)
            st.write(f"Cross-Validation Scores (RF): mean={scores.mean():.2f}, std={scores.std():.2f}")
            logging.info(f"RandomForest Cross-validation scores: mean={scores.mean():.2f}, std={scores.std():.2f}")
        except Exception as ex:
            st.warning(f"⚠️ RandomForest Cross-validation failed: {ex}")
            logging.warning(f"RandomForest Cross-validation failed: {ex}")

        # --- Feature Importance Plot (RandomForest) ---
        feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        fig_feat = px.bar(x=feat_imp[:10].values, y=feat_imp[:10].index, orientation='h',
                          title=f"Top Features Influencing {target_col} (RF)",
                          labels={'x': 'Importance', 'y': 'Feature'})
        st.plotly_chart(fig_feat)

        # --- Feature Importance Plot (Neural Network) ---
        if nn_model:
            try:
                perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
                perm_imp_df = pd.Series(perm_importance.importances_mean, index=X.columns).sort_values(ascending=False)
                fig_perm = px.bar(x=perm_imp_df[:10].values, y=perm_imp_df[:10].index, orientation='h',
                                  title=f"Top Features Influencing {target_col} (NN - Permutation Importance)",
                                  labels={'x': 'Importance', 'y': 'Feature'})
                st.plotly_chart(fig_perm)
            except Exception as ex:
                st.warning(f"⚠️ Neural Network permutation importance failed: {ex}")
                logging.warning(f"Neural Network permutation importance failed: {ex}")

        # --- Confusion Matrix ---
        unique_labels = model.classes_
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
        test_label_counts = pd.Series(y_test).value_counts().loc[unique_labels].fillna(0)
        top_labels_positions = np.argsort(test_label_counts.values)[::-1][:10]
        top_encoded_labels = unique_labels[top_labels_positions]
        top_display = le_dict[target_col].inverse_transform(top_encoded_labels)
        cm_small = cm[np.ix_(top_labels_positions, top_labels_positions)]
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm_small, x=top_display, y=top_display, colorscale='Blues',
            text=cm_small, texttemplate="%{text}", textfont={"size": 10}))
        fig_cm.update_layout(title=f"Confusion Matrix for {target_col} (Top classes)",
                             xaxis_title=target_col, yaxis_title=target_col)
        st.plotly_chart(fig_cm)

        # --- Save Models ---
        joblib.dump(model, f'{target_col.lower()}_model_rf.pkl')
        if xgb_model:
            joblib.dump(xgb_model, f'{target_col.lower()}_model_xgb.pkl')
        if nn_model:
            nn_model.model_.save(f'{target_col.lower()}_model_nn.h5')
        joblib.dump(le_dict, 'le_dict.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        st.info("💾 Models, encoders, and scaler saved successfully.")
        logging.info("Models, encoders, and scaler saved.")

# --- Treemap Visualization ---
location_col = district_col if district_col else ps_col
if location_col and major_head_col:
    crime_pivot = df.groupby([location_col, major_head_col]).size().reset_index(name='Count')
    if not crime_pivot.empty:
        fig_tree = px.treemap(crime_pivot, path=[location_col, major_head_col], values='Count',
                              title=f'Crime Hierarchy by {location_col} and Type')
        st.plotly_chart(fig_tree)
        logging.info("Treemap visualization shown.")

# --- Prediction Section ---
st.subheader("🔮 Predict Crime Location")
user_input_raw = st.text_input("Enter the crime type:", "")
user_input = user_input_raw.strip().lower()

if user_input:
    # Prepare crime list for fuzzy matching
    crime_choices = [str(c).lower() for c in le_dict[major_head_col].classes_] if model and major_head_col in le_dict else df[major_head_col].astype(str).str.lower().unique().tolist()
    if not crime_choices:
        st.error("❌ No crime types available in the dataset.")
        logging.error("No crime types available in the dataset.")
        st.stop()

    best_match, score = process.extractOne(user_input, crime_choices) if crime_choices else (None, 0)
    chosen_crime = best_match if best_match and score >= 70 else user_input
    if best_match and score >= 70:
        st.write(f"Interpreted as: **{chosen_crime.title()}** (confidence {score})")
        logging.info(f"Fuzzy match: {chosen_crime} (score: {score})")
    else:
        st.warning("⚠️ No close fuzzy match found — proceeding with exact input.")
        logging.warning("No fuzzy match found.")

    # --- ML Prediction ---
    if not model or not le_dict or major_head_col not in le_dict:
        st.error("❌ No trained model or encoders available. Ensure a valid ML target was found.")
        logging.error("ML model or encoders not available.")
    else:
        crime_classes_lower = [c.lower() for c in le_dict[major_head_col].classes_]
        if chosen_crime not in crime_classes_lower:
            available_crimes = le_dict[major_head_col].classes_[:10]
            st.error(f"❌ Crime '{chosen_crime.title()}' not found in trained classes. Available sample: {', '.join(available_crimes)}")
            logging.error(f"Crime '{chosen_crime}' not found in trained classes.")
            st.stop()

        # Build prediction sample
        sample_dict = {}
        missing_mode_cols = []
        for col in X.columns:
            if col == major_head_col:
                sample_dict[col] = None
                continue
            if col in df_ml.columns:
                mode_values = df_ml[col].mode()
                if not mode_values.empty:
                    sample_dict[col] = mode_values.iloc[0]
                else:
                    missing_mode_cols.append(col)
                    logging.warning(f"Empty mode for column: {col}")
                    sample_dict[col] = df_ml[col].dropna().iloc[0] if not df_ml[col].dropna().empty else 0
            else:
                sample_dict[col] = 0
                logging.warning(f"Column {col} not in dataset; using default value 0.")

        if missing_mode_cols:
            st.warning(f"⚠️ Columns with no mode detected: {', '.join(missing_mode_cols)}")

        # Set encoded crime
        encoded_val = le_dict[major_head_col].transform([le_dict[major_head_col].classes_[crime_classes_lower.index(chosen_crime)]])[0]
        sample_dict[major_head_col] = encoded_val
        sample = pd.DataFrame([sample_dict], columns=X.columns)

        # Ensure categorical columns are encoded
        for col in sample.columns:
            if col in le_dict and sample[col].dtype == object:
                try:
                    sample[col] = le_dict[col].transform([sample[col].astype(str)])[0]
                except Exception as e:
                    st.warning(f"⚠️ Failed to encode column {col}: {e}. Using default value 0.")
                    logging.warning(f"Failed to encode column {col}: {e}")
                    sample[col] = 0

        # Debug output
        st.write("🧩 Prepared Sample for Prediction:")
        st.dataframe(sample)
        logging.info(f"Sample DataFrame for prediction: {sample.to_dict()}")

        # Predict with RandomForest
        try:
            probs = model.predict_proba(sample)[0]
            top_idx = np.argsort(probs)[::-1][:5]
            top_locs = le_dict[target_col].inverse_transform(model.classes_[top_idx])
            st.subheader(f"ML Prediction (RandomForest) for {target_col}")
            for i, loc in enumerate(top_locs, start=1):
                st.write(f"{i}. {loc} (Probability: {probs[top_idx[i-1]]:.3f})")
            logging.info(f"RandomForest prediction successful: {top_locs.tolist()}")
        except Exception as e:
            error_msg = f"❌ RandomForest Prediction Error: {str(e)}"
            st.error(error_msg)
            logging.error(error_msg)

        # Predict with XGBoost
        if xgb_model:
            try:
                xgb_probs = xgb_model.predict_proba(sample)[0]
                xgb_top_idx = np.argsort(xgb_probs)[::-1][:5]
                xgb_top_locs = le_dict[target_col].inverse_transform(xgb_model.classes_[xgb_top_idx])
                st.subheader(f"ML Prediction (XGBoost) for {target_col}")
                for i, loc in enumerate(xgb_top_locs, start=1):
                    st.write(f"{i}. {loc} (Probability: {xgb_probs[xgb_top_idx[i-1]]:.3f})")
                logging.info(f"XGBoost prediction successful: {xgb_top_locs.tolist()}")
            except Exception as e:
                error_msg = f"❌ XGBoost Prediction Error: {str(e)}"
                st.error(error_msg)
                logging.error(error_msg)

        # Predict with Neural Network
        if nn_model:
            try:
                sample_scaled = scaler.transform(sample)
                if np.any(np.isnan(sample_scaled)):
                    st.error("❌ NaN values detected in scaled prediction sample.")
                    logging.error("NaN values in scaled prediction sample.")
                    st.stop()
                nn_probs = nn_model.predict_proba(sample_scaled)[0]
                nn_top_idx = np.argsort(nn_probs)[::-1][:5]
                nn_top_locs = le_dict[target_col].inverse_transform(np.arange(len(np.unique(y)))[nn_top_idx])
                st.subheader(f"ML Prediction (Neural Network) for {target_col}")
                for i, loc in enumerate(nn_top_locs, start=1):
                    st.write(f"{i}. {loc} (Probability: {nn_probs[nn_top_idx[i-1]]:.3f})")
                logging.info(f"Neural Network prediction successful: {nn_top_locs.tolist()}")
            except Exception as e:
                error_msg = f"❌ Neural Network Prediction Error: {str(e)}"
                st.error(error_msg)
                logging.error(error_msg)

        # --- Frequency-Based Prediction ---
        freq_target_col = ps_col if ps_col else district_col
        if freq_target_col:
            filtered = df[df[major_head_col].astype(str).str.lower().str.contains(chosen_crime, na=False)]
            if filtered.empty:
                st.warning(f"⚠️ No historical records found for '{chosen_crime.title()}' in frequency analysis.")
                logging.warning(f"No records for '{chosen_crime}' in frequency analysis.")
            else:
                counts = filtered[freq_target_col].value_counts(normalize=True).head(10)
                freq_df = counts.reset_index()
                freq_df.columns = [freq_target_col, 'Proportion']
                st.subheader(f"📊 Frequency-based Top {freq_target_col}")
                st.dataframe(freq_df)
                logging.info(f"Frequency analysis completed for {freq_target_col}")

                out = io.StringIO()
                freq_df.to_csv(out, index=False)
                st.download_button("Download Frequency Predictions (CSV)", data=out.getvalue(),
                                   file_name="freq_predictions.csv", mime="text/csv")

                fig_freq = px.bar(freq_df, x='Proportion', y=freq_target_col, orientation='h',
                                  title=f"Top {freq_target_col} for {chosen_crime.title()}")
                st.plotly_chart(fig_freq)

st.markdown("---")
st.write("✅ Process completed. Check `crime_predictor.log` for detailed logs.")