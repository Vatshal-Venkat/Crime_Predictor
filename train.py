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
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier
import plotly.express as px
import plotly.graph_objects as go
import joblib
from fuzzywuzzy import process
import logging
import io
import os
import tensorflow as tf
import threading
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.utils import resample
import uuid

# Suppress oneDNN custom operations message
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Professional color palette
PROFESSIONAL_PALETTE = ['#FF3D00', '#FF7043', '#FF8F00', '#FFB300', '#FFCA28', '#FFA000']

# Logging setup
logging.basicConfig(filename='crime_predictor.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Streamlit app started.")

# Streamlit app configuration
st.set_page_config(page_title="Crime Predictor", layout="wide")
st.title("Crime Predictor Dashboard")

# --- Sidebar ---
with st.sidebar:
    st.header("Data Upload & Info")
    uploaded_file = st.file_uploader("Upload your Excel file (e.g., FIR_data_2022.xlsx)", type=["xlsx"], key="file_uploader")
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            st.success(f"✅ Data Loaded. Shape: {df.shape}")
            logging.info(f"Data loaded successfully: {df.shape}")
            if st.checkbox("Show first few rows of data", key="preview_data"):
                st.dataframe(df.head())
        except Exception as e:
            st.error(f"❌ Failed to read uploaded file: {e}")
            logging.exception("Excel Read Error")
            st.stop()
    else:
        st.info("Please upload an Excel file to proceed.")
        st.stop()

    # Move blue and yellow info boxes to sidebar
    df.columns = df.columns.str.strip().str.upper()
    drop_cols = ['ACTS_SEC', 'ALTERATION_DT', 'CS_DATE',
                 'TAKENONFILE_DATE', 'DISPOSAL_DT', 'BRIEF_FACTS']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    df = df.fillna('Unknown')
    before = df.shape[0]
    df = df.drop_duplicates()
    st.info(f"Removed {before - df.shape[0]} duplicate rows. New shape: {df.shape}")
    logging.info(f"Duplicates removed: {before - df.shape[0]} rows. New shape: {df.shape}")

    if 'FIR_YEAR' in df.columns:
        df['FIR_YEAR'] = pd.to_numeric(df['FIR_YEAR'], errors='coerce')
        df = df[df['FIR_YEAR'].between(2000, 2030) | df['FIR_YEAR'].isna()]
        logging.info("FIR_YEAR validated.")
    if 'FIR_MONTH' in df.columns:
        nan_before = df['FIR_MONTH'].isna().sum()
        df['FIR_MONTH'] = pd.to_numeric(df['FIR_MONTH'], errors='coerce').clip(1, 12)
        nan_after = df['FIR_MONTH'].isna().sum()
        if nan_after > 0:
            st.warning(f"Found {nan_after} NaN values in FIR_MONTH after conversion. Imputing with 1 (January).")
            logging.info(f"Found {nan_after} NaN values in FIR_MONTH after conversion. Imputing with 1.")
            df['FIR_MONTH'] = df['FIR_MONTH'].fillna(1)
        if nan_before > 0 or nan_after > 0:
            st.info(f"FIR_MONTH: {nan_before} NaN values before conversion, {nan_after} after conversion, all imputed.")
            logging.info(f"FIR_MONTH: {nan_before} NaN before, {nan_after} after, all imputed.")
        logging.info("FIR_MONTH validated.")

# --- Robust Column Detection ---
cols_set = set(df.columns)

def find_col(candidates):
    for c in candidates:
        if c and c.upper() in cols_set:
            return c.upper()
    return None

major_head_col = find_col(['MAJOR_HEAD', 'CRIME TYPE', 'CRIME', 'OFFENCE', 'OFFENCE TYPE'])
district_col = find_col(['DISTRICT'])
minor_head_col = find_col(['MINOR_HEAD'])
ps_col = find_col(['PS', 'POLICE STATION', 'POLICE_STN'])
disp_col = find_col(['TYPE_OF_DISP', 'DISPOSITION', 'DISPOSAL_TYPE', 'FINAL_DISP'])

if not major_head_col:
    st.error(f"❌ Could not find a crime type column. Columns present: {list(df.columns)}")
    logging.error(f"Could not find crime type column. Columns present: {list(df.columns)}")
    st.stop()

# --- Clean Text Fields ---
for c in [major_head_col, ps_col, district_col, minor_head_col, disp_col]:
    if c and c in df.columns:
        df[c] = df[c].astype(str).str.strip().str.title()

# --- Target Selection for Location Prediction ---
target_col = None
min_samples_for_district = 5
min_samples_for_ps = 3

if district_col:
    vc = df[district_col].value_counts()
    if vc.size >= 2 and vc.min() >= min_samples_for_district:
        target_col = district_col
        st.sidebar.write(f"📊 Using {district_col} as target ({vc.size} classes, min {vc.min()} samples).")
    else:
        st.sidebar.write(f"{district_col} has low variance ({vc.size} classes, min {vc.min()} samples).")

if target_col is None and ps_col:
    vcps = df[ps_col].value_counts()
    if vcps.size >= 2 and vcps.min() >= min_samples_for_ps:
        target_col = ps_col
        st.sidebar.write(f"📊 Using {ps_col} as target ({vcps.size} classes, min {vcps.min()} samples).")
    else:
        st.sidebar.write(f"{ps_col} has low variance ({vcps.size if ps_col in df.columns else 0} classes, min {vcps.min() if ps_col in df.columns else 0} samples).")

# --- ML Training for Location Prediction ---
model = None
xgb_model = None
nn_model = None
le_dict = {}
scaler = None
X = None
y = None
df_ml = None

if target_col:
    st.subheader("Training Location Prediction Models")
    keep = [target_col, major_head_col, minor_head_col, 'FIR_MONTH', 'FIR_YEAR',
            'CASE_STAGE', disp_col, 'GRAVE', 'OCCURENCE_PLACE']
    keep = [c for c in keep if c in df.columns]
    df_ml = df[keep].copy().dropna(subset=[target_col])

    min_samples = min_samples_for_district if target_col == district_col else min_samples_for_ps
    class_counts = df_ml[target_col].value_counts()
    good_classes = class_counts[class_counts >= min_samples].index
    df_ml = df_ml[df_ml[target_col].isin(good_classes)]
    st.info(f"ML dataset shape after filtering: {df_ml.shape} (classes: {len(good_classes)})")

    if df_ml[target_col].nunique() < 2:
        st.warning("⚠️ Not enough distinct classes to train ML model.")
        logging.warning("Not enough classes for ML training.")
    else:
        # --- Impute NaN ---
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
        if major_head_col in df_ml.columns and disp_col in df_ml.columns:
            df_ml['CRIME_DISP_INTERACT'] = df_ml[major_head_col].astype(str) + '_' + df_ml[disp_col].astype(str)
            df_ml['CRIME_DISP_INTERACT'] = df_ml['CRIME_DISP_INTERACT'].fillna('Unknown_Unknown')
            st.write("Added CRIME_DISP_INTERACT feature.")
            logging.info("Added CRIME_DISP_INTERACT feature.")

        # --- Label Encoding ---
        cat_cols = df_ml.select_dtypes(include=['object', 'category']).columns.tolist()
        cat_cols = [c for c in cat_cols if c != target_col]
        for col in cat_cols:
            le = LabelEncoder()
            df_ml[col] = le.fit_transform(df_ml[col].astype(str))
            le_dict[col] = le
        target_le = LabelEncoder()
        df_ml[target_col] = target_le.fit_transform(df_ml[target_col].astype(str))
        le_dict[target_col] = target_le

        # --- Train/Test Split ---
        X = df_ml.drop(columns=[target_col])
        y = df_ml[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # --- Standardize Features ---
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        joblib.dump(scaler, 'scaler.pkl')

        # --- RandomForest Training ---
        try:
            st.write("Training RandomForest.")
            model = RandomForestClassifier(n_estimators=200, max_depth=14, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rf_accuracy = accuracy_score(y_test, y_pred)
            st.success(f"✅ RandomForest Accuracy on {target_col}: {rf_accuracy:.3f}")
            logging.info(f"RandomForest trained. Accuracy: {rf_accuracy}")
        except Exception as ex:
            st.error(f"❌ RandomForest training failed: {ex}")
            logging.exception("RandomForest error")
            model = None

        # --- XGBoost Training ---
        try:
            st.write("Training XGBoost.")
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
            st.error(f"❌ XGBoost training failed: {ex}")
            logging.exception("XGBoost error")
            xgb_model = None

        # --- Neural Network Training for Location ---
        def train_nn():
            try:
                st.write("Training Neural Network for Location...")
                tf.compat.v1.reset_default_graph()
                def create_model(neurons=128, layers=2, dropout_rate=0.3, learning_rate=0.001):
                    model = Sequential([
                        Input(shape=(X_train.shape[1],)),
                        Dense(neurons, activation='relu'),
                        BatchNormalization(),
                        Dropout(dropout_rate),
                        *[Dense(neurons // 2, activation='relu') for _ in range(layers - 1)],
                        *[BatchNormalization() for _ in range(layers - 1)],
                        *[Dropout(dropout_rate) for _ in range(layers - 1)],
                        Dense(len(np.unique(y)), activation='softmax')
                    ])
                    model.compile(optimizer=Adam(learning_rate=learning_rate),
                                  loss='categorical_crossentropy',
                                  metrics=['accuracy'])
                    return model

                nn_model = KerasClassifier(model=create_model, verbose=0)
                param_grid = {
                    'model__neurons': [128],
                    'model__layers': [2],
                    'model__dropout_rate': [0.3],
                    'model__learning_rate': [0.001],
                    'batch_size': [32],
                    'epochs': [50]
                }
                grid = GridSearchCV(estimator=nn_model, param_grid=param_grid, cv=3, n_jobs=1)
                y_train_cat = to_categorical(y_train, num_classes=len(np.unique(y)))
                grid_result = grid.fit(X_train_scaled, y_train_cat)
                nn_model = grid_result.best_estimator_
                st.success(f"✅ Neural Network Best Params: {grid_result.best_params_}")
                st.success(f"✅ Neural Network Best CV Score: {grid_result.best_score_:.3f}")
                logging.info(f"Neural Network best params: {grid_result.best_params_}")
                logging.info(f"Neural Network best CV score: {grid_result.best_score_}")

                y_test_cat = to_categorical(y_test, num_classes=len(np.unique(y)))
                nn_pred = nn_model.predict(X_test_scaled)
                nn_accuracy = accuracy_score(y_test, nn_pred)
                st.success(f"✅ Neural Network Accuracy on {target_col}: {nn_accuracy:.3f}")
                logging.info(f"Neural Network trained. Accuracy: {nn_accuracy}")
                return nn_model
            except Exception as ex:
                st.error(f"❌ Neural Network training failed: {ex}")
                logging.exception("Neural Network error")
                return None

        nn_thread = threading.Thread(target=lambda: globals().update(nn_model=train_nn()))
        nn_thread.start()
        nn_thread.join(timeout=300)
        if nn_thread.is_alive():
            st.error("❌ Neural Network training timed out after 5 minutes.")
            logging.error("Neural Network training timed out after 5 minutes.")
            nn_model = None

        # --- Save Models ---
        try:
            if model:
                joblib.dump(model, f'{target_col.lower()}_model_rf.pkl')
            if xgb_model:
                joblib.dump(xgb_model, f'{target_col.lower()}_model_xgb.pkl')
            if nn_model:
                nn_model.model_.save(f'{target_col.lower()}_model_nn.h5')
            if le_dict:
                joblib.dump(le_dict, 'le_dict.pkl')
            if scaler:
                joblib.dump(scaler, 'scaler.pkl')
            if model or xgb_model or nn_model:
                st.success("✅ Models saved successfully.")
                logging.info("Models, encoders, and scaler saved.")
        except Exception as ex:
            st.error(f"❌ Failed to save models: {ex}")
            logging.exception("Model saving error")

        # --- Cross-Validation for RandomForest ---
        if model:
            try:
                st.write("Performing RandomForest Cross-Validation...")
                cv = min(5, max(2, len(y) // 20))
                scores = cross_val_score(model, X, y, cv=cv)
                st.write(f"Cross-Validation Scores (RF): mean={scores.mean():.2f}, std={scores.std():.2f}")
                logging.info(f"RandomForest Cross-validation scores: mean={scores.mean():.2f}, std={scores.std():.2f}")
            except Exception as ex:
                st.warning(f"⚠️ RandomForest Cross-validation failed: {ex}")
                logging.warning(f"RandomForest Cross-validation failed: {ex}")

        # --- Feature Importance Plot (RandomForest) ---
        if model:
            try:
                feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
                fig_feat = px.bar(
                    x=feat_imp[:10].values,
                    y=feat_imp[:10].index,
                    orientation='h',
                    title=f"Top Features Influencing {target_col} (RandomForest)",
                    labels={'x': 'Importance', 'y': 'Feature'},
                    color=feat_imp[:10].values,
                    color_continuous_scale=PROFESSIONAL_PALETTE
                )
                fig_feat.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font_color='#333333',
                    title_font_color='#FF3D00',
                    xaxis_title="Importance",
                    yaxis_title="Feature",
                    xaxis=dict(showgrid=True, gridcolor='#E0E0E0'),
                    yaxis=dict(showgrid=True, gridcolor='#E0E0E0')
                )
                st.plotly_chart(fig_feat)
            except Exception as ex:
                st.warning(f"⚠️ RandomForest feature importance plot failed: {ex}")
                logging.warning(f"RandomForest feature importance plot failed: {ex}")

        # --- Feature Importance Plot (Neural Network) ---
        if nn_model:
            try:
                perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
                perm_imp_df = pd.Series(perm_importance.importances_mean, index=X.columns).sort_values(ascending=False)
                fig_perm = px.bar(
                    x=perm_imp_df[:10].values,
                    y=perm_imp_df[:10].index,
                    orientation='h',
                    title=f"Top Features Influencing {target_col} (Neural Network - Permutation Importance)",
                    labels={'x': 'Importance', 'y': 'Feature'},
                    color=perm_imp_df[:10].values,
                    color_continuous_scale=PROFESSIONAL_PALETTE
                )
                fig_perm.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font_color='#333333',
                    title_font_color='#FF3D00',
                    xaxis_title="Importance",
                    yaxis_title="Feature",
                    xaxis=dict(showgrid=True, gridcolor='#E0E0E0'),
                    yaxis=dict(showgrid=True, gridcolor='#E0E0E0')
                )
                st.plotly_chart(fig_perm)
            except Exception as ex:
                st.warning(f"⚠️ Neural Network permutation importance failed: {ex}")
                logging.warning(f"Neural Network permutation importance failed: {ex}")

        # --- Confusion Matrix ---
        if model:
            try:
                unique_labels = model.classes_
                cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
                test_label_counts = pd.Series(y_test).value_counts().loc[unique_labels].fillna(0)
                top_labels_positions = np.argsort(test_label_counts.values)[::-1][:10]
                top_encoded_labels = unique_labels[top_labels_positions]
                top_display = le_dict[target_col].inverse_transform(top_encoded_labels)
                cm_small = cm[np.ix_(top_labels_positions, top_labels_positions)]

                PROFESSIONAL_COLORSCALE = [
                    [0, '#F5F5F5'], [0.2, '#E0E0E0'], [0.4, '#B0BEC5'],
                    [0.6, '#78909C'], [0.8, '#455A64'], [1.0, '#263238']
                ]

                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm_small,
                    x=top_display,
                    y=top_display,
                    colorscale=PROFESSIONAL_COLORSCALE,
                    text=cm_small,
                    texttemplate="%{text}",
                    textfont={"size": 12, "color": "white"},
                    colorbar=dict(title="Count", titleside="right", titlefont=dict(size=14, color='#263238'))
                ))
                fig_cm.update_layout(
                    title=f"Confusion Matrix for {target_col} (Top 10 Classes)",
                    xaxis_title=f"Predicted {target_col}",
                    yaxis_title=f"Actual {target_col}",
                    xaxis=dict(tickangle=45, tickfont=dict(size=12, color='#263238')),
                    yaxis=dict(tickfont=dict(size=12, color='#263238')),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    margin=dict(l=50, r=50, t=100, b=100),
                    width=800,
                    height=600
                )
                st.plotly_chart(fig_cm)
            except Exception as ex:
                st.warning(f"⚠️ Confusion matrix plot failed: {ex}")
                logging.warning(f"Confusion matrix plot failed: {ex}")

# --- Treemap Visualization ---
try:
    location_col = district_col if district_col else ps_col
    if location_col and major_head_col:
        crime_pivot = df.groupby([location_col, major_head_col]).size().reset_index(name='Count')
        if not crime_pivot.empty:
            fig_tree = px.treemap(crime_pivot, path=[location_col, major_head_col], values='Count',
                                  title=f'Crime Hierarchy by {location_col} and Type')
            st.plotly_chart(fig_tree)
            logging.info("Treemap visualization shown.")
except Exception as ex:
    st.warning(f"⚠️ Treemap visualization failed: {ex}")
    logging.warning(f"Treemap visualization failed: {ex}")

# --- Disposition Model Training ---
disp_model = None
disp_le = None
disp_labels = None

if disp_col:
    df_disp = df[[major_head_col, minor_head_col, 'FIR_MONTH', 'FIR_YEAR', 'OCCURENCE_PLACE', disp_col]].copy().dropna(subset=[disp_col])
    valid_dispositions = ['Police Station', 'Court', 'Self-Compromise', 'Withdraw']
    df_disp = df_disp[df_disp[disp_col].isin(valid_dispositions)]
    
    if df_disp.shape[0] >= 50 and df_disp[disp_col].nunique() >= 2:
        # Text feature engineering
        text_column_name = "_SIM_TEXT"
        df[text_column_name] = df[[major_head_col, minor_head_col, 'OCCURENCE_PLACE']].apply(lambda r: " || ".join(r.astype(str)), axis=1)
        if 'tfidf_vect' not in st.session_state:
            vect = TfidfVectorizer(max_features=5000, stop_words='english')
            st.session_state['tfidf_matrix'] = vect.fit_transform(df[text_column_name].astype(str))
            st.session_state['tfidf_vect'] = vect
        
        X_disp = pd.DataFrame()
        X_disp['FIR_MONTH'] = df_disp['FIR_MONTH'].astype(float).fillna(1)
        X_disp['FIR_YEAR'] = df_disp['FIR_YEAR'].astype(float).fillna(df_disp['FIR_YEAR'].median())
        
        arr = st.session_state['tfidf_matrix'].toarray()[df_disp.index]
        k = min(32, arr.shape[1])
        chunk_size = max(1, arr.shape[1] // k)
        agg = np.array([arr[:, i:i+chunk_size].mean(axis=1) for i in range(0, arr.shape[1], chunk_size)]).T
        for i in range(min(16, agg.shape[1])):
            X_disp[f"txt_{i}"] = agg[:, i]
        
        y_disp = df_disp[disp_col]
        disp_le = LabelEncoder()
        y_disp_encoded = disp_le.fit_transform(y_disp)
        le_dict[disp_col] = disp_le
        disp_labels = disp_le.classes_
        
        # Balance classes
        combined = X_disp.copy()
        combined[disp_col] = y_disp
        max_count = combined[disp_col].value_counts().max()
        dfs = [resample(grp, replace=True, n_samples=max_count, random_state=42) if len(grp) < max_count else grp for cls, grp in combined.groupby(disp_col)]
        balanced = pd.concat(dfs, ignore_index=True)
        y_disp_bal = balanced[disp_col]
        X_disp_bal = balanced.drop(columns=[disp_col])
        y_disp_bal_encoded = disp_le.transform(y_disp_bal)
        
        # Train disposition neural network
        def create_disp_nn():
            model = Sequential([
                Input(shape=(X_disp_bal.shape[1],)),
                Dense(256, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(64, activation='relu'),
                Dense(len(np.unique(y_disp_encoded)), activation='softmax')
            ])
            model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            return model
        
        disp_model = KerasClassifier(model=create_disp_nn, epochs=100, batch_size=32, verbose=0)
        disp_model.fit(X_disp_bal, y_disp_bal_encoded)
        disp_accuracy = accuracy_score(y_disp_bal_encoded, disp_model.predict(X_disp_bal))
        st.success(f"✅ Disposition Neural Network Accuracy: {disp_accuracy:.3f}")
        disp_model.model_.save('disposition_model_nn.h5')
    else:
        st.warning("⚠️ Insufficient data for disposition model.")
        logging.info("Not enough disposition-labeled data.")
else:
    st.warning("⚠️ No disposition column found.")
    logging.info("No disposition column found.")

# --- Prediction Section ---
st.header("Predict Crime Location & Case Outcome")
tab1, tab2 = st.tabs(["📍 Location Prediction", "📝 Lodge New FIR"])

with tab1:
    st.subheader("Predict Crime Location")
    user_input = st.text_input("Enter Crime Type:", key="crime_type_input")
    if user_input:
        crime_choices = [str(c).lower() for c in le_dict[major_head_col].classes_]
        best_match, score = process.extractOne(user_input.strip().lower(), crime_choices)
        chosen_crime = best_match if score >= 70 else user_input.strip().lower()
        if score >= 70:
            st.write(f"Interpreted as: **{chosen_crime.title()}** (Confidence: {score}%)")
        else:
            st.warning("⚠️ No close match found. Using exact input.")
        
        # Frequency-based prediction
        freq_target_col = ps_col if ps_col else district_col
        if freq_target_col:
            try:
                filtered = df[df[major_head_col].astype(str).str.lower().str.contains(chosen_crime, na=False)]
                if filtered.empty:
                    st.warning(f"⚠️ No historical records for '{chosen_crime.title()}'.")
                else:
                    counts = filtered[freq_target_col].value_counts(normalize=True).head(5)
                    freq_df = counts.reset_index()
                    freq_df.columns = [freq_target_col, 'Proportion']
                    st.subheader(f"Frequency-based Top {freq_target_col}")
                    st.dataframe(freq_df)
                    out = io.StringIO()
                    freq_df.to_csv(out, index=False)
                    st.download_button("Download Frequency Predictions", data=out.getvalue(),
                                       file_name=f"freq_predictions_{chosen_crime.replace(' ', '_')}.csv")
                    fig_freq = px.bar(freq_df, x='Proportion', y=freq_target_col, orientation='h',
                                      title=f"Top {freq_target_col} for {chosen_crime.title()}",
                                      color_discrete_sequence=['#FF3D00'])
                    st.plotly_chart(fig_freq)
            except Exception as ex:
                st.error(f"❌ Frequency-based prediction failed: {ex}")
                logging.exception("Frequency-based prediction error")

        # ML predictions
        if model and le_dict and major_head_col in le_dict:
            try:
                sample_dict = {col: df_ml[col].mode().iloc[0] if col in df_ml.columns else 0 for col in X.columns}
                crime_classes_lower = [c.lower() for c in le_dict[major_head_col].classes_]
                if chosen_crime in crime_classes_lower:
                    sample_dict[major_head_col] = le_dict[major_head_col].transform([le_dict[major_head_col].classes_[crime_classes_lower.index(chosen_crime)]])[0]
                sample = pd.DataFrame([sample_dict], columns=X.columns)
                
                # RandomForest prediction
                probs = model.predict_proba(sample)[0]
                top_idx = np.argsort(probs)[::-1][:5]
                top_locs = le_dict[target_col].inverse_transform(model.classes_[top_idx])
                st.subheader(f"RandomForest Prediction for {target_col}")
                for i, loc in enumerate(top_locs, 1):
                    st.write(f"{i}. {loc} (Probability: {probs[top_idx[i-1]]:.3f})")
                
                # XGBoost prediction
                if xgb_model:
                    xgb_probs = xgb_model.predict_proba(sample)[0]
                    xgb_top_idx = np.argsort(xgb_probs)[::-1][:5]
                    xgb_top_locs = le_dict[target_col].inverse_transform(xgb_model.classes_[xgb_top_idx])
                    st.subheader(f"XGBoost Prediction for {target_col}")
                    for i, loc in enumerate(xgb_top_locs, 1):
                        st.write(f"{i}. {loc} (Probability: {xgb_probs[xgb_top_idx[i-1]]:.3f})")
                
                # Neural Network prediction
                if nn_model:
                    sample_scaled = scaler.transform(sample)
                    nn_probs = nn_model.predict_proba(sample_scaled)[0]
                    nn_top_idx = np.argsort(nn_probs)[::-1][:5]
                    nn_top_locs = le_dict[target_col].inverse_transform(np.arange(len(np.unique(y)))[nn_top_idx])
                    st.subheader(f"Neural Network Prediction for {target_col}")
                    for i, loc in enumerate(nn_top_locs, 1):
                        st.write(f"{i}. {loc} (Probability: {nn_probs[nn_top_idx[i-1]]:.3f})")
                
                fig = px.bar(x=probs[top_idx], y=top_locs, orientation='h',
                             title=f"Top {target_col}s (RandomForest)", labels={'x': 'Probability', 'y': target_col},
                             color_discrete_sequence=['#FF3D00'])
                st.plotly_chart(fig)
            except Exception as ex:
                st.error(f"❌ ML prediction failed: {ex}")
                logging.exception("ML prediction error")

with tab2:
    st.subheader("Lodge New FIR")
    with st.form("new_fir_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            inp_major = st.text_input("Crime Type", key="fir_major")
            inp_minor = st.text_input("Minor Head", key="fir_minor")
            inp_place = st.text_input("Occurrence Place", key="fir_place")
        with col2:
            inp_month = st.number_input("Month (1-12)", min_value=1, max_value=12, value=1, key="fir_month")
            inp_year = st.number_input("Year", min_value=2000, max_value=2030, value=2025, key="fir_year")
            inp_brief = st.text_area("Brief Facts", key="fir_brief")
        submit_fir = st.form_submit_button("Analyze & Predict")

    if submit_fir:
        new_fingerprint = " || ".join([inp_major, inp_minor, inp_place, inp_brief])
        st.info("Analyzing similar cases and predicting case closure...")

        # Find similar cases with FIR numbers
        def find_similar_cases(new_text, top_k=5):
            results = []
            try:
                if 'tfidf_vect' in st.session_state:
                    new_vec = st.session_state['tfidf_vect'].transform([new_text])
                    sims = cosine_similarity(new_vec, st.session_state['tfidf_matrix'])[0]
                    idxs = np.argsort(sims)[::-1][:top_k]
                    for i in idxs:
                        row = df.iloc[i].copy()
                        row['_SIM_SCORE'] = float(sims[i])
                        results.append(row)
                else:
                    tmp = df[df[major_head_col].astype(str).str.lower().str.contains(new_text.split("||")[0].strip().lower(), na=False)]
                    for _, r in tmp.head(top_k).iterrows():
                        r = r.copy()
                        r['_SIM_SCORE'] = 1.0
                        results.append(r)
            except Exception as e:
                logging.exception("Similarity computation failed.")
            return results

        similar = find_similar_cases(new_fingerprint)
        if similar:
            sim_df = pd.DataFrame(similar)
            sim_df['Status'] = sim_df.apply(lambda r: 'Closed' if any(x in str(r.get(c, '')).lower() for c in ['CASE_STAGE', disp_col] for x in ['close', 'disposed', 'final', 'withdraw', 'compromise']) else 'Running', axis=1)
            st.subheader("Similar Historical Cases")
            display_cols = ['FIR_NO'] + [c for c in [ps_col, district_col, major_head_col, minor_head_col, 'CASE_STAGE', disp_col, 'Status', '_SIM_SCORE'] if c in sim_df.columns or c == 'Status' or c == '_SIM_SCORE']
            st.table(sim_df[display_cols].head())
            status_counts = sim_df['Status'].value_counts()
            cols = st.columns(len(status_counts))
            for i, (k, v) in enumerate(status_counts.items()):
                cols[i].metric(k, v)
        
        # Predict disposition
        if disp_model:
            X_sample = pd.DataFrame()
            X_sample['FIR_MONTH'] = [float(inp_month)]
            X_sample['FIR_YEAR'] = [float(inp_year)]
            new_vec = st.session_state['tfidf_vect'].transform([new_fingerprint])
            arr = new_vec.toarray()
            k = min(32, arr.shape[1])
            chunk_size = max(1, arr.shape[1] // k)
            agg = np.array([arr[:, i:i+chunk_size].mean(axis=1) for i in range(0, arr.shape[1], chunk_size)]).T
            for i in range(min(16, agg.shape[1])):
                X_sample[f"txt_{i}"] = agg[:, i]
            
            try:
                probs = disp_model.predict_proba(X_sample)[0]
                disp_probs = {disp_le.classes_[i]: float(p) for i, p in enumerate(probs)}
                st.subheader("Case Closure Probabilities")
                st.write("Your case has high probability of:")
                fig = px.bar(x=list(disp_probs.values()), y=list(disp_probs.keys()), orientation='h',
                             title="Closure Outcome Probabilities", labels={'x': 'Probability', 'y': 'Outcome'},
                             color_discrete_sequence=['#FF3D00'])
                st.plotly_chart(fig)
                for k, v in disp_probs.items():
                    st.write(f"{k}: {v*100:.1f}%")
            except Exception as e:
                st.error(f"❌ Disposition prediction failed: {e}")
                logging.exception("Disposition prediction error")
        else:
            # Frequency-based fallback
            filt = df[df[major_head_col].astype(str).str.lower().str.contains(inp_major.strip().lower(), na=False)]
            if disp_col and not filt.empty:
                freq = filt[disp_col].value_counts(normalize=True).head(4)
                freq_df = freq.reset_index()
                freq_df.columns = [disp_col, 'Proportion']
                st.subheader("Historical Disposition Proportions")
                st.dataframe(freq_df)
                fig = px.bar(freq_df, x='Proportion', y=disp_col, orientation='h',
                             title="Historical Closure Proportions", color_discrete_sequence=['#FF3D00'])
                st.plotly_chart(fig)

st.markdown("---")
st.write("✅ Process completed. Check `crime_predictor.log` for detailed logs.")
