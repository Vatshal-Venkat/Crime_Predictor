Crime Predictor Dashboard
This is a Streamlit-based web application designed to predict crime locations (district or police station) and case closure probabilities based on historical First Information Report (FIR) data. The tool leverages machine learning models (RandomForest, XGBoost, Neural Networks) and includes features for analyzing similar historical cases when lodging new FIRs.
Overview
The Crime Predictor Dashboard allows users to:

Upload an Excel file containing FIR data.
Predict crime locations using multiple ML models.
Analyze similar historical cases and their status (closed/running) when lodging a new FIR.
Predict case closure probabilities across four outcomes: Police Station, Court, Self-Compromise, and Withdraw.

The application features a clean, interactive UI with visualizations (bar charts, treemaps, confusion matrices) and robust error handling.
Features

Data Upload: Upload Excel files (e.g., FIR_data_2022.xlsx) via a sidebar interface.
Location Prediction: Uses RandomForest, XGBoost, and Neural Networks to predict crime locations.
Similar Case Analysis: Identifies and displays the top 5 similar historical cases with their status.
Disposition Prediction: Predicts case closure probabilities using an enhanced neural network.
Visualizations: Includes feature importance plots, confusion matrices, and treemaps.
Logging: Detailed logs saved to crime_predictor.log for debugging.
Model Persistence: Saves trained models, encoders, and scalers for reuse.

Installation
Prerequisites

Python 3.8 or higher
Pip package manager

Dependencies
Install the required packages using the following command:
pip install streamlit pandas numpy scikit-learn tensorflow scikeras plotly fuzzywuzzy python-Levenshtein joblib xgboost

Setup

Clone the repository or download the crime_predictor.py file.
Ensure you have an Excel file with FIR data (e.g., columns like MAJOR_HEAD, DISTRICT, PS, TYPE_OF_DISP, FIR_MONTH, FIR_YEAR).
Run the application:streamlit run crime_predictor.py



Usage

Upload Data: Open the app in your browser (default: http://localhost:8501) and upload an Excel file via the sidebar.
Preview Data: Check the first few rows using the "Show first few rows of data" checkbox in the sidebar.
Location Prediction:
Go to the "Location Prediction" tab.
Enter a crime type to see frequency-based and ML-based predictions (top 5 locations).


Lodge New FIR:
Go to the "Lodge New FIR" tab.
Fill out the form (Crime Type, Minor Head, Occurrence Place, Month, Year, Brief Facts).
Click "Analyze & Predict" to see similar cases and closure probabilities.


Visualizations: Explore treemaps, confusion matrices, and feature importance plots for insights.

Data Requirements

Excel file with columns such as:
MAJOR_HEAD (crime type)
DISTRICT or PS (police station)
TYPE_OF_DISP (disposition, e.g., Police Station, Court)
FIR_MONTH, FIR_YEAR
Optional: MINOR_HEAD, CASE_STAGE, OCCURENCE_PLACE


At least 50 rows with valid disposition data for the disposition model.
No missing critical target columns (e.g., DISTRICT or PS).

Troubleshooting

Uploader Not Visible: Ensure Streamlit is updated (pip install streamlit --upgrade) and the sidebar is not collapsed.
Upload Fails: Check file format (.xlsx) and size. For large files, add a progress bar by modifying the upload block.
Uniform Predictions: Verify data has sufficient variety and balance in target classes. Adjust min_samples_for_district or min_samples_for_ps in the code if needed.
Errors: Review crime_predictor.log for details and ensure all dependencies are installed.

Contributing
Feel free to fork this repository, submit issues, or propose enhancements. Pull requests are welcome!
License
This project is open-source under the MIT License. See LICENSE for details (if applicable).
Contact
For questions or support, reach out via the repository issues page or contact the developer.