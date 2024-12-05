import docx2txt
import re
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import joblib
from transformers import pipeline
import numpy as np
import logging
import shap
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def explain_with_shap(model, X_train, X_test, feature_names):
    """Generate SHAP explanations for the model predictions."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Check if it's binary classification (shap_values is a list)
    if isinstance(shap_values, list):
        # Use shap_values[1] for the positive class in binary classification
        if len(shap_values) > 1:
            logger.info("Generating SHAP plots for binary classification (positive class).")
            shap.summary_plot(shap_values[1], X_test, feature_names=feature_names)
        else:
            logger.info("Generating SHAP plots for single-class classification.")
            shap.summary_plot(shap_values[0], X_test, feature_names=feature_names)
    else:
        # Handle single-class classification or regression (shap_values is a matrix)
        logger.info("Generating SHAP plots for regression or single-class output.")
        shap.summary_plot(shap_values, X_test, feature_names=feature_names)


def extract_value(pattern, text, default=None, data_type=float):
    """Extract and convert values from text, handling non-numeric values."""
    match = re.search(pattern, text)
    if match:
        value = match.group(1)
        
        # Clean up commas and handle non-numeric values
        if isinstance(value, str):
            value = value.replace(",", "")  # Remove commas
            
            # If the value is non-numeric (like 'Unknown'), return the default value
            try:
                return data_type(value)
            except ValueError:
                return default
        else:
            return default
    return default


def extract_patient_data_and_ranges(report_path):
    """Extract patient health data and reference ranges from a .docx report."""
    try:
        text = docx2txt.process(report_path)
        if not text.strip():
            logging.warning(f"Empty report: {report_path}")
            return None

        patient_data = {
            # Patient Demographics
            "Patient Name": extract_value(r"Mr\. ([\w\s]+)", text, default="Unknown", data_type=str),
            "Age": extract_value(r"Age/Gender (\d+)", text, default=0, data_type=int),
            "Gender": extract_value(r"Age/Gender \d+ yr\(s\) / ([MF])", text, default="Unknown", data_type=str),
            # CBC Parameters
            "WBC Count": extract_value(r"WBC Count\s+([\d,]+)", text),
            "Hemoglobin": extract_value(r"Hemoglobin\s+([\d.]+)", text),
            "Platelet Count": extract_value(r"Platelet Count\s+([\d,]+)", text),
            # Lipid Profile
            "Total Cholesterol": extract_value(r"Total Cholesterol\s+([\d.]+)", text),
            "LDL": extract_value(r"LDL\s+\(Bad Cholesterol\)\s+([\d.]+)", text),
            "HDL": extract_value(r"HDL\s+\(Good Cholesterol\)\s+([\d.]+)", text),
            "Triglycerides": extract_value(r"Triglycerides\s+([\d.]+)", text),
            # Diabetes Metrics
            "Fasting Blood Glucose": extract_value(r"Fasting Blood Glucose\s+([\d.]+)", text),
            "HbA1c": extract_value(r"HbA1c\s+([\d.]+)", text),
            # Sleep Study Metrics
            "Apnea-Hypopnea Index (AHI)": extract_value(r"Apnea-Hypopnea Index \(AHI\)\s+([\d.]+)", text),
            "Oxygen Saturation (SpO2)": extract_value(r"Oxygen Saturation \(SpO2\)\s+([\d.]+)", text),
            "BMI": extract_value(r"BMI\s+([\d.]+)", text),
            # Cardiovascular Metrics
            "Blood Pressure": extract_value(r"Blood Pressure\s+([\d/]+)", text, default="Unknown", data_type=str),
            "Prothrombin Time": extract_value(r"Prothrombin Time\s+([\d.]+)", text),
            "International Normalized Ratio (INR)": extract_value(r"International Normalized Ratio\s+([\d.]+)", text),
            # General Metabolic Markers
            "Sodium": extract_value(r"Sodium\s+([\d.]+)", text),
            "Potassium": extract_value(r"Potassium\s+([\d.]+)", text),
            "Chloride": extract_value(r"Chloride\s+([\d.]+)", text),
            "C-Reactive Protein (CRP)": extract_value(r"C-Reactive Protein\s+([\d.]+)", text),
            # Additional Sleep Study Metrics
            "Sleep Latency": extract_value(r"Sleep Latency\s+([\d.]+)", text),
            "REM Sleep Duration": extract_value(r"REM Sleep Duration\s+([\d.]+)", text),
            "Non-REM Sleep Duration": extract_value(r"Non-REM Sleep Duration\s+([\d.]+)", text),
            "Total Sleep Duration": extract_value(r"Total Sleep Duration\s+([\d'\":]+)", text, default="Unknown", data_type=str),
        }

        return patient_data

    except Exception as e:
        logging.error(f"Error processing report {report_path}: {e}")
        return None


def process_reports(main_folder):
    """Process all .docx reports in the specified folder and its subfolders."""
    reports = []
    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)
        if os.path.isdir(subfolder_path):
            logging.info(f"Processing subfolder: {subfolder_path}")
            for filename in os.listdir(subfolder_path):
                if filename.endswith(".docx"):
                    report_path = os.path.join(subfolder_path, filename)
                    data = extract_patient_data_and_ranges(report_path)
                    if data:
                        reports.append(data)
                    else:
                        logging.warning(f"Skipped report: {report_path}")
    return pd.DataFrame(reports)


def assign_risk_level(row):
    """Assign a risk level based on health metrics and highlight abnormal values across various diseases."""
    detected_disease = None
    explanation = []
    abnormal_values = []

    # Helper function to safely convert values to float
    def safe_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # CBC Parameters
    if "WBC Count" in row and pd.notna(row["WBC Count"]) and safe_float(row["WBC Count"]) > 11:
        abnormal_values.append("WBC Count > 11 x10^3/µL")
    if "Hemoglobin" in row and pd.notna(row["Hemoglobin"]) and safe_float(row["Hemoglobin"]) < 13:
        abnormal_values.append("Hemoglobin < 13 g/dL")
    if "Platelet Count" in row and pd.notna(row["Platelet Count"]) and safe_float(row["Platelet Count"]) < 150:
        abnormal_values.append("Platelet Count < 150 x10^3/µL")

    # Lipid Profile
    if "Total Cholesterol" in row and pd.notna(row["Total Cholesterol"]) and safe_float(row["Total Cholesterol"]) > 240:
        abnormal_values.append("Total Cholesterol > 240 mg/dL")
    if "LDL" in row and pd.notna(row["LDL"]) and safe_float(row["LDL"]) > 160:
        abnormal_values.append("LDL > 160 mg/dL")
    if "HDL" in row and pd.notna(row["HDL"]) and safe_float(row["HDL"]) < 40:
        abnormal_values.append("HDL < 40 mg/dL")
    if "Triglycerides" in row and pd.notna(row["Triglycerides"]) and safe_float(row["Triglycerides"]) > 150:
        abnormal_values.append("Triglycerides > 150 mg/dL")

    # Diabetes Metrics
    if "Fasting Blood Glucose" in row and pd.notna(row["Fasting Blood Glucose"]) and safe_float(row["Fasting Blood Glucose"]) > 126:
        abnormal_values.append("Fasting Blood Glucose > 126 mg/dL")
    if "HbA1c" in row and pd.notna(row["HbA1c"]) and safe_float(row["HbA1c"]) > 6.5:
        abnormal_values.append("HbA1c > 6.5%")

    # Sleep Study Metrics
    if "Apnea-Hypopnea Index (AHI)" in row and pd.notna(row["Apnea-Hypopnea Index (AHI)"]) and safe_float(row["Apnea-Hypopnea Index (AHI)"]) > 15:
        abnormal_values.append("Apnea-Hypopnea Index (AHI) > 15")
    if "Oxygen Saturation (SpO2)" in row and pd.notna(row["Oxygen Saturation (SpO2)"]) and safe_float(row["Oxygen Saturation (SpO2)"]) < 90:
        abnormal_values.append("Oxygen Saturation (SpO2) < 90%")
    if "BMI" in row and pd.notna(row["BMI"]) and safe_float(row["BMI"]) > 30:
        abnormal_values.append("BMI > 30")
    if "Total Sleep Duration" in row and pd.notna(row["Total Sleep Duration"]):
        sleep_duration = safe_float(row["Total Sleep Duration"])
        if sleep_duration is not None and sleep_duration < 6:
            abnormal_values.append("Total Sleep Duration < 6 hours")

    # Cardiovascular Metrics
    if "Blood Pressure" in row and pd.notna(row["Blood Pressure"]):
        try:
            systolic, diastolic = map(int, row["Blood Pressure"].split('/'))
            if systolic > 140 or diastolic > 90:
                abnormal_values.append("Blood Pressure > 140/90 mmHg")
        except (ValueError, AttributeError):
            abnormal_values.append("Blood Pressure: Missing or invalid data")

    # General Metabolic Markers
    if "C-Reactive Protein (CRP)" in row and pd.notna(row["C-Reactive Protein (CRP)"]) and safe_float(row["C-Reactive Protein (CRP)"]) > 3:
        abnormal_values.append("C-Reactive Protein (CRP) > 3 mg/L")

    # Determine the detected disease
    if abnormal_values:
        detected_disease = "Abnormalities Detected"
        explanation = "Patient shows signs of potential health risks based on abnormal metrics."
    else:
        detected_disease = "Healthy"
        explanation = "No significant abnormalities detected."

    # Assign risk level
    risk_level = "High Risk" if abnormal_values else "Low Risk"

    return risk_level, detected_disease, explanation, abnormal_values




def preprocess_new_patient_data(new_patient_data, scaler):
    """Preprocess new patient data, handle missing values, and prepare for prediction."""
    # Convert 'Unknown' to NaN
    new_patient_data.replace('Unknown', pd.NA, inplace=True)
    
    # Convert all columns to numeric values (coerce errors to NaN)
    new_patient_data = new_patient_data.apply(pd.to_numeric, errors='coerce')

    # Handle missing values - Impute missing data with 'mean' strategy
    imputer = SimpleImputer(strategy="mean")
    
    # Identify numeric columns only
    numeric_cols = new_patient_data.select_dtypes(include=[np.number]).columns

    # Remove columns with all missing values (if any), or fill them with 0 (or any default value)
    for col in numeric_cols:
        if new_patient_data[col].isnull().all():
            new_patient_data[col] = 0  # You could also use df[col].fillna(0), depending on the context

    # Perform imputation only on valid numeric columns
    new_patient_data[numeric_cols] = imputer.fit_transform(new_patient_data[numeric_cols])
    
    # Ensure the data is in the same format as the model was trained on
    feature_names = scaler.feature_names_in_
    for col in feature_names:
        if col not in new_patient_data.columns:
            new_patient_data[col] = 0  # Add missing columns with default value

    # Align column order with the model features
    new_patient_data = new_patient_data[feature_names]
    
    # Scale the features using the pre-trained scaler
    new_patient_scaled = scaler.transform(new_patient_data)
    
    return new_patient_scaled




def train_model(data_file, num_epochs=10, test_size=0.2, random_state=42):
    """Train a Random Forest model using an epoch-based approach and evaluate performance."""
    df = pd.read_csv(data_file)

    # Replace 'Unknown' with NaN across all columns
    df.replace("Unknown", pd.NA, inplace=True)

    # Parse 'Blood Pressure' into 'Systolic' and 'Diastolic', handle 'Unknown' values
    if "Blood Pressure" in df.columns:
        df["Blood Pressure"] = df["Blood Pressure"].replace("Unknown", pd.NA)
        bp_split = df["Blood Pressure"].str.split("/", expand=True)
        df["Systolic"] = pd.to_numeric(bp_split[0], errors="coerce")
        df["Diastolic"] = pd.to_numeric(bp_split[1], errors="coerce")
        df.drop(columns=["Blood Pressure"], inplace=True)

    # Convert pd.NA to NaN for compatibility with imputer
    df = df.apply(pd.to_numeric, errors="coerce")

    # Handle missing values
    imputer = SimpleImputer(strategy="mean")
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Drop columns with all NaN values
    nan_cols = [col for col in numeric_cols if df[col].isna().all()]
    if nan_cols:
        logger.warning(f"Skipping columns with all NaN values: {nan_cols}")
        df.drop(columns=nan_cols, inplace=True)

    # Recalculate numeric columns and apply imputation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    # Convert categorical columns to numeric
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"M": 1, "F": 0, "Unknown": -1})

    # Assign risk levels dynamically
    df["Risk Level"], df["Detected Disease"], df["Explanation"], df["Abnormal Values"] = zip(
        *df.apply(assign_risk_level, axis=1)
    )

    # Define target and feature columns, ensuring only valid columns are dropped
    excluded_columns = {"Patient Name", "Risk Level", "Detected Disease", "Explanation", "Abnormal Values"}
    existing_columns = set(df.columns)
    columns_to_drop = excluded_columns.intersection(existing_columns)

    # Train-test split
    X = df.drop(columns=columns_to_drop)
    y = df["Risk Level"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize base hyperparameters
    base_estimators = 100
    base_max_depth = None

    for epoch in range(num_epochs):
        # Adjust hyperparameters dynamically
        n_estimators = base_estimators + epoch * 50
        max_depth = base_max_depth if epoch % 2 == 0 else 10 + epoch

        # Initialize and train the model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        model.fit(X_train, y_train)

        # Evaluate on the training data
        y_train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        logger.info(f"Epoch {epoch + 1}/{num_epochs} - Training Accuracy: {train_acc:.2f}")

        # Evaluate on the testing data
        y_test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        logger.info(f"Epoch {epoch + 1}/{num_epochs} - Testing Accuracy: {test_acc:.2f}")

    # Save the trained model and scaler
    joblib.dump(model, "healthmate.pkl")
    joblib.dump(scaler, "healthmate_scaler.pkl")
    logger.info("Model and scaler saved as 'healthmate.pkl' and 'healthmate_scaler.pkl'.")

    # SHAP explanations
    feature_names = X.columns
    explain_with_shap(model, X_train, X_test, feature_names)

    return model, scaler



def recommend_health_improvements(risk_level, abnormal_values):
    """
    Provide specific health recommendations based on risk level and abnormal metrics.
    """
    try:
        if not abnormal_values:
            return "No significant health issues detected. Maintain a balanced diet and regular exercise.", 1.0

        # Define recommendations for known abnormalities
        abnormality_recommendations = {
            "Triglycerides > 150 mg/dL": "Consider reducing your dietary fat intake and incorporating regular aerobic exercise like walking or cycling. Aim for at least 30 minutes of exercise most days of the week.",
            "HbA1c > 6.5%": "Monitor your blood sugar levels closely. Follow a diabetic-friendly diet, including whole grains, lean proteins, and plenty of vegetables. Consult with a healthcare provider about medication if needed.",
            "BMI > 30": "Consult a dietitian to develop a personalized weight management plan. Aim for gradual weight loss with a balanced diet and regular physical activity.",
            "Blood Pressure > 140/90 mmHg": "Adopt a low-sodium diet (aim for less than 2,300 mg of sodium per day), limit alcohol intake, manage stress, and consult your doctor about antihypertensive medications.",
            "C-Reactive Protein (CRP) > 3 mg/L": "Investigate potential underlying causes of inflammation, such as infection, autoimmune disorders, or chronic diseases. A healthy anti-inflammatory diet, including omega-3 rich foods, can help.",
            "LDL > 160 mg/dL": "Reduce your intake of saturated fats and trans fats. Incorporate more fiber-rich foods like oats, beans, and fruits. Consider discussing cholesterol-lowering medications with your healthcare provider.",
            "HDL < 40 mg/dL": "Increase physical activity, such as brisk walking or swimming, and consume heart-healthy fats like those found in fish, nuts, and avocados to boost HDL cholesterol.",
            "WBC Count > 11 x10^3/µL": "An elevated white blood cell count may indicate infection or inflammation. It's essential to discuss this with your doctor for further evaluation and potential treatment.",
            "Hemoglobin < 13 g/dL": "Low hemoglobin levels could suggest anemia. It's important to address any underlying causes, such as iron or vitamin deficiencies. Consider incorporating iron-rich foods like spinach, beans, and red meat into your diet.",
            "Platelet Count < 150 x10^3/µL": "A low platelet count can increase the risk of bleeding. It's important to discuss this result with your doctor to rule out underlying conditions like infections, medications, or bone marrow disorders.",
            "Total Cholesterol > 240 mg/dL": "Consider reducing saturated fat and cholesterol in your diet. Focus on heart-healthy foods like fruits, vegetables, whole grains, and lean proteins. Regular exercise is also crucial.",
            "Fasting Blood Glucose > 126 mg/dL": "Elevated fasting blood glucose levels may indicate prediabetes or diabetes. Focus on a balanced diet rich in fiber and low in processed sugars, and discuss management options with your doctor.",
            "Apnea-Hypopnea Index (AHI) > 15": "If your AHI is elevated, you may have obstructive sleep apnea. A sleep study and consultation with a sleep specialist are recommended. Lifestyle changes like weight loss and avoiding alcohol may help.",
            "Oxygen Saturation (SpO2) < 90%": "Low oxygen saturation could indicate a respiratory issue. If this is persistent, a visit to a healthcare provider is necessary for further investigation.",
            "Total Sleep Duration < 6 hours": "Aim to improve sleep hygiene and aim for at least 7-9 hours of sleep each night. Practice relaxation techniques and avoid caffeine late in the day to enhance sleep quality.",
            "Blood Pressure > 140/90 mmHg": "Elevated blood pressure requires lifestyle changes, such as reducing sodium intake, increasing physical activity, and possibly taking medications as prescribed by your doctor."
        }

        # Generate recommendations for detected abnormalities
        recommendations = [
            abnormality_recommendations[abnormality]
            for abnormality in abnormal_values
            if abnormality in abnormality_recommendations
        ]

        # Combine recommendations into a single response
        if recommendations:
            return "\n".join(recommendations), 1.0
        else:
            return "No specific recommendations could be generated. Please consult a specialist.", 0.5

    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return "Unable to generate recommendations. Please consult a specialist.", 0.0






def log_recommendations_and_results(new_patient_data, risk_level, detected_disease, explanation, abnormal_values, recommendations, confidence_score):
    """Log detailed patient results and recommendations."""
    logging.info("\n--- Patient Health Analysis ---")
    logging.info(f"Risk Level: {risk_level}")
    logging.info(f"Detected Disease: {detected_disease}")
    logging.info(f"Explanation: {explanation}")
    logging.info(f"Abnormal Values: {', '.join(abnormal_values)}")

    logging.info("\n--- Health Recommendations ---")
    logging.info(f"Model-Generated Recommendations: {recommendations} (Confidence: {confidence_score:.2f})")

    # Add actionable advice for common abnormalities
    if "Triglycerides > 150 mg/dL" in abnormal_values:
        logging.info("- Reduce dietary fat intake and incorporate regular exercise.")
    if "HbA1c > 6.5%" in abnormal_values:
        logging.info("- Monitor blood sugar levels and follow a diabetic-friendly diet.")
    if "BMI > 30" in abnormal_values:
        logging.info("- Consult a dietitian for a personalized weight-loss plan.")
    if "Blood Pressure > 140/90 mmHg" in abnormal_values:
        logging.info("- Adopt a low-sodium diet and discuss medications with your doctor.")
    if "C-Reactive Protein (CRP) > 3 mg/L" in abnormal_values:
        logging.info("- Investigate underlying causes of inflammation or infection.")

    logging.info("\n--- End of Analysis ---")


def main():
    main_folder = r"C:\Users\manas\OneDrive\Desktop\Medical Test Reports Dataset"
    output_file = r"C:\Users\manas\OneDrive\Desktop\data.csv"
    log_file = r"C:\Users\manas\OneDrive\Desktop\predictions_log.csv"

    # Process reports and train the model
    if os.path.exists(output_file):
        os.remove(output_file)
    df = process_reports(main_folder)
    df.to_csv(output_file, index=False)
    logging.info(f"Data saved to {output_file}")
    model, scaler = train_model(output_file)

    # Handle new patient report
    new_patient_report = input("Enter the path to the new patient report (.docx): ").strip()
    new_patient_data = extract_patient_data_and_ranges(new_patient_report)
    if not new_patient_data:
        logging.error("Failed to extract data from the report.")
        return

    # Convert new patient data to DataFrame
    new_patient_df = pd.DataFrame([new_patient_data])

    # Preprocess the new patient data
    new_patient_scaled = preprocess_new_patient_data(new_patient_df, scaler)

    # Predict the risk level for the new patient
    risk = model.predict(new_patient_scaled)[0]
    risk_level, detected_disease, explanation, abnormal_values = assign_risk_level(new_patient_data)

    # Log health analysis
    logging.info(f"Predicted Risk Level: {risk}")
    logging.info(f"Detected Disease: {detected_disease}")
    logging.info(f"Explanation: {explanation}")
    logging.info(f"Abnormal Values: {abnormal_values}")

    # Generate health recommendations using the new model
    recommendations, score = recommend_health_improvements(risk_level, abnormal_values)
    logging.info(f"Health Recommendations: {recommendations} (Confidence: {score:.2f})")

    # Log patient prediction results
    new_patient_data["Risk Level"] = risk_level
    new_patient_data["Detected Disease"] = detected_disease
    new_patient_data["Explanation"] = explanation
    new_patient_data["Abnormal Values"] = ", ".join(abnormal_values)
    new_patient_data["Recommendations"] = recommendations

    log_df = pd.DataFrame([new_patient_data])
    if not os.path.exists(log_file):
        log_df.to_csv(log_file, index=False)
    else:
        log_df.to_csv(log_file, mode="a", header=False, index=False)

    logging.info(f"Prediction and recommendations logged to {log_file}")

if __name__ == "__main__":
    main()