import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from flask import Flask, request, render_template
from markupsafe import Markup
import os

# Example: Handling CSV upload from an HTML page using Flask

app = Flask(__name__)

# Load model and scaler for use in Flask app
model = joblib.load(r'C:\dev\heart_disease\heart_disease_model.pkl')
scaler = joblib.load(r'C:\dev\heart_disease\scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    predictions = None
    accuracy = None
    report = None
    error = None
    table_html = None
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                error = "No file part"
            else:
                file = request.files['file']
                if file.filename == '':
                    error = "No selected file"
                else:
                    filepath = os.path.join('uploads', file.filename)
                    file.save(filepath)
                    uploaded_data = pd.read_csv(filepath)

                    # If 'target' column exists, use it for evaluation, but drop it for prediction
                    if 'target' in uploaded_data.columns:
                        X = uploaded_data.drop('target', axis=1)
                        y_true = uploaded_data['target']
                        X_scaled = scaler.transform(X)
                        predictions = model.predict(X_scaled).tolist()
                        accuracy = accuracy_score(y_true, predictions)
                        report = classification_report(y_true, predictions)
                    else:
                        X_scaled = scaler.transform(uploaded_data)
                        predictions = model.predict(X_scaled)
                    
                    uploaded_data['Prediction'] = predictions
                    table_html = Markup(uploaded_data.to_html(classes='table table-bordered', index=False))
                    
        except Exception as e:
            required_cols = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']

            error = (f"Please ensure the CSV file is formatted correctly with the following columns:\n\n"
                        + ", ".join(f"'{col}'" for col in required_cols))

    return render_template('index.html', predictions=predictions, accuracy=accuracy, report=report, error=error, table_html=table_html)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)

# Function to create and train the model

def train_and_save_model():
    # Load the dataset
    data = pd.read_csv('your_training_data.csv') # adjust path as needed

    # Features and target
    x = data.drop('target', axis=1)
    y = data['target']

    # Split into train and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Create and train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(x_train, y_train)

    # Make predictions
    y_pred = model.predict(x_test)
        
    # Evaluate performance
    # print("Accuracy:", accuracy_score(y_test, y_pred))
    # print("Classification Report:\n", classification_report(y_test, y_pred))

    joblib.dump(model, r'C:\dev\heart_disease\heart_disease_model.pkl')
    joblib.dump(scaler, r'C:\dev\heart_disease\scaler.pkl')
    return model, scaler