import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, render_template, request, jsonify
from joblib import dump, load

app = Flask(__name__)

# Load and preprocess the data
data = pd.read_csv('used_cars.csv')

# Convert price to numeric, removing "$" and "," characters
data['price'] = data['price'].str.replace('$', '').str.replace(',', '').astype(float)

# Convert mileage to numeric, removing "mi." and "," characters
data['milage'] = data['milage'].str.replace(' mi.', '').str.replace(',', '').astype(float)

# Handle NaN values in fuel_type
data['fuel_type'] = data['fuel_type'].fillna('Unknown')

# Prepare features and target
X = data[['brand', 'model', 'model_year', 'milage', 'fuel_type']]
y = data['price']

# Perform one-hot encoding for categorical variables
X = pd.get_dummies(X, columns=['brand', 'model', 'fuel_type'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save the model and scaler
dump(model, 'random_forest_model.joblib')
dump(scaler, 'scaler.joblib')

# Load the model and scaler (in a real-world scenario, you'd do this instead of training every time)
model = load('random_forest_model.joblib')
scaler = load('scaler.joblib')

# Prepare lists for the dropdown menus
companies = sorted(data['brand'].unique())
car_models = {company: sorted(data[data['brand'] == company]['model'].unique()) for company in companies}
years = sorted(data['model_year'].unique(), reverse=True)
fuel_types = sorted(data['fuel_type'].unique())

@app.route('/')
def home():
    return render_template('index.html', companies=companies, car_models=car_models, years=years, fuel_types=fuel_types)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            brand = request.form['company']
            model_name = request.form['car_models']
            year = int(request.form['year'])
            fuel_type = request.form['fuel_type']
            kms_driven = float(request.form['kilo_driven'])

            # Prepare the input data
            input_data = pd.DataFrame({
                'brand': [brand],
                'model': [model_name],
                'model_year': [year],
                'milage': [kms_driven],
                'fuel_type': [fuel_type]
            })

            # Perform one-hot encoding
            input_encoded = pd.get_dummies(input_data, columns=['brand', 'model', 'fuel_type'])

            # Create a DataFrame with all columns from X, initialized with zeros
            full_columns = pd.DataFrame(0, index=input_encoded.index, columns=X.columns)

            # Update the values for the columns that exist in input_encoded
            for col in input_encoded.columns:
                if col in full_columns.columns:
                    full_columns[col] = input_encoded[col]

            # Scale the input data
            input_scaled = scaler.transform(full_columns)

            # Make prediction
            prediction = model.predict(input_scaled)

            return jsonify({'prediction': f'${prediction[0]:,.2f}'})
        except Exception as e:
            app.logger.error(f"An error occurred: {str(e)}")
            return jsonify({'error': 'An error occurred during prediction. Please try again.'}), 500

    return jsonify({'error': 'Invalid request method'}), 400

@app.route('/get_models/<company>')
def get_models(company):
    return jsonify(sorted(car_models.get(company, [])))

if __name__ == '__main__':
    app.run(debug=True)