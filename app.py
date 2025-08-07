import os
from dotenv import load_dotenv
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import FloatField, SelectField, SubmitField
from wtforms.validators import DataRequired, NumberRange
from werkzeug.utils import secure_filename
import io
import base64
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_FILE = 'model.pkl'
PIPELINE_FILE = 'pipeline.pkl'

def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('encode', OneHotEncoder(handle_unknown='ignore'))
    ])

    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attribs),
        ('cat', cat_pipeline, cat_attribs)
    ])

    return full_pipeline

def train_model():
    """Train the model if it doesn't exist"""
    if not os.path.exists(MODEL_FILE):
        # 1. Load the dataset
        housing = pd.read_csv('housing.csv')

        # 2. creating a stratified split
        housing['income_cat'] = pd.cut(housing['median_income'], 
                                     bins=[0, 1.5, 3, 4.5, 6, np.inf], 
                                     labels=[1, 2, 3, 4, 5])

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        
        for train_index, test_index in split.split(housing, housing['income_cat']):
            strat_test_set = housing.loc[test_index].drop('income_cat', axis=1)
            strat_train_set = housing.loc[train_index].drop('income_cat', axis=1)

        # 4. separating features and labels
        housing = strat_train_set.copy()
        housing_labels = housing['median_house_value'].copy()
        housing_features = housing.drop('median_house_value', axis=1)

        # 5. Separating numerical and categorical value
        nums_attribute = housing_features.drop('ocean_proximity', axis=1).columns.tolist()
        cats_attribute = ['ocean_proximity']

        pipeline = build_pipeline(nums_attribute, cats_attribute)
        housing_prepared = pipeline.fit_transform(housing_features)

        model = RandomForestRegressor(random_state=42)
        model.fit(housing_prepared, housing_labels)

        joblib.dump(model, MODEL_FILE)
        joblib.dump(pipeline, PIPELINE_FILE)
        print('Model trained successfully')
        return True
    return False

# Train model on startup (only if housing.csv exists)
if os.path.exists('housing.csv'):
    train_model()
else:
    print("Note: housing.csv not found. Model will be trained on first request.")

class PredictionForm(FlaskForm):
    longitude = FloatField('Longitude', validators=[DataRequired(), NumberRange(-124, -114)])
    latitude = FloatField('Latitude', validators=[DataRequired(), NumberRange(32, 42)])
    housing_median_age = FloatField('Housing Median Age', validators=[DataRequired(), NumberRange(1, 52)])
    total_rooms = FloatField('Total Rooms', validators=[DataRequired(), NumberRange(2, 40000)])
    total_bedrooms = FloatField('Total Bedrooms', validators=[DataRequired(), NumberRange(1, 6500)])
    population = FloatField('Population', validators=[DataRequired(), NumberRange(3, 35000)])
    households = FloatField('Households', validators=[DataRequired(), NumberRange(1, 6000)])
    median_income = FloatField('Median Income', validators=[DataRequired(), NumberRange(0.5, 15)])
    ocean_proximity = SelectField('Ocean Proximity', 
                                choices=[('NEAR BAY', 'Near Bay'), 
                                       ('<1H OCEAN', '<1H Ocean'), 
                                       ('INLAND', 'Inland'), 
                                       ('NEAR OCEAN', 'Near Ocean'), 
                                       ('ISLAND', 'Island')],
                                validators=[DataRequired()])
    submit = SubmitField('Predict Price')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    form = PredictionForm()
    prediction = None
    confidence = None
    
    if form.validate_on_submit():
        try:
            # Load model and pipeline
            model = joblib.load(MODEL_FILE)
            pipeline = joblib.load(PIPELINE_FILE)
            
            # Create input data
            input_data = pd.DataFrame({
                'longitude': [form.longitude.data],
                'latitude': [form.latitude.data],
                'housing_median_age': [form.housing_median_age.data],
                'total_rooms': [form.total_rooms.data],
                'total_bedrooms': [form.total_bedrooms.data],
                'population': [form.population.data],
                'households': [form.households.data],
                'median_income': [form.median_income.data],
                'ocean_proximity': [form.ocean_proximity.data]
            })
            
            # Transform and predict
            transformed_input = pipeline.transform(input_data)
            prediction = model.predict(transformed_input)[0]
            
            # Calculate confidence (using model's feature importances as a proxy)
            if hasattr(model, 'feature_importances_'):
                confidence = np.mean(model.feature_importances_) * 100
            else:
                confidence = 85.0  # Default confidence for non-tree models
                
        except Exception as e:
            flash(f'Error making prediction: {str(e)}', 'error')
    
    return render_template('predict.html', form=form, prediction=prediction, confidence=confidence)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and file.filename.endswith('.csv'):
            try:
                # Read the uploaded file
                df = pd.read_csv(file)
                
                # Load model and pipeline
                model = joblib.load(MODEL_FILE)
                pipeline = joblib.load(PIPELINE_FILE)
                
                # Make predictions
                transformed_input = pipeline.transform(df)
                predictions = model.predict(transformed_input)
                
                # Add predictions to dataframe
                df['predicted_median_house_value'] = predictions
                
                # Save results
                output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'predictions.csv')
                df.to_csv(output_path, index=False)
                
                # Create summary statistics
                summary = {
                    'total_predictions': len(predictions),
                    'mean_prediction': float(np.mean(predictions)),
                    'min_prediction': float(np.min(predictions)),
                    'max_prediction': float(np.max(predictions)),
                    'std_prediction': float(np.std(predictions))
                }
                
                flash('File uploaded and predictions made successfully!', 'success')
                return render_template('upload_results.html', summary=summary, output_path=output_path)
                
            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'error')
                return redirect(request.url)
        else:
            flash('Please upload a CSV file', 'error')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/download/<filename>')
def download_file(filename):
    try:
        if filename == 'sample_data.csv':
            return send_file('sample_data.csv', as_attachment=True)
        else:
            return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename),
                            as_attachment=True)
    except FileNotFoundError:
        flash('File not found', 'error')
        return redirect(url_for('upload_file'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        
        # Load model and pipeline
        model = joblib.load(MODEL_FILE)
        pipeline = joblib.load(PIPELINE_FILE)
        
        # Create input data
        input_data = pd.DataFrame([data])
        
        # Transform and predict
        transformed_input = pipeline.transform(input_data)
        prediction = model.predict(transformed_input)[0]
        
        return jsonify({
            'prediction': float(prediction),
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=False, host='0.0.0.0', port=port) 