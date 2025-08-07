# California House Pricing Predictor

A Flask-based web application that predicts house prices in California using machine learning. The application uses a Random Forest regression model trained on the California Housing Dataset.

## Features

- **Single Prediction**: Enter individual property details to get instant price predictions
- **Batch Prediction**: Upload CSV files with multiple properties for bulk predictions
- **Modern UI**: Beautiful, responsive web interface built with Bootstrap 5
- **API Endpoint**: RESTful API for programmatic access
- **Download Results**: Export prediction results as CSV files
- **Real-time Validation**: Form validation with helpful error messages

## Technology Stack

- **Backend**: Flask (Python)
- **Machine Learning**: Scikit-learn (Random Forest Regressor)
- **Data Processing**: Pandas & NumPy
- **Frontend**: Bootstrap 5, Font Awesome
- **Model Storage**: Joblib
- **Form Handling**: Flask-WTF

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd california-house-pricing
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have the `housing.csv` file in the project directory.

## Usage

### Running the Application

1. Start the Flask development server:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

### Single Prediction

1. Go to the "Predict" page
2. Fill in the property details:
   - Longitude (-124.35 to -114.31)
   - Latitude (32.54 to 41.95)
   - Housing Median Age (1 to 52 years)
   - Total Rooms (2 to 39,320)
   - Total Bedrooms (1 to 6,435)
   - Population (3 to 35,682)
   - Households (1 to 6,082)
   - Median Income (0.5 to 15.0)
   - Ocean Proximity (NEAR BAY, <1H OCEAN, INLAND, NEAR OCEAN, ISLAND)
3. Click "Predict Price" to get instant results

### Batch Prediction

1. Prepare a CSV file with the required columns (see sample format below)
2. Go to the "Upload" page
3. Upload your CSV file
4. Wait for processing to complete
5. Download the results with predictions

### CSV Format

Your CSV file must include these columns:
```csv
longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income,ocean_proximity
-122.23,37.88,41,880,129,322,126,8.3252,NEAR BAY
```

Download the sample CSV file from the upload page to see the exact format.

### API Usage

Make POST requests to `/api/predict` with JSON data:

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41,
    "total_rooms": 880,
    "total_bedrooms": 129,
    "population": 322,
    "households": 126,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY"
  }'
```

Response:
```json
{
  "prediction": 452600.0,
  "status": "success"
}
```

## Machine Learning Model

- **Algorithm**: Random Forest Regressor
- **Training Data**: California Housing Dataset
- **Features**: 9 property characteristics
- **Preprocessing**: Automated pipeline with imputation and scaling
- **Validation**: Stratified cross-validation

### Features Used

1. **Location Features**:
   - Longitude & Latitude
   - Ocean Proximity

2. **Property Features**:
   - Housing Median Age
   - Total Rooms
   - Total Bedrooms

3. **Demographic Features**:
   - Population
   - Households
   - Median Income

## Project Structure

```
california-house-pricing/
├── app.py                 # Main Flask application
├── main.py               # Original script (for reference)
├── housing.csv           # Training dataset
├── requirements.txt      # Python dependencies
├── sample_data.csv      # Sample CSV for batch predictions
├── templates/           # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   ├── upload.html
│   ├── upload_results.html
│   └── about.html
└── uploads/             # Directory for uploaded files
```

## Model Files

The application automatically generates these files on first run:
- `model.pkl`: Trained Random Forest model
- `pipeline.pkl`: Data preprocessing pipeline

## Important Notes

- Predictions are based on historical California housing data
- Results should be used as estimates, not exact values
- Market conditions may affect actual prices
- For CSV uploads, ensure all required columns are present
- Ocean proximity values must match the specified options

## Development

To run in development mode with auto-reload:
```bash
export FLASK_ENV=development
python app.py
```

## License

This project is for educational purposes. The California Housing Dataset is publicly available.

## Contributing

Feel free to submit issues and enhancement requests!
