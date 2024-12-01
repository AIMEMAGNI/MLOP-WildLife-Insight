from io import StringIO

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from tensorflow.keras.models import load_model

# Load the pre-trained model and encoders
model = load_model("model1_simple.h5")
label_encoders = joblib.load("label_encoders.pkl")
scaler = joblib.load("scaler.pkl")

# Initialize FastAPI app
app = FastAPI()

# Welcoming message for the API


@app.get("/")
async def root():
    return {"message": "Welcome to the Prediction API! Use /predict for predictions and /upload_data to retrain the model."}

# Define input schema for prediction


class PredictionInput(BaseModel):
    speciesName: str
    systems: str
    scopes: str

# Prediction endpoint


@app.post("/predict")
async def predict(data: PredictionInput):
    try:
        # Encode input data
        species_encoded = label_encoders["speciesName"].transform([data.speciesName])[
            0]
        systems_encoded = label_encoders["systems"].transform([data.systems])[
            0]
        scopes_encoded = label_encoders["scopes"].transform([data.scopes])[0]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Encoding error: {e}")

    # Prepare input for prediction
    input_data = np.array([[species_encoded, systems_encoded, scopes_encoded]])
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_category = label_encoders["Category"].inverse_transform([
                                                                      predicted_class])[0]

    return {
        "predicted_category": predicted_category,
        "confidence_scores": prediction.tolist()
    }

# Endpoint to upload new data and trigger model retraining


@app.post("/upload_data")
async def upload_data(file: UploadFile = File(...)):
    # Load the uploaded CSV data into a DataFrame
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading file: {e}")

    # Ensure the CSV file contains exactly the required columns
    required_columns = ["speciesName", "systems", "scopes"]
    if list(df.columns) != required_columns:
        raise HTTPException(
            status_code=400, detail="CSV must contain exactly 'speciesName', 'systems', and 'scopes' columns.")

    # Trigger retraining (example: retrain the model with the new data)
    # This step should be adjusted to your model retraining pipeline
    try:
        retrained_model = retrain_model(df)
        retrained_model.save("model1_retrained.h5")  # Save the new model
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Model retraining failed: {e}")

    return {"message": "Model retrained successfully with uploaded data!"}

# Model retraining function (Placeholder - Implement your retraining pipeline)


def retrain_model(df):
    # Assuming `df` contains the necessary data for retraining
    # You would split the data, preprocess it, and train the model again
    X = df.drop(columns=["Category"])  # Example feature columns
    y = df["Category"]  # Target column

    # Apply necessary preprocessing steps (e.g., scaling, encoding)
    X_scaled = scaler.transform(X)

    # Train a new model (Example: simple model retraining)
    # Replace this with your actual model retraining pipeline
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential

    retrained_model = Sequential([
        Dense(64, activation='relu', input_dim=X_scaled.shape[1]),
        Dense(32, activation='relu'),
        Dense(len(label_encoders["Category"].classes_), activation='softmax')
    ])
    retrained_model.compile(
        loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    retrained_model.fit(X_scaled, y, epochs=10, batch_size=32)

    return retrained_model
