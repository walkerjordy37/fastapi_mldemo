
import joblib
from utils import extract_features_from_bytes

# Charger le modèle une seule fois
model = joblib.load("fleur_classifier.pkl")

def predict_image(image_bytes):
    features = extract_features_from_bytes(image_bytes).reshape(1, -1)
    prediction = model.predict(features)[0]
    return "fleur" if prediction == 1 else "non-fleur"

