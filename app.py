from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib
import cv2
import numpy as np

app = FastAPI()

#  Autoriser les requêtes depuis ton frontend React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = joblib.load("fleur_classifier.pkl")  


# Route racine (accueil)
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head><title>API Fleur</title></head>
        <body>
            <h1>Bienvenue sur l'API de classification 🌸</h1>
            <p>Utilisez <code>/predict</code> pour envoyer une image.</p>
            <p>Accédez à la documentation : <a href="/docs">/docs</a></p>
        </body>
    </html>
    """

#  Endpoint de prédiction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        features = extract_features_from_bytes(image_bytes).reshape(1, -1)
        prediction = model.predict(features)[0]
        label = "fleur" if prediction == 1 else "non-fleur"
        return JSONResponse(content={"prediction": label})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
