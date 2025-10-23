
import cv2
import numpy as np

def extract_features_from_bytes(image_bytes):
    # Lire l'image depuis les bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image non lisible")

    img = cv2.resize(img, (100, 100))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    texture = [gray.mean(), gray.std()]
    shape_ratio = img.shape[0] / img.shape[1]

    return np.hstack([hist, texture, shape_ratio])

