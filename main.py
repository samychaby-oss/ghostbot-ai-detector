import warnings
import os
import re
import pickle
from fastapi import FastAPI, UploadFile, File, HTTPException
from sqlalchemy import create_engine, text
from dotenv import load_dotenv # <--- Ajoute ça

# 1. Charger les variables d'environnement
load_dotenv()

warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

app = FastAPI()

# 2. Construction dynamique de l'URL de connexion
# Si os.getenv ne trouve rien, il utilisera la valeur par défaut après la virgule
USER = os.getenv("DB_USER", "postgres")
PASSWORD = os.getenv("DB_PASSWORD", "")
HOST = os.getenv("DB_HOST", "localhost")
PORT = os.getenv("DB_PORT", "5432")
NAME = os.getenv("DB_NAME", "projet_s")

DATABASE_URL = f"postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{NAME}"
engine = create_engine(DATABASE_URL)

def load_model():
    # En entreprise, ici on ajouterait une logique pour vérifier si le modèle 
    # doit être téléchargé depuis le Cloud ou lu localement
    try:
        with open('models_bundle.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

@app.post("/analyze-file/")
async def analyze_file(file: UploadFile = File(...)):
    pipeline = load_model()
    if not pipeline:
        raise HTTPException(status_code=500, detail="Modèle non trouvé.")
    
    model_name = type(pipeline.named_steps['clf']).__name__
    content = await file.read()
    
    # ... (Le reste de ton code reste exactement le même)
    try:
        text_content = content.decode("utf-8")
    except UnicodeDecodeError:
        text_content = content.decode("latin-1")
    
    sentences = [s.strip() for s in re.split(r'[.!?\n]+', text_content) if len(s.strip()) > 2]
    results = []

    try:
        with engine.begin() as conn:
            for s in sentences:
                probs = pipeline.predict_proba([s])[0]
                score_ia = round(float(probs[1] * 100), 2)
                verdict = "IA" if score_ia > 30 else "Humain"
                clean_sentence = s.replace('\x00', '') 

                query = text("""
                    INSERT INTO resultats_detailles 
                    (texte_extrait, score_ia_num, verdict, nom_document, modele_utilise) 
                    VALUES (:t, :s, :v, :doc, :mod)
                """)
                
                conn.execute(query, {
                    "t": clean_sentence, "s": score_ia, "v": verdict,
                    "doc": file.filename, "mod": model_name
                })
                
                results.append({"phrase": clean_sentence, "score": f"{score_ia}%", "verdict": verdict})
        
        return {"filename": file.filename, "modele": model_name, "analyse": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))