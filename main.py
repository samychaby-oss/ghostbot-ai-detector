import warnings
import os
import re
import pickle
from fastapi import FastAPI, UploadFile, File, HTTPException
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# 1. Configuration initiale
load_dotenv()
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

app = FastAPI()

# On écrit l'adresse en dur pour être sûr que ça marche
DATABASE_URL = "postgresql://ton_utilisateur:ton_mot_de_passe@dpg-xxx-a.oregon-postgres.render.com/projet_s"

if DATABASE_URL:
    # Render donne souvent postgres://, mais SQLAlchemy a besoin de postgresql://
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
else:
    # Version de secours si tu testes sur ton PC en local
    DATABASE_URL = "postgresql://postgres:ton_pass@localhost:5432/projet_s"

engine = create_engine(DATABASE_URL)

def load_model():
    try:
        with open('models_bundle.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

@app.post("/analyze-file/")
async def analyze_file(file: UploadFile = File(...)):
    pipeline = load_model()
    if not pipeline:
        raise HTTPException(status_code=500, detail="Modèle models_bundle.pkl non trouvé.")
    
    model_name = type(pipeline.named_steps['clf']).__name__
    content = await file.read()
    
    try:
        text_content = content.decode("utf-8")
    except UnicodeDecodeError:
        text_content = content.decode("latin-1")
    
    # Découpage en phrases
    sentences = [s.strip() for s in re.split(r'[.!?\n]+', text_content) if len(s.strip()) > 2]
    results = []

    try:
        with engine.begin() as conn:
            for s in sentences:
                # Prediction
                probs = pipeline.predict_proba([s])[0]
                score_ia = round(float(probs[1] * 100), 2)
                verdict = "IA" if score_ia > 30 else "Humain"
                
                # Nettoyage pour SQL
                clean_sentence = s.replace('\x00', '') 

                # Insertion SQL
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
        # Si ça plante ici, c'est l'URL de la DB qui est mauvaise
        raise HTTPException(status_code=500, detail=f"Erreur Database: {str(e)}")