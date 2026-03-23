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

# 2. Connexion à ta base de données (Francfort)
DATABASE_URL = "postgresql://amine:3HINAdOayp2boRkjiPLMVxHYaOjSLkjx@dpg-d70gbuv5gffc73ds859g-a.frankfurt-postgres.render.com/projet_s"
engine = create_engine(DATABASE_URL)

# 3. Chargement du modèle IA
def load_model():
    try:
        with open('models_bundle.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

@app.get("/")
def home():
    return {"status": "GhostBot Live", "mode": "Single Execution Save"}

@app.post("/analyze-file/")
async def analyze_file(file: UploadFile = File(...)):
    pipeline = load_model()
    if not pipeline:
        raise HTTPException(status_code=500, detail="Fichier models_bundle.pkl introuvable.")
    
    model_name = type(pipeline.named_steps['clf']).__name__
    content = await file.read()
    
    try:
        text_content = content.decode("utf-8")
    except UnicodeDecodeError:
        text_content = content.decode("latin-1")
    
    sentences = [s.strip() for s in re.split(r'[.!?\n]+', text_content) if len(s.strip()) > 5]
    
    if not sentences:
        return {"error": "Texte trop court."}

    results_list = []

    try:
        with engine.begin() as conn:
            # --- ÉTAPE CRUCIALE : ON VIDE LA TABLE AVANT D'ÉCRIRE ---
            # Cela permet de ne garder QUE l'exécution présente
            conn.execute(text("TRUNCATE TABLE resultats_detailles;"))
            
            for s in sentences:
                probs = pipeline.predict_proba([s])[0]
                score_ia = round(float(probs[1] * 100), 2)
                verdict = "IA" if score_ia > 40 else "Humain"
                clean_text = s.replace('\x00', '')

                query = text("""
                    INSERT INTO resultats_detailles 
                    (texte_extrait, score_ia_num, verdict, nom_document, modele_utilise) 
                    VALUES (:t, :s, :v, :doc, :mod)
                """)
                
                conn.execute(query, {
                    "t": clean_text, "s": score_ia, "v": verdict,
                    "doc": file.filename, "mod": model_name
                })
                
                results_list.append({
                    "phrase": clean_text[:100] + "...", 
                    "score": f"{score_ia}%", 
                    "verdict": verdict
                })
        
        return {
            "status": "Success - Table Refreshed",
            "document": file.filename, 
            "model": model_name, 
            "total_phrases": len(results_list),
            "analyses": results_list
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur SQL : {str(e)}")