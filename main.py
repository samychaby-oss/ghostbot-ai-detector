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

# 2. Connexion à la base de données (URL EXTERNE DE FRANCFORT)
DATABASE_URL = "postgresql://amine:3HINAdOayp2boRkjiPLMVxHYaOjSLkjx@dpg-d70gbuv5gffc73ds859g-a.frankfurt-postgres.render.com/projet_s"

engine = create_engine(DATABASE_URL)

# 3. Création automatique de la table au démarrage
def init_db():
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS resultats_detailles (
                    id SERIAL PRIMARY KEY,
                    texte_extrait TEXT,
                    score_ia_num FLOAT,
                    verdict VARCHAR(50),
                    nom_document VARCHAR(255),
                    modele_utilise VARCHAR(100),
                    date_analyse TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))
            print("Base de données initialisée avec succès.")
    except Exception as e:
        print(f"Erreur lors de l'initialisation de la DB : {e}")

# Lancer l'initialisation
init_db()

def load_model():
    try:
        # Assure-toi que ce fichier est bien à la racine de ton dossier GhostBot_Project
        with open('models_bundle.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

@app.get("/")
def home():
    return {"status": "GhostBot API is running", "database": "Connected"}

@app.post("/analyze-file/")
async def analyze_file(file: UploadFile = File(...)):
    pipeline = load_model()
    if not pipeline:
        raise HTTPException(status_code=500, detail="Modèle models_bundle.pkl non trouvé sur le serveur.")
    
    model_name = type(pipeline.named_steps['clf']).__name__
    content = await file.read()
    
    try:
        text_content = content.decode("utf-8")
    except UnicodeDecodeError:
        text_content = content.decode("latin-1")
    
    # Découpage en phrases
    sentences = [s.strip() for s in re.split(r'[.!?\n]+', text_content) if len(s.strip()) > 5]
    
    if not sentences:
        return {"error": "Le fichier est vide ou trop court."}

    results = []

    try:
        with engine.begin() as conn:
            for s in sentences:
                # Prédiction
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
                    "t": clean_sentence, 
                    "s": score_ia, 
                    "v": verdict,
                    "doc": file.filename, 
                    "mod": model_name
                })
                
                results.append({
                    "phrase": clean_sentence[:100] + "...", 
                    "score": f"{score_ia}%", 
                    "verdict": verdict
                })
        
        return {
            "status": "Success",
            "filename": file.filename, 
            "modele": model_name, 
            "nb_phrases": len(sentences),
            "analyse": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur Database: {str(e)}")