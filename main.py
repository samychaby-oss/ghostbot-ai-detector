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
        # Assure-toi que ce fichier est bien à la racine de ton projet GitHub
        with open('models_bundle.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

@app.get("/")
def home():
    return {"status": "GhostBot Live", "database": "Connected to resultats_detailles"}

@app.post("/analyze-file/")
async def analyze_file(file: UploadFile = File(...)):
    # Réveil du modèle
    pipeline = load_model()
    if not pipeline:
        raise HTTPException(status_code=500, detail="Fichier models_bundle.pkl introuvable sur le serveur.")
    
    # Identification du modèle (ex: MultinomialNB ou GradientBoosting)
    model_name = type(pipeline.named_steps['clf']).__name__
    
    # Lecture du contenu du fichier
    content = await file.read()
    try:
        text_content = content.decode("utf-8")
    except UnicodeDecodeError:
        text_content = content.decode("latin-1")
    
    # Découpage du texte en phrases propres
    sentences = [s.strip() for s in re.split(r'[.!?\n]+', text_content) if len(s.strip()) > 5]
    
    if not sentences:
        return {"error": "Le texte est vide ou trop court."}

    results_list = []

    try:
        # On ouvre la connexion pour enregistrer dans TA table pgAdmin
        with engine.begin() as conn:
            for s in sentences:
                # Calcul de la probabilité IA
                probs = pipeline.predict_proba([s])[0]
                score_ia = round(float(probs[1] * 100), 2)
                
                # Verdict basé sur ton seuil
                verdict = "IA" if score_ia > 40 else "Humain"
                
                # Nettoyage anti-bug SQL
                clean_text = s.replace('\x00', '')

                # INSERTION DANS TA TABLE : resultats_detailles
                query = text("""
                    INSERT INTO resultats_detailles 
                    (texte_extrait, score_ia_num, verdict, nom_document, modele_utilise) 
                    VALUES (:t, :s, :v, :doc, :mod)
                """)
                
                conn.execute(query, {
                    "t": clean_text, 
                    "s": score_ia, 
                    "v": verdict,
                    "doc": file.filename, 
                    "mod": model_name
                })
                
                # Préparation de l'affichage pour Swagger
                results_list.append({
                    "phrase": clean_text[:100] + "...", 
                    "score": f"{score_ia}%", 
                    "verdict": verdict
                })
        
        # Réponse finale envoyée à l'écran
        return {
            "status": "Success",
            "document": file.filename, 
            "model_used": model_name, 
            "total_phrases": len(results_list),
            "analyses": results_list
        }

    except Exception as e:
        # En cas d'erreur avec la base pgAdmin
        raise HTTPException(status_code=500, detail=f"Erreur SQL : {str(e)}")