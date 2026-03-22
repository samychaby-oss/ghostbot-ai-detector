import warnings
import os
import sys

# 1. ON FORCE LE SILENCE AVANT TOUT LE RESTE
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report

def train_ghostbot_hetic():
    # Bloquer la sortie d'erreur de joblib pour éviter les textes roses
    stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')

    try:
        df = pd.read_csv('data_ghostbot.csv')
        taille_sample = min(10000, len(df))
        df = df.sample(n=taille_sample, random_state=42)
        
        # On remet la sortie erreur juste pour ce message
        sys.stderr = stderr
        print(f"--- {len(df)} phrases chargées ---")
        sys.stderr = open(os.devnull, 'w')
    except Exception:
        sys.stderr = stderr
        print("Erreur : data_ghostbot.csv introuvable.")
        return

    X = df['text'].astype('U')
    y = df['generated']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ('vect', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression())
    ])

    param_grid = [
        {'clf': [LogisticRegression()], 'clf__C': [0.1, 1, 10], 'vect__ngram_range': [(1, 1), (1, 2)]},
        {'clf': [RandomForestClassifier()], 'clf__n_estimators': [50], 'clf__max_depth': [10]},
        {'clf': [SVC(probability=True)], 'clf__C': [1], 'clf__kernel': ['linear']},
        {'clf': [MultinomialNB()], 'clf__alpha': [0.1, 1.0], 'vect__ngram_range': [(1, 1), (1, 2)]}
    ]

    print("--- Lancement du bouclage (GridSearch) ---")
    # n_jobs=1 au lieu de -1 pour stopper les erreurs roses de joblib
    grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=1, verbose=0)
    grid.fit(X_train, y_train)

    # ON RÉACTIVE LA CONSOLE PROPRE
    sys.stderr = stderr

    champion = grid.best_estimator_
    print(f"\n🏆 LE MEILLEUR MODÈLE : {grid.best_params_['clf']}")
    print(f"📊 Score : {grid.best_score_:.2%}")

    y_pred = champion.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n--- MATRICE DE CONFUSION ---")
    print(f"Vrais Humains : {cm[0][0]} | Faux Humains : {cm[1][0]}")
    print(f"Vraies IA : {cm[1][1]} | Fausses IA : {cm[0][1]}")

    with open('models_bundle.pkl', 'wb') as f:
        pickle.dump(champion, f)
    
    print("\n✅ Terminé sans erreurs !")

if __name__ == "__main__":
    train_ghostbot_hetic()



