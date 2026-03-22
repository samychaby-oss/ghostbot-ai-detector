import os
from sqlalchemy import create_engine

# Render donnera cette adresse automatiquement via l'onglet Environment
DATABASE_URL = os.getenv("DATABASE_URL") 
engine = create_engine(DATABASE_URL)