from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import spacy
from typing import List, Optional
import uvicorn

app = FastAPI()

# Cargar modelo de español de spaCy (se descargará en el Dockerfile)
try:
    nlp = spacy.load("es_core_news_sm")
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "es_core_news_sm"])
    nlp = spacy.load("es_core_news_sm")

class EntityRequest(BaseModel):
    texts: List[str]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/")
async def extract_entities(req: EntityRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    entity_counts = {}
    
    for text in req.texts:
        doc = nlp(text)
        for ent in doc.ents:
            label = ent.label_
            text_ent = ent.text
            key = f"{text_ent}|{label}"
            entity_counts[key] = entity_counts.get(key, 0) + 1
    
    entities = []
    for key, count in entity_counts.items():
        text_ent, label = key.split("|")
        
        # Mapeo de etiquetas de spaCy a categorías legibles
        label_map = {
            "PER": "persona",
            "LOC": "lugar",
            "ORG": "organizacion",
            "MISC": "otro"
        }
        
        entities.append({
            "text": text_ent,
            "label": label_map.get(label, label),
            "mentions": count
        })
    
    # Ordenar por número de menciones
    entities.sort(key=lambda x: x["mentions"], reverse=True)
    
    return {
        "status": "ok",
        "entities": entities[:50]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
