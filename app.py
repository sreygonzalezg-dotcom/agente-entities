from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from textblob import TextBlob
from typing import List
import uvicorn

app = FastAPI()

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
        blob = TextBlob(text)
        # Extraer frases nominales (potenciales entidades)
        for np in blob.noun_phrases:
            # Clasificación simple por palabras clave
            label = "unknown"
            if np.lower() in ["candidato a", "carlos", "a"]:
                label = "candidate A"
            elif np.lower() in ["candidato b", "maria", "b"]:
                label = "candidate B"
            elif any(word in np.lower() for word in ["calle", "barrio", "ciudad", "parque"]):
                label = "place"
            elif any(word in np.lower() for word in ["empresa", "gobierno", "municipio"]):
                label = "organization"
            else:
                label = "other"
            
            key = f"{np}|{label}"
            entity_counts[key] = entity_counts.get(key, 0) + 1
    
    entities = []
    for key, count in entity_counts.items():
        text_ent, label = key.split("|")
        entities.append({
            "text": text_ent,
            "label": label,
            "mentions": count
        })
    
    entities.sort(key=lambda x: x["mentions"], reverse=True)
    
    return {
        "status": "ok",
        "entities": entities[:50]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
