from fastapi import FastAPI
from pydantic import BaseModel

import pickle
from pickle import dump

app = FastAPI()

# Desempacar el modelo que está en formato pkl
archivo = "ClasificadorTweets.pkl"
# Se guardó el modelo entrenado y los vectores asignados a las palabras
clasificador, Tfidf_vectores = pickle.load(open(archivo, "rb"))

@app.get("/")
async def root():
    return {"message": "Hello World"}

# Declaramos nuestra clase Tweet que hereda de BaseModel
# Este tendrá un atributo que es el contenido del tweet (un string)
class Tweet(BaseModel):
    text: str

@app.post("/clasificar_tweet/")
async def classificar_tweet(tweet: Tweet):

    # Usamos el modelo para clasificar el contenido del tweet
    
    # Vectorizamos las palabras del tweet con tfidf
    vectorizado = Tfidf_vectores.transform([tweet.text])
    # Hacemos la predicción con el modelo
    prediccion = clasificador.predict_proba(vectorizado)

    # la prediccion es un arreglo que contiene porcentajes de la postura
    # [[porcentaje_liberal, porcentaje_conservador]]

    p_liberal = round(prediccion[0][0], 4)
    p_conservador = round(prediccion[0][1], 4)

    print(f"Su tweet es {p_liberal}% liberal y {p_conservador}% conservador")

    return {
            "mensaje": tweet.text,
            "liberal": p_liberal,
            "conservador": p_conservador
           }