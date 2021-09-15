import requests

prueba = {"text": "Loret De Mola"}
resp = requests.post("http://localhost:8000/clasificar_tweet/", json=prueba)

print(resp.content)