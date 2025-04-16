import requests

url = "https://ws.smn.gob.ar/map_items/weather/rosario"
response = requests.get(url)

print("Código de estado:", response.status_code)
print("Texto de respuesta:", response.text)
