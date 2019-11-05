import requests

def checkPlate(plate, source):
    payload = { 'license_plate': plate }  
    r = requests.get('http://localhost:3000/api/hasUser', params=payload).json()

    if r.get('hasUser'):
        print("[Detection type: " + source + "] User exists in database.")
    else:
        print("[Detection type: " + source + "] User does not exist in database.")
