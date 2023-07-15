import json
import requests


token='ya29.a0AbVbY6PKpLRTGOBrrLD9a65oAnjgFzav4-4tyr-UAs3xnQrAHm5CUfVjHf_zjJ5m3MI8VrZvPNxsD6N5TTE-_02VMbRr7oDZtUnZYkFOfAiBOYiRSRjivE-utSH4FpRCpj_xZ9Es2xccNr5w02BwEr-6ZFkBaCgYKAUgSARASFQFWKvPlMICO2Xi_sqnzZ2J4WJQ5ig0163'
headers = {
    "Authorization": f"Bearer {token}"
}

para = {
    "name": "check.rar",
    #"parents": [""]
}

files = {
    'data': ('metadata', json.dumps(para), 'application/json;charset=UTF-8'),#charset=UTF-8
    'file': open('./check.rar', 'rb')
}

r = requests.post("https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
                  headers=headers,
                  files=files
                  )

print(r.text)