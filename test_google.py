import json
import requests

token= 'ya29.a0AbVbY6NhWtoOHNDXKpRSOZhAVEXeeUJ4ot9lTUri8EQJiKMtGcKOALWVTC0IKaJ9h6Q139ttH8k3DE1sPtSuvPkukgT_v906fPcpAqBss1EBIbvcdfB95UqvFJiu234aov6BQZA30KRyet-4a005uGBOBUNyaCgYKAWMSARASFQFWKvPlD5PNmWckrSqHuLbXvUFLIw0163'
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