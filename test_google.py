import json
import requests


token='ya29.a0AbVbY6OdFxPl6QgQsLfdh6nNyPCgdArXbyNSWOKx4hVsglqpoGQ3gJYEKxoYFJRV_yFPVP-MOzbSyk2NtrUIdwrwqJ9hcOt1Q0YTKDGvLcavM7WU5zSdU-wd-Hi1ZGzShmp7HcpCl40uqyobh6ff2IiGbj9haCgYKAW4SARASFQFWKvPl75clyPpnG11q61nJqjAIaA0163'
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