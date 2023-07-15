import os


token='ya29.a0AbVbY6P4kzLg9uYwMnl70L1zvfplQ4cQ5SQUr6LjuxOMO-SeIWHbRLBIKiRYigW6XGK3zl8TkiFl5r44VJqi5Udak5JILX9nPMQbUiy-JoI83EBPkwAO8Fz8EuIwEeKxR9nQCIux4vA7bWmSKJ8YfaWQftZfaCgYKAV4SARASFQFWKvPlEJaH_NjJndF2Ur6IbrPcpw0163'
fileid='1GDU_OyCAiGLCOJkO13xO4k52qOVmVAnR'
filename='./test.rar'
cmd=f'wget -H "Authorization: Bearer {token}" https://www.googleapis.com/drive/v3/files/{fileid}?alt=media -o {filename} '

os.system(cmd)