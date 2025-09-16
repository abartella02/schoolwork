import json

username = "alexis"
passEnter = 'bartellucho'

with open("userpass.json", "r") as f:
    data = json.load(f)
    data.append({'username':username, 'password':passEnter})
    '''
    for index, object in enumerate(data):
        if object['username'] == 'alex':
            data.pop(index)
    '''

with open("userpass.json", "w") as f:
    f.write(json.dumps(data, indent=2))


