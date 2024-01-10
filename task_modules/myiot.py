import requests
from dotenv import load_dotenv
import os

load_dotenv()

url = os.environ.get('MYIOT_SERVER_IP')

module_list = {
    'main': '11111',
    'sub': '11112',
    'closet': '11113',
    'ac': '11121',
    'ac_temp': '11122',
    'alarm': '11123'
}


def getJson(module, value):
    assert module in module_list.keys()

    if value == 'ON' or value == 'on':
        return {"requestId": "VAL", "data": {"id": module_list[module], 'reqVal': 'ON'}}
    elif value == 'OFF' or value == 'off':
        return {"requestId": "VAL", "data": {"id": module_list[module], 'reqVal': 'OFF'}}
    else:
        assert value.isnumeric()
        return {"requestId": "VAL", "data": {"id": module_list[module], 'reqVal': value}}


def execute(command, query=None):
    # ex) main off
    split = command.split()

    assert len(split) == 2
    jsonData = getJson(split[0], split[1])

    try:
        requests.request("GET", url,
                         json=jsonData, timeout=5),
    except:
        return None, False

    return None, False


if __name__ == "__main__":
    print(execute("sub on"))
