import requests
import json
from constants import constants


def send_classification(user_firebase_token, notification_text, notification_id):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'key=' + constants.serverToken,
    }

    body = {
        'notification':
            {
                'title': notification_text,
                'body': 'Cuidado',
                'tag': notification_id
            },
        'to': user_firebase_token,
        'priority': 'high',
    }
    response = requests.post(
        constants.serverUrl,
        headers=headers,
        data=json.dumps(body)
    )
    if response.status_code == 200:
        print('[SUCCESS]: Push enviado')
    else:
        print("[ERROR] Falha ao enviar o push")
