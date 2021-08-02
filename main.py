# Imports
import socket
import threading
import pyaudio

# "From" imports
from utils import NNET
from constants import constants
from nnet import nnet_logic
from datetime import datetime


def handle_connection(conn, connection_address, loaded_model):
    print("[CONNECTED]: Um novo usuario se conectou pelo endereço", connection_address)

    user_firebase_token = conn.recv(constants.CHUNK).decode("utf-8")
    print("[FIREBASE TOKEN]:", user_firebase_token)

    user_sounds_configuration = conn.recv(constants.CHUNK).decode("utf-8")
    print("[USER SOUNDS CONFIG]:", user_sounds_configuration)
    [isCarHornEnable, isGunShotEnable, isDogBarkEnable, isSirenEnable] = user_sounds_configuration.split()

    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=constants.FORMAT,
        channels=constants.CHANNELS,
        rate=constants.RATE,
        output=True,
        frames_per_buffer=constants.CHUNK
    )

    startTime = datetime.now()
    audio_data = []
    while True:
        try:
            data = conn.recv(constants.CHUNK)
            if not data:
                print("[ERROR]: Conexão interrompida")
                break
            stream.write(data)
            audio_data.append(data)
            if (datetime.now() - startTime).total_seconds() > 3:
                print("[START ANALYSIS]:")
                thread = threading.Thread(
                    target=nnet_logic.classification,
                    args=(
                        audio,
                        audio_data,
                        model,
                        user_firebase_token,
                        isCarHornEnable,
                        isGunShotEnable,
                        isDogBarkEnable,
                        isSirenEnable
                    )
                )
                thread.start()
                audio_data = []
                startTime = datetime.now()

        except:
            print("[ERROR]: Conexão interrompida")
            break

    print("[OK]: Conexão encerrada")
    conn.close()


def start_server(loaded_model):
    print("[START]: Iniciando Socket")
    server.listen()
    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_connection, args=(conn, addr, loaded_model))
        thread.start()


# Start
nnet_logic.model_creation()
model = nnet_logic.load_model()

# Start socket server
ip = socket.gethostbyname(socket.gethostname())
address = (ip, constants.port)
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(address)

start_server(model)

