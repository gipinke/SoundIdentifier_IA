# Imports
import socket
import threading
import pyaudio
import os
import shutil

# "From" imports
from constants import constants
from nnet import nnet_logic
from datetime import datetime
from os import listdir
from os.path import join, isdir

# Global variables
folder_path = "C:\\Users\\giova\\Desktop\\Rede Neural\\"


def remove_older_user_folders():
    dir_folds = [f for f in listdir(folder_path) if isdir(join(folder_path, f))]
    folds_to_remove = [fold for fold in dir_folds if "user_" in fold]
    for fold_name in folds_to_remove:
        shutil.rmtree(f"{folder_path}{fold_name}")


def handle_connection(conn, connection_address, loaded_model):
    print("[CONNECTED]: Um novo usuario se conectou pelo endereço", connection_address)
    user_dir_path = f"{folder_path}user_{connection_address[0]}_{connection_address[1]}"

    try:
        os.mkdir(user_dir_path)
    except OSError:
        print("[ERROR]: Falha na criação do diretório do usuário")
        conn.close()
        return
    else:
        print("[SUCCESS]: Sucesso na criação do diretório do usuário")
        print("\n--------------------------------------------------------------------")

    try:
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
                    print("\n--------------------------------------------------------------------")
                    print(f"[START ANALYSIS]: {datetime.now()}")
                    thread = threading.Thread(
                        target=nnet_logic.classification,
                        args=(
                            audio_data,
                            loaded_model,
                            user_firebase_token,
                            isCarHornEnable,
                            isGunShotEnable,
                            isDogBarkEnable,
                            isSirenEnable,
                            user_dir_path
                        )
                    )
                    thread.start()
                    audio_data = []
                    startTime = datetime.now()

            except:
                print("[ERROR]: Conexão interrompida")
                break
    except:
        print("[ERROR]: Conexão interrompida")

    print("[END]: Conexão encerrada")
    conn.close()


def start_server(loaded_model):
    print("[START]: Iniciando Socket")
    server.listen()
    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_connection, args=(conn, addr, loaded_model))
        thread.start()


# Remove older user folders in case they were not deleted
remove_older_user_folders()

# Start
nnet_logic.model_creation()
model = nnet_logic.load_model()

# Start socket server
ip = socket.gethostbyname(socket.gethostname())
address = (ip, constants.port)
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(address)

start_server(model)
