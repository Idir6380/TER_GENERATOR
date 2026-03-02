import subprocess
import os
import json
import requests
import time
from grobid_client.grobid_client import GrobidClient


GROBID_VERSION = "0.8.1"
GROBID_FOLDER = f"grobid-{GROBID_VERSION}"


def run_command(command, cwd=None):
    """Exécute une commande shell proprement."""
    subprocess.run(command, shell=True, check=True, cwd=cwd)


def install_dependencies():
    print("Installation de Java...")
    run_command("apt-get install -y openjdk-11-jdk-headless")


def download_grobid():
    if not os.path.exists(GROBID_FOLDER):
        print("Téléchargement de Grobid...")
        run_command(
            f"wget https://github.com/kermitt2/grobid/archive/refs/tags/{GROBID_VERSION}.tar.gz"
        )
        run_command(f"tar -xzf {GROBID_VERSION}.tar.gz")


def build_grobid():
    print("Compilation de Grobid (2-3 minutes)...")
    run_command("./gradlew clean assemble", cwd=GROBID_FOLDER)


def create_directories():
    os.makedirs("/Users/vanessaguerrier/Downloads/projet_TER_M2/data/GreenMIR/text_xml1", exist_ok=True)


def create_config():
    config = {
        "grobid_server": "http://localhost:8070",
        "batch_size": 10,
        "sleep_time": 5,
        "timeout": 1000000,
        "coordinates": True
    }

    with open("config_grobid.json", "w") as f:
        json.dump(config, f)

    print("config_grobid.json créé.")


def start_grobid_server():
    print("Lancement du serveur Grobid...")
    log_file = open("grobid_server.log", "w")

    process = subprocess.Popen(
        ["./gradlew", "run"],
        cwd=GROBID_FOLDER,
        stdout=log_file,
        stderr=log_file
    )

    return process


def wait_for_server():
    url = "http://localhost:8070/api/isalive"

    print("Attente du démarrage du serveur...")
    for _ in range(20):
        try:
            r = requests.get(url)
            if r.status_code == 200:
                print("Grobid est prêt !")
                return
        except:
            pass

        time.sleep(10)

    raise Exception("Le serveur Grobid ne répond pas.")


def process_pdfs():
    client = GrobidClient(config_path="./config_grobid.json")

    client.process(
        "processFulltextDocument",
        input_path="/Users/vanessaguerrier/Downloads/projet_TER_M2/data/GreenMIR/pdfs",
        output="/Users/vanessaguerrier/Downloads/projet_TER_M2/data/GreenMIR/text_xml1",
        force=True
    )

    print("Traitement terminé !")


def main():
    install_dependencies()
    download_grobid()
    build_grobid()
    create_directories()
    create_config()

    process = start_grobid_server()
    wait_for_server()

    process_pdfs()

    print("Terminé.")
    process.terminate()


if __name__ == "__main__":
    main()