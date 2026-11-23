import csv
import os
import re
from pathlib import Path

ARABIC_PARENTHESIS = re.compile(r"\s*\([^)]*[\u0600-\u06FF][^)]*\)")

ARABIC_TEXT = re.compile(r"\s*/?\s*[\u0600-\u06FF].*")


def locate_data_directory():
    base_directory = Path(__file__).resolve().parent
    candidates = [base_directory, base_directory / "data"]
    if os.environ.get("DATA_DIR"):
        candidates.insert(0, Path(os.environ["DATA_DIR"]))
    for candidate in candidates:
        if (candidate / "informations.csv").exists():
            return candidate
    return base_directory


DATA_DIRECTORY = locate_data_directory()


def remove_arabic(value):
    return ARABIC_TEXT.sub("", ARABIC_PARENTHESIS.sub("", value)).strip()


def read_csv(file_name):
    path = DATA_DIRECTORY / file_name
    if not path.exists():
        print(f"[Avertissement] Fichier introuvable : {path}")
        return []
    with open(path, encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    return [{key.strip(): remove_arabic((value or "").strip()) for key, value in row.items()} for row in rows]


INFO_ROWS = read_csv("informations.csv")
ORG_ROWS = read_csv("organigramme.csv")
COUNCIL_ROWS = read_csv("conseil_scientifique_technologique.csv")
CALENDAR_ROWS = read_csv("calendrier_universitaire.csv")
FILIERE_ROWS = read_csv("filieres.csv")
FORMATION_ROWS = read_csv("formations.csv")
