import datetime as dt
import difflib
import re
import unicodedata
from typing import NamedTuple

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier

from database import (
    CALENDAR_ROWS, COUNCIL_ROWS, DATA_DIRECTORY, FILIERE_ROWS, FORMATION_ROWS, INFO_ROWS, ORG_ROWS,
)


INFO_SOURCE = "informations.csv"
ORG_SOURCE = "organigramme.csv"
COUNCIL_SOURCE = "conseil_scientifique_technologique.csv"
CALENDAR_SOURCE = "calendrier_universitaire.csv"
FILIERES_SOURCE = "filieres.csv"
FORMATIONS_SOURCE = "formations.csv"

STOP_WORDS = set("""
le la les l un une des du de d et en au aux est sont a ont ete etre que qui quoi qu quel quels quelle quelles
comment combien ou il elle ils elles on nous vous je tu me te se ce cet cette ces mon ma mes ton ta tes son sa
ses notre votre leur leurs dans sur sous pour par avec sans chez y t s m n c j ne pas plus donne donnez dis dites
parle parlez moi toi lui svp plait stp voudrais veux souhaite savoir connaitre dire peux pouvez peut possible
ai as avons avez fait faire lieu occupe occuper concerne interesse
""".split())

WEAK_QUERY_WORDS = {"existe", "existent", "disponible"}

COVERAGE_IGNORED_WORDS = {"quand", "institut", "iset"}

WHEN_WORDS = {"quand", "date"}

SEMESTER_WORDS = {"semestre", "s1", "s2", "premier", "deuxieme", "second", "1er"}

NEXT_WORDS = {"prochain", "prochains", "prochaine", "prochaines", "suivant", "suivante", "bientot", "venir"}

TODAY_WORDS = {"aujourd", "hui", "maintenant", "actuellement"}

HEAD_WORDS = {"chef", "cheffe", "responsable", "president"}

DIRECTOR_WORDS = {"directeur", "directrice", "doyen"}

DIRECTOR_CONTEXT_WORDS = {"institut", "iset", "kebili", "etude", "stage"}

STUDIES_WORDS = {"etude", "stage"}

IDENTITY_WORDS = {"institut", "iset", "kebili", "isetk", "presentation", "carte", "identite"}

DIRECTOR_ROLE = "Directeur de l'Institut"

STUDIES_DIRECTOR_ROLE = "Directeur des Études et des Stages"

MIN_SCORE = 0.14

MIN_COVERAGE = 0.6

GROUP_CHUNK_BOOST = 1.2

SMALLTALK_MAX_WORDS = 5

SMALLTALK_MIN_KNOWN_RATIO = 0.6

MONTHS = [
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
]

WEEKDAYS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]

INFO_KEYWORDS = {
    "Pays": "pays nation tunisie",
    "Ministère de tutelle": "ministere tutelle enseignement superieur recherche scientifique",
    "Direction générale": "direction generale etudes technologiques",
    "Établissement": "etablissement nom de l'institut c'est quoi definition",
    "Abréviation": "abreviation sigle iset",
    "Adresse": "adresse localisation situe se trouve ou route de gabes bp lieu institut emplacement plan comment venir acces aller code postal",
    "Téléphone": "telephone numero appeler tel portable gsm joindre contacter contact standard appel",
    "Fax": "fax telecopie numero fax",
    "Email": "email mail courriel adresse electronique ecrire message contact",
    "Décret de création": "decret creation cree arrete fondation loi texte juridique numero decret",
    "Date d'ouverture": "date ouverture quand ouvert ouvre cree creation fondation annee ouverture institut historique",
    "Présentation de la région": "presentation region gouvernorat ville kebili tourisme agriculture oasis desert eau nappe sud-ouest situation geographique parle de la region souf nefzaoua climat",
    "Contexte de création": "contexte creation historique",
}

INFO_KEYWORD_PREFIXES = {
    "Raison de création": "raison raisons pourquoi creation institut",
    "Mission principale": "mission missions role objectifs fonction",
}

INFO_GROUPS = [
    (
        "Carte d'identité de l'institut",
        ["Établissement", "Abréviation", "Pays", "Ministère de tutelle", "Direction générale",
         "Décret de création", "Date d'ouverture"],
        "carte identite presentation institut qu'est-ce que l'institut informations generales a propos "
        "iset kebili etablissement institut superieur etudes technologiques",
        "Institut Supérieur des Études Technologiques de Kébili — carte d'identité :",
    ),
    (
        "Coordonnées et adresse",
        ["Adresse", "Téléphone", "Fax", "Email"],
        "contact coordonnees adresse telephone fax email mail comment contacter joindre administration "
        "ou se trouve situe localisation institut",
        "Coordonnées de l'institut :",
    ),
    (
        "Raisons de la création de l'institut",
        ["Contexte de création"] + [f"Raison de création {i}" for i in range(1, 8)],
        "raisons creation institut pourquoi contexte motifs",
        "Raisons de la création de l'institut :",
    ),
    (
        "Missions principales de l'institut",
        [f"Mission principale {i}" for i in range(1, 8)],
        "missions institut role objectifs que fait l'institut fonction",
        "Missions principales de l'institut :",
    ),
]

IDENTITY_CARD_TITLE = INFO_GROUPS[0][0]

ENTITY_KEYWORDS = {
    "Directeur de L'ISET": "directeur chef responsable doyen direction",
    "Conseil Scientifique et Technologique": "conseil scientifique",
    "Secrétariat (Direction)": "secretariat direction secretaire du directeur",
    "Comité de direction": "comite direction gestion",
    "Directeur des Etudes et des Stages": "directeur etudes stages",
    "Secrétaire Générale": "secretaire generale general secretariat general services",
    "Secrétariat (Direction des Etudes et des Stages)": "secretariat etudes stages",
    "Département S.Economiques et Gestion": "sciences economiques gestion economie management",
    "Département Technologies de L'informatique": "informatique technologies information",
    "Département Génie mécanique": "mecanique genie mecanique",
    "Département Génie Electrique": "electrique electricite genie electrique",
    "Département Génie des Procédés": "procedes chimie genie procedes",
    "Service Formation continue et relations extérieures": "formation continue relations exterieures",
    "Centre 4C": "centre 4c carriere certification competences emploi",
    "Service des Stages et PFEs": "stage stages pfe projets fin etudes",
    "Laboratoires , Ateliers et Centre Informatique": "laboratoire laboratoires atelier ateliers centre informatique",
    "Enseignement à distance": "enseignement distance cours en ligne plateforme",
    "Partenariat et relations extérieures": "partenariat cooperation relations exterieures",
    "Suivi des diplômés": "suivi diplomes anciens etudiants insertion emploi",
    "Secrétariat (Secrétariat Général)": "secretariat general",
    "Bureau d'ordre": "bureau ordre courrier arrivee depart",
    "Service Tirage": "tirage photocopie impression reprographie",
    "Service des Activités Socioculturelles": "activites socioculturelles culture clubs",
    "Secrétaire principale": "secretaire principale",
    "Service technique et maintenance": "technique maintenance",
    "Secrétaire d'établissement": "secretaire etablissement",
    "Service scolarité": "scolarite inscription reinscription etudiants",
    "Service suivi des cours": "suivi cours seances absences enseignants",
    "Service examens": "examens devoirs controles",
    "Service personnels": "personnel personnels agents employes",
    "Service financier": "financier finances budget comptabilite",
    "Service achats": "achats marches approvisionnement",
    "Magasin": "magasin stock",
    "Bibliothèque": "bibliotheque livres ouvrages lecture",
    "Service archives": "archives documents",
    "Strandard": "standard telephonique accueil",
    "Service de communication": "communication media",
    "Cellule d'écoute": "cellule ecoute soutien aide psychologique reclamations",
}

ORG_TYPE_GROUPS = [
    (("Département",), "Départements", "departement departements"),
    (("Service", "Service / Unité", "Service / Accueil"), "Services", "service services"),
    (("Secrétariat",), "Secrétariats", "secretariat secretariats secretaire"),
    (("Centre", "Structure pédagogique"), "Centres et structures pédagogiques", "centre centres laboratoire atelier"),
    (("Sous-unité / Activité",), "Sous-unités et activités", "sous-unite sous-unites activite activites"),
]

DEPARTMENT_KEYWORDS = {
    "Sciences Économiques et Gestion": "economie economiques gestion",
    "Génie des Procédés": "procedes chimie",
    "Technologies de l'Informatique": "informatique",
    "Génie Électrique": "electrique electricite",
    "Génie Mécanique": "mecanique",
}

EVENT_GROUPS = {
    "rentree": {
        "types": ["Rentrée / Événement"],
        "title": "Rentrée universitaire",
        "keywords": "rentree universitaire debut des cours commence commencer reprise debut annee",
        "filters": "rentree",
    },
    "exams": {
        "types": ["Examens / Évaluation"],
        "title": "Devoirs et examens",
        "keywords": "examen examens devoir devoirs controle controles synthese evaluation",
        "filters": "examen examens devoir devoirs controle controles synthese evaluation",
    },
    "holidays": {
        "types": ["Jour férié / Fête"],
        "title": "Jours fériés et fêtes officielles",
        "keywords": "jour ferie feries fete fetes officielle independance martyrs travail aid nouvel an",
        "filters": "ferie feries fete fetes",
    },
    "vacations": {
        "types": ["Vacances"],
        "title": "Vacances universitaires",
        "keywords": "vacances conge conges hiver printemps",
        "filters": "vacances conge conges",
    },
    "revision": {
        "types": ["Révision", "Rattrapage et Révision", "Arrêt des cours"],
        "title": "Révisions, rattrapage et arrêt des cours",
        "keywords": "revision revisions rattrapage arret des cours",
        "filters": "revision revisions rattrapage",
    },
    "stages": {
        "types": ["Stages", "Démarrage / Stages PFE"],
        "title": "Stages",
        "keywords": "stage stages pfe projet fin etudes",
        "filters": "stage stages",
    },
    "pfe": {
        "types": ["Soutenances PFE"],
        "title": "Soutenances de PFE",
        "keywords": "soutenance soutenances pfe projet fin etudes",
        "filters": "soutenance soutenances pfe",
    },
    "deliberations": {
        "types": ["Délibérations"],
        "title": "Délibérations",
        "keywords": "deliberation deliberations jury",
        "filters": "deliberation deliberations",
    },
    "results": {
        "types": ["Résultats"],
        "title": "Annonce des résultats",
        "keywords": "resultat resultats annonce affichage",
        "filters": "resultat resultats",
    },
    "reclamations": {
        "types": ["Réclamations / Notes", "Administration / Notes", "Administration / PVs"],
        "title": "Réclamations, révision des notes et procès-verbaux",
        "keywords": "reclamation reclamations notes pv proces verbaux correction",
        "filters": "reclamation reclamations pv",
    },
    "documents": {
        "types": ["Documents officiels"],
        "title": "Attestations et diplômes",
        "keywords": "attestation attestations diplome diplomes releve de notes remise",
        "filters": "attestation attestations diplome diplomes releve releves",
    },
}

SEMESTER_KEYWORDS = {
    "Premier semestre": "premier semestre semestre 1 s1 1er",
    "Deuxième semestre": "deuxieme second semestre semestre 2 s2",
}

MENTION_KEYWORDS = {
    "Génie Electrique": "electrique electricite automatisme electronique",
    "Génie Mécanique": "mecanique construction fabrication production",
    "Management": "gestion management finance comptabilite commerce",
    "Technologies de l'Informatique": "informatique developpement systemes information programmation reseaux",
}

LEVEL_KEYWORDS = {
    "L1": "l1 licence 1 premiere annee niveau 1",
    "L2": "l2 licence 2 deuxieme annee niveau 2",
    "L3": "l3 licence 3 troisieme annee niveau 3",
}

SEMESTER_ORDINALS = {
    "S1": "premier", "S2": "deuxieme", "S3": "troisieme", "S4": "quatrieme", "S5": "cinquieme", "S6": "sixieme",
}

FORMATION_KEYWORDS = "programme matieres unites enseignement ue credits coefficients modules cours specialite filiere mention"

TRAINING_PHRASES = {
    "greeting": [
        "bonjour", "salut", "bonsoir", "hello", "coucou", "hey", "bonjour tout le monde",
        "bonne journée", "salutations", "bonsoir à tous",
    ],
    "identity": [
        "qui es-tu", "comment tu t'appelles", "que fais-tu", "comment ça marche",
        "que peux-tu faire", "présente-toi", "qui t'a créé",
    ],
    "thanks": [
        "merci", "merci beaucoup", "je te remercie", "c'est gentil", "super merci", "mille mercis",
    ],
    "goodbye": [
        "au revoir", "à bientôt", "bye", "adieu", "bonne soirée",
    ],
}

REPLIES = {
    "greeting": "Bonjour ! Je suis l'assistant de l'Institut Supérieur des Études Technologiques de Kébili. "
                "Posez-moi vos questions sur l'institut, son organisation, le Conseil scientifique, le "
                "calendrier universitaire ou les formations (matières, crédits, coefficients). "
                "Je peux aussi calculer votre moyenne et vos crédits : écrivez « moyenne ».",
    "identity": "Je suis un assistant conçu avec des modèles d'apprentissage automatique (TF-IDF et arbre de "
                "décision), sans API externe. Je réponds à partir des données de l'ISET Kébili : informations de "
                "l'institut, organigramme, Conseil scientifique et technologique, calendrier universitaire et formations. "
                "Écrivez « moyenne » pour calculer votre moyenne et vos crédits.",
    "thanks": "Je vous en prie ! Je suis toujours ravi de vous aider.",
    "goodbye": "Au revoir ! Bonne continuation.",
}

SUGGESTIONS = (
    "Essayez par exemple :\n"
    "• Quelle est l'adresse de l'institut et son numéro de téléphone ?\n"
    "• Qui est le directeur de l'institut ?\n"
    "• Quels sont les départements de l'institut ?\n"
    "• Qui est le chef du département informatique ?\n"
    "• Quand commence la rentrée universitaire ?\n"
    "• Quand ont lieu les examens ?\n"
    "• Que se passe-t-il en décembre ?\n"
    "• Quelles sont les matières d'informatique en S3 ?\n"
    "• Calculer ma moyenne et mes crédits"
)

EMPTY_MESSAGE_REPLY = "Écrivez votre question pour que je puisse vous répondre."

NOT_FOUND_REPLY = "Je n'ai trouvé aucune information correspondant à votre question dans les données de l'institut.\n"


def normalize(text):
    text = unicodedata.normalize("NFD", str(text).lower())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = text.replace("œ", "oe").replace("æ", "ae")
    text = re.sub(r"[^\w\s]|_", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text):
    result = []
    for word in normalize(text).split():
        if word in STOP_WORDS:
            continue
        if len(word) > 3 and word.endswith("s"):
            word = word[:-1]
        if len(word) >= 2 or word.isdigit():
            result.append(word)
    return result


def prepare(text):
    return " ".join(tokens(text))


def prepare_query(text):
    words = tokens(text)
    strong_words = [word for word in words if word not in WEAK_QUERY_WORDS]
    return " ".join(strong_words or words)


SYNONYM_GROUPS = [
    "telephone tel portable gsm appeler appel joindre contacter contact standard",
    "email mail courriel ecrire message",
    "adresse localisation situe trouve emplacement lieu plan route gabes venir aller chemin acces localise",
    "fax telecopie",
    "directeur directrice doyen dirigeant patron",
    "chef cheffe responsable",
    "enseignant prof professeur formateur maitre",
    "etudiant eleve apprenant",
    "inscription inscrire reinscription scolarite admission candidature",
    "examen ds devoir controle test epreuve evaluation synthese",
    "vacances conge repos ferie",
    "rentree reprise debut commencement commence demarrage",
    "fin termine finit cloture",
    "stage stagiaire pfe projet",
    "diplome licence certificat attestation reussite",
    "matiere module cours discipline ue unite enseignement",
    "credit ects",
    "coefficient coef coeff poids",
    "semestre sem",
    "filiere specialite mention parcours branche section option formation",
    "bibliotheque bibli livre ouvrage lecture",
    "informatique info dsi tic ti programmation developpement",
    "electrique electricite elec automatisme",
    "mecanique meca fabrication conception",
    "gestion management economie commerce",
    "creation cree fonde fondation ouverture ouvert histoire",
    "mission role objectif but vocation",
    "region gouvernorat ville kebili",
    "reclamation recorrection contestation plainte",
    "conseil comite commission",
    "membre composition",
    "club activite socioculturel culture sport loisir association",
    "rattrapage rattraper session controle",
    "resultat releve",
    "date quand jour calendrier planning agenda",
]


def build_synonyms():
    lookup = {}
    for group in SYNONYM_GROUPS:
        words = {prepare(word) for word in group.split()}
        words.discard("")
        for word in words:
            lookup.setdefault(word, set()).update(words)
    return lookup


SYNONYMS = build_synonyms()


def expand_words(words):
    extra = []
    for word in words:
        for synonym in sorted(SYNONYMS.get(word, ())):
            if synonym != word and synonym not in words and synonym not in extra:
                extra.append(synonym)
    return extra


def bullets(items):
    return "\n".join(f"• {item}" for item in items)


MONTH_LOOKUP = {normalize(name): number for number, name in enumerate(MONTHS, start=1)}

TYPE_TO_GROUP = {
    event_type: group
    for group, details in EVENT_GROUPS.items()
    for event_type in details["types"]
}

GROUP_FILTER_WORDS = {group: set(tokens(details["filters"])) for group, details in EVENT_GROUPS.items()}

ISO_DATE = re.compile(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b")

FRENCH_DATE = re.compile(r"\b(\d{1,2})[-/](\d{1,2})[-/](20\d{2})\b")


def parse_date(value):
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


def load_events():
    events = []
    for row in CALENDAR_ROWS:
        start, end = parse_date(row.get("Date_Debut", "")), parse_date(row.get("Date_Fin", ""))
        if start and end:
            events.append({
                **row,
                "start": start,
                "end": end,
                "is_week": row.get("Type_Evenement") == "Semaine de cours",
                "group": TYPE_TO_GROUP.get(row.get("Type_Evenement")),
            })
    return events


EVENTS = load_events()


def format_date(date, with_year=True):
    text = f"{WEEKDAYS[date.weekday()]} {date.day} {MONTHS[date.month - 1]}"
    return f"{text} {date.year}" if with_year else text


def format_range(start, end):
    if start == end:
        return format_date(start)
    if start.year == end.year:
        return f"du {format_date(start, False)} au {format_date(end)}"
    return f"du {format_date(start)} au {format_date(end)}"


def event_line(event):
    return f"• {event['Intitule_FR']} : {format_range(event['start'], event['end'])}"


def academic_year_of(month):
    if not EVENTS:
        return dt.date.today().year
    first_start = min(event["start"] for event in EVENTS)
    return first_start.year if month >= first_start.month else first_start.year + 1


class Chunk(NamedTuple):
    source: str
    kind: str
    title: str
    search: str
    answer: str
    body: str = ""


def make_chunk(source, kind, title, search, answer):
    return Chunk(source, kind, title, prepare(f"{title} {search}"), answer, prepare(answer))


def info_keywords(rubric):
    if rubric in INFO_KEYWORDS:
        return INFO_KEYWORDS[rubric]
    prefix = next((prefix for prefix in INFO_KEYWORD_PREFIXES if rubric.startswith(prefix)), None)
    return INFO_KEYWORD_PREFIXES.get(prefix, "")


def build_info_chunks():
    values = {row["Rubrique"]: row["Valeur"] for row in INFO_ROWS if row.get("Rubrique")}

    for rubric, value in values.items():
        yield make_chunk(INFO_SOURCE, "info", rubric, f"{info_keywords(rubric)} {value}", f"{rubric} : {value}")

    for title, rubrics, keywords, intro in INFO_GROUPS:
        available = [rubric for rubric in rubrics if rubric in values]
        if available:
            lines = bullets(f"{rubric} : {values[rubric]}" for rubric in available)
            yield make_chunk(INFO_SOURCE, "info", title, keywords, f"{intro}\n{lines}")


def build_org_chunks():
    by_id = {row["ID"]: row for row in ORG_ROWS if row.get("ID")}
    children = {}
    for row in ORG_ROWS:
        if row.get("Parent_ID"):
            children.setdefault(row["Parent_ID"], []).append(row)

    for row in ORG_ROWS:
        name, kind = row["Entite"], row.get("Type", "")
        parent = by_id.get(row.get("Parent_ID", ""))
        keywords = ENTITY_KEYWORDS.get(name, "")
        answer = f"{name} — {kind}"
        answer += f", rattaché à : {parent['Entite']}." if parent else ", tête de la structure organisationnelle de l'institut."
        sub_structures = children.get(row.get("ID"), [])
        if sub_structures:
            answer += f"\nComprend {len(sub_structures)} structures :\n"
            answer += bullets(sub["Entite"] for sub in sub_structures)
            keywords += " rattachees comprend structures organisation"
        yield make_chunk(ORG_SOURCE, "org", name, f"{keywords} {kind} structure organisationnelle", answer)

    for types, title, keywords in ORG_TYPE_GROUPS:
        entities = [row for row in ORG_ROWS if row.get("Type") in types]
        if entities:
            answer = f"Nombre de {title.lower()} à l'institut : {len(entities)}\n"
            answer += bullets(entity["Entite"] for entity in entities)
            yield make_chunk(ORG_SOURCE, "org", title, f"{keywords} nombre combien liste tous quels", answer)

    top = by_id.get("1")
    if top:
        answer = f"L'institut est dirigé par : {top['Entite']}. Dépendent directement de lui :\n"
        answer += bullets(sub["Entite"] for sub in children.get("1", []))
        yield make_chunk(
            ORG_SOURCE, "org", "Organigramme de l'institut",
            "organigramme structure organisationnelle organisation de l'institut hierarchie composition direction",
            answer,
        )


COUNCIL_MEMBERS = [
    row for row in COUNCIL_ROWS if row.get("Section") == "Membres" and row.get("Nom_Complet")
]

COUNCIL_YEAR = next((row["Annee_Universitaire"] for row in COUNCIL_ROWS if row.get("Annee_Universitaire")), "")


def describe_person(row):
    name, role, department = row["Nom_Complet"], row.get("Role_ou_Qualite", ""), row.get("Departement", "")
    feminine = name.startswith(("Mme", "Mlle"))
    if role == "Chef de département":
        role = f"{'Cheffe' if feminine else 'Chef'} du département {department}"
    elif role.startswith("Membre représentant"):
        role = f"Membre représentant les enseignants du département {department}"
    return f"{name} — {role}"


def build_council_chunks():
    source, base = COUNCIL_SOURCE, "conseil scientifique et technologique"
    suffix = f" (Conseil scientifique et technologique {COUNCIL_YEAR})"

    for row in COUNCIL_ROWS:
        if row.get("Section") == "Missions" and row.get("Texte_Francais"):
            yield make_chunk(
                source, "council", "Missions du Conseil scientifique et technologique",
                f"{base} missions role attributions competences que fait le conseil",
                row["Texte_Francais"],
            )

    for row in COUNCIL_MEMBERS:
        role, department = row.get("Role_ou_Qualite", ""), row.get("Departement", "")
        position = "qui est chef responsable directeur" if "Chef" in role or "Directeur" in role else "membre representant enseignants"
        yield make_chunk(
            source, "council", row["Nom_Complet"],
            f"{base} {role} {department} {DEPARTMENT_KEYWORDS.get(department, '')} {describe_person(row)} {position}",
            describe_person(row) + suffix,
        )

    if not COUNCIL_MEMBERS:
        return

    yield make_chunk(
        source, "council", "Membres du Conseil scientifique et technologique",
        f"{base} membres liste qui sont les membres composition du conseil nombre combien",
        f"Membres du Conseil scientifique et technologique ({COUNCIL_YEAR}) — {len(COUNCIL_MEMBERS)} membres :\n"
        + bullets(describe_person(row) for row in COUNCIL_MEMBERS),
    )

    heads = [row for row in COUNCIL_MEMBERS if row["Role_ou_Qualite"] == "Chef de département"]
    if heads:
        yield make_chunk(
            source, "council", "Chefs de département",
            f"{base} chefs responsables qui sont les chefs",
            "Chefs de département :\n" + bullets(describe_person(row) for row in heads),
        )

    representatives = [row for row in COUNCIL_MEMBERS if row["Role_ou_Qualite"].startswith("Membre représentant")]
    if representatives:
        yield make_chunk(
            source, "council", "Représentants des enseignants au Conseil",
            f"{base} representants des enseignants membres representant corps enseignant",
            "Représentants des enseignants au Conseil :\n" + bullets(describe_person(row) for row in representatives),
        )

    for department, keywords in DEPARTMENT_KEYWORDS.items():
        members = [row for row in COUNCIL_MEMBERS if row["Departement"] == department]
        if members:
            yield make_chunk(
                source, "council", f"Département {department} au Conseil",
                f"{base} {department} {keywords} membres enseignants representants du departement",
                f"Représentants du département {department} au Conseil scientifique et technologique :\n"
                + bullets(describe_person(row) for row in members),
            )

    direction = [row for row in COUNCIL_MEMBERS if row["Role_ou_Qualite"].startswith("Directeur")]
    if direction:
        yield make_chunk(
            source, "council", "Direction de l'institut",
            f"{base} directeurs de l'institut direction responsables administration",
            bullets(describe_person(row) for row in direction),
        )


def build_calendar_chunks():
    if not EVENTS:
        return

    year = EVENTS[0].get("Annee_Universitaire", "")
    first = min(event["start"] for event in EVENTS)
    last = max(event["end"] for event in EVENTS)
    week_titles = {event["Intitule_FR"].split(" (")[0] for event in EVENTS if event["is_week"]}

    yield make_chunk(
        CALENDAR_SOURCE, "calendar", f"Année universitaire {year}",
        f"calendrier universitaire {year} debut fin de l'annee universitaire quand commence quand se termine "
        "programme de l'annee dates importantes resume du calendrier",
        f"Année universitaire {year} (Licence) :\n"
        f"• Début de l'année : {format_date(first)}\n"
        f"• Dernière échéance du calendrier : {format_date(last)}\n"
        f"• Nombre de semaines de cours enregistrées : {len(week_titles)}\n"
        "Vous pouvez demander par exemple : Quand ont lieu les examens ? Quelles sont les vacances ? "
        "Que se passe-t-il en décembre ?",
    )

    for event in EVENTS:
        months = {event["start"].month, event["end"].month}
        month_keywords = " ".join(MONTHS[month - 1] for month in months)
        semester = event.get("Axe_Principal", "")
        yield make_chunk(
            CALENDAR_SOURCE, "calendar", event["Intitule_FR"],
            f"{event.get('Type_Evenement', '')} {month_keywords} "
            f"{SEMESTER_KEYWORDS.get(semester, 'stage stages')} quand date {event['start']} {event['end']}",
            f"{event_line(event)}\n{semester} — Licence — Année universitaire {event.get('Annee_Universitaire', '')}",
        )

    for group, details in EVENT_GROUPS.items():
        events = [event for event in EVENTS if event["group"] == group]
        if not events:
            continue
        title = details["title"]
        yield make_chunk(
            CALENDAR_SOURCE, "calendar", title,
            f"{details['keywords']} quand dates tous liste calendrier annee {year}",
            f"{title} — année universitaire {year} :\n" + "\n".join(event_line(event) for event in events),
        )
        for semester, semester_keywords in SEMESTER_KEYWORDS.items():
            semester_events = [event for event in events if event.get("Axe_Principal") == semester]
            if semester_events and len(semester_events) != len(events):
                yield make_chunk(
                    CALENDAR_SOURCE, "calendar", f"{title} ({semester})",
                    f"{details['keywords']} {semester_keywords} quand dates",
                    f"{title} — {semester} :\n" + "\n".join(event_line(event) for event in semester_events),
                )


def format_number(value):
    return f"{float(value):g}".replace(".", ",")


def describe_units(rows):
    units = {}
    for row in rows:
        units.setdefault(row["Unite_Enseignement"], []).append(row)
    lines = []
    for unit, subjects in units.items():
        lines.append(
            f"• {unit} — {format_number(subjects[0]['Credits_UE'])} crédits, "
            f"coefficient {format_number(subjects[0]['Coefficient_UE'])}"
        )
        lines.extend(
            f"    – {row['Matiere']} : {format_number(row['Credits'])} crédits, coefficient {format_number(row['Coefficient'])}"
            for row in subjects
        )
    return "\n".join(lines)


def build_filiere_chunks():
    mentions = {}
    for row in FILIERE_ROWS:
        mention = mentions.setdefault(row["Mention"], {"domain": row["Domaine"], "specialties": {}})
        mention["specialties"].setdefault(row["Specialite"], []).append(row["Niveaux"])

    lines = []
    for name, details in mentions.items():
        specialties = " ; ".join(
            f"{specialty} ({', '.join(levels)})" for specialty, levels in details["specialties"].items()
        )
        line = f"{name} ({details['domain']}) : {specialties}"
        lines.append(line)
        yield make_chunk(
            FILIERES_SOURCE, "formation", f"Filière {name}",
            f"{MENTION_KEYWORDS.get(name, '')} filiere mention specialite specialites parcours licence domaine formation",
            f"Filière {line}",
        )

    if lines:
        yield make_chunk(
            FILIERES_SOURCE, "formation", "Filières et spécialités de l'institut",
            "filieres specialites mentions parcours licence formations offre de formation diplomes quelles filieres "
            "quelles specialites domaine",
            "Offre de formation de l'institut (Licence) :\n" + bullets(lines),
        )


def build_formation_chunks():
    programs = {}
    subjects = {}
    for row in FORMATION_ROWS:
        programs.setdefault((row["Mention"], row["Specialite"], row["Niveau"], row["Semestre"]), []).append(row)
        subjects.setdefault(row["Matiere"], []).append(row)

    for (mention, specialty, level, semester), rows in programs.items():
        obligatory = [row for row in rows if row["Type_UE"] == "Obligatoire"]
        optional = [row for row in rows if row["Type_UE"] == "Optionnelle"]
        unit_credits = {row["Unite_Enseignement"]: float(row["Credits_UE"]) for row in obligatory}
        answer = f"{mention} — {specialty} — {level}, {semester}\n"
        answer += f"Crédits des unités obligatoires : {format_number(sum(unit_credits.values()))}\n"
        if obligatory:
            answer += "\nUnités obligatoires :\n" + describe_units(obligatory)
        if optional:
            answer += "\n\nUnités optionnelles :\n" + describe_units(optional)
        ordinal = SEMESTER_ORDINALS.get(semester, "")
        unit_titles = " ".join(dict.fromkeys(row["Unite_Enseignement"] for row in rows))
        yield make_chunk(
            FORMATIONS_SOURCE, "formation", f"Programme {mention} — {specialty} — {level} {semester}",
            f"{FORMATION_KEYWORDS} {mention} {MENTION_KEYWORDS.get(mention, '')} {specialty} "
            f"{LEVEL_KEYWORDS.get(level, '')} {semester.lower()} semestre {semester[1:]} {ordinal} semestre "
            f"{unit_titles}",
            answer,
        )

    for subject, rows in subjects.items():
        occurrences = dict.fromkeys(
            f"• {row['Mention']} — {row['Specialite']}, {row['Niveau']} ({row['Semestre']}), unité « "
            f"{row['Unite_Enseignement']} » ({row['Type_UE'].lower()}) : {format_number(row['Credits'])} crédits, "
            f"coefficient {format_number(row['Coefficient'])}"
            for row in rows
        )
        places = " ".join(f"{row['Mention']} {row['Specialite']}" for row in rows)
        yield make_chunk(
            FORMATIONS_SOURCE, "formation", subject,
            f"matiere module cours credit credits coefficient coefficients {places}",
            f"{subject} :\n" + "\n".join(occurrences),
        )


CHUNKS = [
    *build_info_chunks(),
    *build_org_chunks(),
    *build_council_chunks(),
    *build_calendar_chunks(),
    *build_filiere_chunks(),
    *build_formation_chunks(),
]

IDENTITY_CARD = next((chunk for chunk in CHUNKS if chunk.title == IDENTITY_CARD_TITLE), None)

SEMESTER_SUFFIXES = tuple(f"({semester})" for semester in SEMESTER_KEYWORDS)

IS_SEMESTER_CHUNK = np.array([chunk.title.endswith(SEMESTER_SUFFIXES) for chunk in CHUNKS] or [False])

IS_CALENDAR_CHUNK = np.array([chunk.kind == "calendar" for chunk in CHUNKS] or [False])

GROUP_TITLES = {details["title"] for details in EVENT_GROUPS.values()}

IS_GROUP_CHUNK = np.array([chunk.title in GROUP_TITLES for chunk in CHUNKS] or [False])

char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)

word_vectorizer = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", sublinear_tf=True)

SEARCH_TEXTS = [chunk.search for chunk in CHUNKS] or ["vide"]

CHAR_MATRIX = char_vectorizer.fit_transform(SEARCH_TEXTS)

WORD_MATRIX = word_vectorizer.fit_transform(SEARCH_TEXTS)


def common_prefix_length(first, second):
    length = 0
    for a, b in zip(first, second):
        if a != b:
            break
        length += 1
    return length


def words_match(word, candidates):
    return any(
        candidate == word
        or (min(len(candidate), len(word)) >= 3
            and common_prefix_length(candidate, word) >= max(3, min(len(candidate), len(word)) - 2))
        for candidate in candidates
    )


def word_in_chunk(word, candidates):
    if words_match(word, candidates):
        return True
    return any(words_match(synonym, candidates) for synonym in SYNONYMS.get(word, ()) if synonym != word)


def coverage(query, chunk):
    query_words = [word for word in prepare_query(query).split() if word not in COVERAGE_IGNORED_WORDS]
    if not query_words:
        return 1.0
    chunk_words = chunk.search.split() + chunk.body.split()
    found = sum(1 for word in query_words if word_in_chunk(word, chunk_words))
    return found / len(query_words)


CHUNK_WORD_LISTS = [chunk.search.split() + chunk.body.split() for chunk in CHUNKS]

CHUNK_TITLE_WORDS = [set(prepare(chunk.title).split()) for chunk in CHUNKS]

TITLE_BONUS = 0.6


def raw_scores(text):
    return (
        0.55 * (CHAR_MATRIX @ char_vectorizer.transform([text]).T).toarray().ravel()
        + 0.45 * (WORD_MATRIX @ word_vectorizer.transform([text]).T).toarray().ravel()
    )


def retrieve(query, limit=10):
    prepared = prepare_query(query)
    if not prepared or not CHUNKS:
        return []
    query_words = set(prepared.split())
    scores = raw_scores(prepared)
    extra = expand_words(prepared.split())
    if extra:
        scores = np.maximum(scores, 0.9 * raw_scores(prepared + " " + " ".join(extra)))
    scores = np.where(IS_GROUP_CHUNK, scores * GROUP_CHUNK_BOOST, scores)
    title_hits = np.array([len(query_words & title) for title in CHUNK_TITLE_WORDS] or [0])
    scores = scores * (1 + TITLE_BONUS * title_hits / len(query_words))
    if not query_words & SEMESTER_WORDS:
        scores = np.where(IS_SEMESTER_CHUNK, scores * 0.9, scores)
    if query_words & WHEN_WORDS:
        scores = np.where(IS_CALENDAR_CHUNK, scores * 1.7, scores * 0.8)
    elif normalize(query).startswith("qui "):
        scores = np.where(IS_CALENDAR_CHUNK, scores * 0.6, scores)
    best = np.argsort(-scores)[:limit]
    return [(CHUNKS[index], float(scores[index])) for index in best]


FALLBACK_MIN_RATIO = 0.7


def keyword_fallback(message):
    """Recherche mot par mot dans tout le contenu des fichiers CSV (dernier recours)."""
    words = [word for word in prepare_query(message).split() if word not in COVERAGE_IGNORED_WORDS]
    if not words or not CHUNKS:
        return None
    total = len(CHUNKS)
    weights = {}
    for word in words:
        document_frequency = sum(1 for candidates in CHUNK_WORD_LISTS if word_in_chunk(word, candidates))
        weights[word] = float(np.log((total + 1) / (document_frequency + 1))) + 0.1
    best, best_ratio, best_score = None, 0.0, 0.0
    for chunk, candidates in zip(CHUNKS, CHUNK_WORD_LISTS):
        title_words = chunk.search.split()
        matched = [word for word in words if word_in_chunk(word, candidates)]
        if not matched:
            continue
        ratio = sum(weights[word] for word in matched) / sum(weights.values())
        score = ratio + 0.05 * sum(1 for word in matched if word_in_chunk(word, title_words))
        if score > best_score:
            best, best_ratio, best_score = chunk, ratio, score
    if best and best_ratio >= FALLBACK_MIN_RATIO:
        matched_count = sum(1 for word in words if word_in_chunk(word, CHUNK_WORD_LISTS[CHUNKS.index(best)]))
        if matched_count / len(words) >= 0.6:
            return best
    return None


def detect_groups(words):
    found = set()
    for group, filter_words in GROUP_FILTER_WORDS.items():
        if any(word in filter_words or any(word.startswith(f) and len(f) >= 4 for f in filter_words) for word in words):
            found.add(group)
    return found


def find_explicit_date(message):
    match = ISO_DATE.search(message)
    if match:
        year, month, day = map(int, match.groups())
    else:
        match = FRENCH_DATE.search(message)
        if not match:
            return None
        day, month, year = map(int, match.groups())
    try:
        return dt.date(year, month, day)
    except ValueError:
        return None


def find_month(words):
    return next((MONTH_LOOKUP[word] for word in words if word in MONTH_LOOKUP), None)


def events_on(day):
    events = [event for event in EVENTS if event["start"] <= day <= event["end"]]
    return sorted(events, key=lambda event: (event["is_week"], event["start"]))


def describe_day(day, head):
    events = events_on(day)
    if not events:
        return f"{head}\nAucune activité enregistrée dans le calendrier pour ce jour (repos hebdomadaire ou hors année universitaire)."
    return head + "\n" + "\n".join(event_line(event) for event in events)


def describe_month(month, year, groups, group_title):
    start = dt.date(year, month, 1)
    end = dt.date(year + (month == 12), month % 12 + 1, 1) - dt.timedelta(days=1)
    overlapping = [event for event in EVENTS if event["start"] <= end and event["end"] >= start]
    events = sorted(
        (event for event in overlapping if not event["is_week"] and (not groups or event["group"] in groups)),
        key=lambda event: event["start"],
    )
    head = f"{group_title + ' — ' if group_title else ''}{MONTHS[month - 1].capitalize()} {year} :"
    if not events:
        return f"{head}\nAucun événement enregistré ce mois-ci dans le calendrier."
    tail = ""
    if not groups and any(event["is_week"] for event in overlapping):
        tail = "\nLes autres jours : semaines de cours normales."
    return head + "\n" + "\n".join(event_line(event) for event in events) + tail


def describe_upcoming(groups, group_title, today):
    upcoming = sorted(
        (event for event in EVENTS
         if event["end"] >= today and not event["is_week"] and (not groups or event["group"] in groups)),
        key=lambda event: event["start"],
    )[:3]
    if not upcoming:
        subject = f" pour « {group_title} »" if group_title else ""
        return f"Aucun événement à venir{subject} dans le calendrier après la date du jour ({format_date(today)})."
    head = f"{'Le plus proche : ' + group_title if group_title else 'Prochains événements'} (à partir d'aujourd'hui, {format_date(today)}) :"
    return head + "\n" + "\n".join(event_line(event) for event in upcoming)


def date_answer(message):
    if not EVENTS:
        return None

    words = normalize(message).split()
    groups = detect_groups(tokens(message))
    group_title = ", ".join(EVENT_GROUPS[group]["title"] for group in groups)
    today = dt.date.today()

    day = find_explicit_date(message)
    month = None if day else find_month(words)

    if month:
        numbers = [int(number) for number in re.findall(r"\b\d{1,4}\b", normalize(message))]
        year = next((number for number in numbers if 2000 <= number <= 2100), None) or academic_year_of(month)
        day_number = next((number for number in numbers if 1 <= number <= 31), None)
        if not day_number:
            return describe_month(month, year, groups, group_title)
        try:
            day = dt.date(year, month, day_number)
        except ValueError:
            return f"La date {day_number} {MONTHS[month - 1]} n'est pas valide."

    if day:
        return describe_day(day, f"{format_date(day)} :")
    if set(words) & NEXT_WORDS:
        return describe_upcoming(groups, group_title, today)
    if set(words) & TODAY_WORDS and not groups:
        return describe_day(today, f"Aujourd'hui, {format_date(today)} :")
    return None


def build_department_aliases():
    aliases = {}
    for department, keywords in DEPARTMENT_KEYWORDS.items():
        candidates = [tokens(department)] + [tokens(word) for word in keywords.split()]
        aliases[department] = [alias for alias in candidates if alias]
    return aliases


DEPARTMENT_ALIASES = build_department_aliases()


def who_answer(message):
    if not COUNCIL_MEMBERS:
        return None

    words = set(tokens(message))
    suffix = f" (Conseil scientifique et technologique {COUNCIL_YEAR})"

    matches = sorted(
        ((max((len(alias) for alias in aliases if set(alias) <= words), default=0), department)
         for department, aliases in DEPARTMENT_ALIASES.items()),
        reverse=True,
    )
    matches = [match for match in matches if match[0]]
    if matches and (len(matches) == 1 or matches[0][0] > matches[1][0]) and words & HEAD_WORDS:
        heads = [
            row for row in COUNCIL_MEMBERS
            if row["Role_ou_Qualite"] == "Chef de département" and row["Departement"] == matches[0][1]
        ]
        if heads:
            return "\n".join(describe_person(row) for row in heads) + suffix

    if words & DIRECTOR_WORDS and words <= DIRECTOR_WORDS | DIRECTOR_CONTEXT_WORDS:
        role = STUDIES_DIRECTOR_ROLE if words & STUDIES_WORDS else DIRECTOR_ROLE
        rows = [row for row in COUNCIL_MEMBERS if normalize(row["Role_ou_Qualite"]) == normalize(role)]
        if rows:
            return "\n".join(describe_person(row) for row in rows) + suffix
    return None



MENTION_ALIASES = {
    "Technologies de l'Informatique": "informatique info dsi tic ti",
    "Génie Electrique": "electrique electricite elec ge automatisme aii",
    "Génie Mécanique": "mecanique meca gm cfm",
    "Management": "management gestion eco economie economique commerce mgt",
}

MENTION_ALIAS_WORDS = {mention: set(prepare(aliases).split()) for mention, aliases in MENTION_ALIASES.items()}

MENTION_ORDER = list(MENTION_ALIASES)

PROCEDES_WORDS = {"procede", "chimie"}

ORDINAL_LEVELS = {"premiere": 1, "1ere": 1, "deuxieme": 2, "seconde": 2, "2eme": 2, "troisieme": 3, "3eme": 3}

PROGRAM_INTENT_WORDS = {"matiere", "programme", "cour", "unite", "ue", "credit", "coefficient", "module", "formation"}

PROGRAM_FILLER_WORDS = {"semestre", "annee", "licence", "tronc", "commun", "premier", "deuxieme", "second", "niveau", "liste", "quelle", "tout"}

PROGRAM_BY_KEY = {}
for _chunk in CHUNKS:
    if _chunk.kind == "formation" and _chunk.title.startswith("Programme "):
        _mention, _specialty, _level_semester = _chunk.title[len("Programme "):].split(" — ")
        PROGRAM_BY_KEY[(_mention, _level_semester.split()[-1])] = _chunk


MENTION_PHRASES = {
    "Technologies de l'Informatique": ("systemes d information", "developpement des systemes", "technologies de l informatique"),
    "Génie Electrique": ("genie electrique", "informatique industrielle"),
    "Génie Mécanique": ("genie mecanique", "conception et fabrication"),
}


def find_mention(message):
    n = normalize(message)
    for mention, phrases in MENTION_PHRASES.items():
        if any(phrase in n for phrase in phrases):
            return mention
    words = set(tokens(message))
    scores = {mention: len(aliases & words) for mention, aliases in MENTION_ALIAS_WORDS.items()}
    best = max(scores.values())
    winners = [mention for mention, score in scores.items() if score == best]
    return winners[0] if best and len(winners) == 1 else None


def find_level(n):
    match = (re.search(r"\bl\s?([1-3])\b", n) or re.search(r"\blicence\s*([1-3])\b", n)
             or re.search(r"\b([1-3])\s*(?:er|ere|e|eme|ieme)?\s*annee\b", n))
    if match:
        return int(match.group(1))
    match = re.search(r"\b(premiere|deuxieme|seconde|troisieme)\s+annee\b", n)
    return ORDINAL_LEVELS[match.group(1)] if match else None


def find_semester_number(n):
    match = (re.search(r"\bs\s?([1-6])\b", n) or re.search(r"\bsem(?:estre)?\s*([1-6])\b", n)
             or re.search(r"\b([1-6])\s*(?:er|ere|e|eme|ieme)?\s*semestre\b", n))
    return int(match.group(1)) if match else None


def find_semester_ordinal(n):
    if re.search(r"\b(premier|1er)\s+semestre\b", n):
        return 1
    if re.search(r"\b(deuxieme|second|seconde)\s+semestre\b", n):
        return 2
    return None


def resolve_semesters(message):
    n = normalize(message)
    level, number, ordinal = find_level(n), find_semester_number(n), find_semester_ordinal(n)
    if not level and "tronc commun" in n and not number:
        level = 1
    if number:
        return [f"S{number}"], level
    if level and ordinal:
        return [f"S{2 * (level - 1) + ordinal}"], level
    if level:
        return [f"S{2 * level - 1}", f"S{2 * level}"], level
    return [], level


def mention_menu():
    return "\n".join(f"{index}. {mention}" for index, mention in enumerate(MENTION_ORDER, start=1))


def program_answer(message):
    words = set(tokens(message))
    if words & WHEN_WORDS:
        return None
    semesters, _ = resolve_semesters(message)
    n = normalize(message)
    if ("genie des procedes" in n or "genie procedes" in n or "chimie" in words) and (semesters or words & PROGRAM_INTENT_WORDS):
        return "Aucune formation de Licence « Génie des Procédés » n'est enregistrée dans le fichier des formations."
    if not semesters:
        return None
    mention = find_mention(message)
    if not mention:
        leftover = {word for word in words if word not in PROGRAM_INTENT_WORDS | PROGRAM_FILLER_WORDS
                    and not re.fullmatch(r"[sl]\d|\d+(?:er|ere|eme)?", word)}
        if words & PROGRAM_INTENT_WORDS and not leftover:
            return "De quelle filière parlez-vous ?\n" + mention_menu() + "\nRépondez par exemple : « informatique S3 »."
        return None
    parts, missing = [], []
    for semester in semesters:
        chunk = PROGRAM_BY_KEY.get((mention, semester))
        if chunk:
            parts.append(chunk.answer)
        else:
            missing.append(semester)
    if missing:
        parts.append(f"Aucune donnée enregistrée pour {mention} ({', '.join(missing)}) dans le fichier des formations.")
    return "\n\n".join(parts)


GRADE_STATES = {}

BREAKOUT = object()

GRADE_STATE_TTL = 2 * 60 * 60

GRADE_STATE_LIMIT = 500

GRADE_ARABIC_WORDS = ("معدل", "المعدل", "معدلي", "كريدي", "كريدية")

GRADE_CREDIT_ACTIONS = {"valide", "valider", "validation", "obtenu", "obtenir", "acquis", "acquerir", "calcul", "calculer", "capitaliser"}

CANCEL_PHRASES = ("annuler", "annule", "stop", "quitter", "arreter", "laisser tomber", "laisse tomber", "non merci", "cancel")

RESTART_PHRASES = ("recommencer", "recommence", "nouveau calcul", "depuis le debut", "reprendre a zero")

SKIP_PHRASES = ("passer", "passe", "ignorer", "ignore", "suivant", "aucune", "nsp", "pas de note", "pas encore", "dispense",
                "je ne sais pas", "je sais pas", "pas passe", "pas cette matiere", "n'ai pas")

BACK_PHRASES = ("retour", "precedent", "revenir", "corriger")

NUMBER_PATTERN = re.compile(r"(?<![\d.,])(\d{1,2}(?:[.,]\d{1,3})?)(?:\s*/\s*20)?(?![\d])")

GRADE_INTRO = (
    "Je peux calculer votre moyenne et vos crédits ! 🧮 Dites-moi votre filière et votre semestre "
    "(par exemple : « informatique S3 »), puis je vous demanderai votre note dans chaque matière, une par une.\n"
    "Filières disponibles :\n"
)


def is_grade_request(message):
    n = normalize(message)
    words = set(tokens(message))
    if any(phrase in message for phrase in GRADE_ARABIC_WORDS):
        return True
    if words & {"moyenne", "moyen"}:
        return True
    if "credit" in words and words & GRADE_CREDIT_ACTIONS:
        return True
    return bool(words & {"calcul", "calculer", "calculatrice", "simulateur", "simulation"} and words & {"note", "semestre", "resultat", "credit"})


def purge_grade_states(now):
    for key in [key for key, state in GRADE_STATES.items() if now - state["time"] > GRADE_STATE_TTL]:
        del GRADE_STATES[key]
    while len(GRADE_STATES) > GRADE_STATE_LIMIT:
        del GRADE_STATES[min(GRADE_STATES, key=lambda key: GRADE_STATES[key]["time"])]


def available_semesters(mention):
    return sorted({semester for (name, semester) in PROGRAM_BY_KEY if name == mention})


def grade_subjects(mention, semester):
    rows = [row for row in FORMATION_ROWS if row["Mention"] == mention and row["Semestre"] == semester]
    return [
        {
            "unit": row["Unite_Enseignement"], "subject": row["Matiere"],
            "coef": float(row["Coefficient"]), "unit_credits": float(row["Credits_UE"]),
            "unit_coef": float(row["Coefficient_UE"]), "specialty": row["Specialite"],
        }
        for row in rows
    ]


def parse_grade(message):
    match = NUMBER_PATTERN.search(message.replace("،", ","))
    if not match:
        return None
    return float(match.group(1).replace(",", "."))


def grade_question(state):
    subjects = state["subjects"]
    item = subjects[state["index"]]
    return (
        f"Matière {state['index'] + 1}/{len(subjects)} — {item['subject']}\n"
        f"(unité : {item['unit']}, coefficient {format_number(item['coef'])})\n"
        "Quelle est votre note sur 20 ?"
    )


def start_notes(state):
    subjects = grade_subjects(state["mention"], state["semester"])
    if not subjects:
        GRADE_STATES.pop(state["id"], None)
        return f"Je n'ai aucune matière enregistrée pour {state['mention']} ({state['semester']}) dans le fichier des formations."
    state.update(step="notes", subjects=subjects, index=0, notes={})
    return (
        f"D'accord : {state['mention']} — {subjects[0]['specialty']} — {state['semester']} "
        f"({len(subjects)} matières).\n"
        "Je vous demande votre note dans chaque matière. Écrivez « passer » pour ignorer une matière, "
        "« retour » pour corriger la précédente ou « annuler » pour arrêter.\n\n" + grade_question(state)
    )


def two_decimals(value):
    return f"{value:.2f}".replace(".", ",")


def grade_result(state):
    subjects, notes = state["subjects"], state["notes"]
    units = {}
    for index, item in enumerate(subjects):
        units.setdefault(item["unit"], {"items": [], "credits": item["unit_credits"], "coef": item["unit_coef"]})
        units[item["unit"]]["items"].append((item, notes.get(index)))
    lines, total_weight, weighted_sum, acquired, base, validated = [], 0.0, 0.0, 0.0, 0.0, 0.0
    skipped = []
    for unit, data in units.items():
        graded = [(item, note) for item, note in data["items"] if note is not None]
        skipped += [item["subject"] for item, note in data["items"] if note is None]
        if not graded:
            lines.append(f"• {unit} — non calculée (aucune note)")
            continue
        average = sum(note * item["coef"] for item, note in graded) / sum(item["coef"] for item, note in graded)
        total_weight += data["coef"]
        weighted_sum += average * data["coef"]
        base += data["credits"]
        status = "validée" if average >= 10 else "non validée"
        if average >= 10:
            validated += data["credits"]
        lines.append(f"• {unit} — moyenne {two_decimals(average)} ({status}, {format_number(data['credits'])} crédits)")
        lines.extend(f"    – {item['subject']} : {format_number(note)}" for item, note in graded)
    if not total_weight:
        return "Aucune note n'a été saisie, le calcul n'est pas possible. Tapez « moyenne » pour recommencer."
    semester_average = weighted_sum / total_weight
    if semester_average >= 10:
        acquired = base
        verdict = f"Semestre validé (moyenne ≥ 10) : tous les crédits sont acquis par compensation."
    else:
        acquired = validated
        verdict = "Semestre non validé (moyenne < 10) : seules les unités avec une moyenne ≥ 10 donnent leurs crédits."
    text = (
        f"Résultat — {state['mention']} — {state['semester']}\n" + "\n".join(lines)
        + f"\n\nMoyenne du semestre : {two_decimals(semester_average)} / 20\n"
        f"Crédits obtenus : {format_number(acquired)} / {format_number(base)}\n{verdict}"
    )
    if skipped:
        text += f"\nMatières ignorées : {', '.join(skipped)} (la moyenne est calculée sur les matières renseignées)."
    text += ("\n\nCalcul indicatif selon la règle LMD usuelle (moyenne d'unité pondérée par les coefficients des matières, "
             "moyenne du semestre pondérée par les coefficients des unités). Le règlement officiel de l'institut fait foi."
             "\nTapez « moyenne » pour un nouveau calcul.")
    return text


def grade_ask_missing(state):
    if not state.get("mention"):
        state["step"] = "mention"
        return "Quelle est votre filière ?\n" + mention_menu() + "\n(écrivez le nom ou le numéro)"
    if not state.get("semester"):
        state["step"] = "semester"
        options = available_semesters(state["mention"])
        if state.get("level"):
            options = [option for option in options if option in (f"S{2 * state['level'] - 1}", f"S{2 * state['level']}")] or options
        return f"Quel semestre ? ({', '.join(options)})"
    return start_notes(state)


def grade_flow(message, session_id):
    now = dt.datetime.now().timestamp()
    purge_grade_states(now)
    state = GRADE_STATES.get(session_id)
    starting = is_grade_request(message)
    if not state and not starting:
        return None
    n = normalize(message)
    words = n.split()

    if state and any(phrase in n for phrase in CANCEL_PHRASES):
        GRADE_STATES.pop(session_id, None)
        return "D'accord, calcul annulé. Tapez « moyenne » quand vous voulez le reprendre."

    restart = state and any(phrase in n for phrase in RESTART_PHRASES)
    if not state or restart or (starting and state["step"] != "notes"):
        previous = state if (state and not restart) else None
        state = {"id": session_id, "step": "mention", "mention": None, "semester": None, "level": None, "time": now}
        if previous:
            state.update(mention=previous.get("mention"), semester=previous.get("semester"), level=previous.get("level"))
        GRADE_STATES[session_id] = state
        fresh = True
    else:
        fresh = False
    state["time"] = now

    if state["step"] in ("mention", "semester"):
        mention = find_mention(message)
        if not mention and state["step"] == "mention":
            digit = re.fullmatch(r"\D*([1-4])\D*", n)
            if digit and len(words) <= 3:
                mention = MENTION_ORDER[int(digit.group(1)) - 1]
        semesters, level = resolve_semesters(message)
        if mention:
            if mention != state.get("mention"):
                state["semester"] = None
            state["mention"] = mention
        if level:
            state["level"] = level
        if len(semesters) == 1:
            state["semester"] = semesters[0]
        elif state["step"] == "semester":
            digit = re.fullmatch(r"\D*([1-6])\D*", n)
            if digit and len(words) <= 3:
                state["semester"] = f"S{digit.group(1)}"
        if state.get("mention") and state.get("semester") and state["semester"] not in available_semesters(state["mention"]):
            semester = state["semester"]
            state["semester"] = None
            return (f"Je n'ai pas de matières enregistrées pour {state['mention']} en {semester}. "
                    f"Semestres disponibles : {', '.join(available_semesters(state['mention']))}.")
        if not fresh and not mention and not len(semesters) == 1 and state["step"] in ("mention", "semester"):
            if state["step"] == "semester" and state.get("semester"):
                pass
            elif len(tokens(message)) >= 3:
                GRADE_STATES.pop(session_id, None)
                return BREAKOUT
            else:
                prefix = "Je n'ai pas reconnu votre réponse. "
                return prefix + grade_ask_missing(state)
        if fresh and not state.get("mention") and not state.get("semester"):
            state["step"] = "mention"
            return GRADE_INTRO + mention_menu()
        return grade_ask_missing(state)

    subjects = state["subjects"]
    if any(phrase in n for phrase in BACK_PHRASES) and parse_grade(message) is None:
        if state["index"] == 0:
            return "Vous êtes déjà à la première matière.\n" + grade_question(state)
        state["index"] -= 1
        state["notes"].pop(state["index"], None)
        return grade_question(state)

    value = parse_grade(message)
    if value is None:
        if any(phrase in n for phrase in SKIP_PHRASES):
            state["index"] += 1
        elif len(tokens(message)) >= 2 or "?" in message:
            GRADE_STATES.pop(session_id, None)
            return BREAKOUT
        else:
            return "Donnez une note entre 0 et 20 (exemple : 12,5), ou « passer » pour ignorer cette matière.\n" + grade_question(state)
    else:
        if not 0 <= value <= 20:
            return "La note doit être comprise entre 0 et 20.\n" + grade_question(state)
        state["notes"][state["index"]] = value
        state["index"] += 1

    if state["index"] >= len(subjects):
        result = grade_result(state)
        GRADE_STATES.pop(session_id, None)
        return result
    return grade_question(state)


OUT_OF_SCOPE = {
    "emploi du temps": "l'emploi du temps",
    "cantine": "la cantine",
    "restaurant universitaire": "le restaurant universitaire",
    "foyer": "le foyer",
    "transport": "le transport",
    "hebergement": "l'hébergement",
}


intent_vectorizer = CountVectorizer(preprocessor=normalize, token_pattern=r"\S+")

INTENT_PHRASES = [phrase for phrases in TRAINING_PHRASES.values() for phrase in phrases]

INTENT_LABELS = [intent for intent, phrases in TRAINING_PHRASES.items() for _ in phrases]

intent_model = DecisionTreeClassifier(random_state=42)
intent_model.fit(intent_vectorizer.fit_transform(INTENT_PHRASES), INTENT_LABELS)


def smalltalk_intent(message):
    words = normalize(message).split()
    if not words or len(words) > SMALLTALK_MAX_WORDS:
        return None
    known = sum(1 for word in words if word in intent_vectorizer.vocabulary_)
    if known / len(words) < SMALLTALK_MIN_KNOWN_RATIO:
        return None
    return intent_model.predict(intent_vectorizer.transform([message]))[0]


def contact_hint():
    values = {row.get("Rubrique"): row.get("Valeur") for row in INFO_ROWS}
    parts = [f"{label} : {values[key]}" for label, key in (("Téléphone", "Téléphone"), ("Email", "Email")) if values.get(key)]
    return ("\nPour plus d'informations, contactez l'institut — " + " ; ".join(parts)) if parts else ""


def out_of_scope_answer(message):
    n = normalize(message)
    for phrase, label in OUT_OF_SCOPE.items():
        if phrase in n:
            return f"Je n'ai pas d'information sur {label} dans les données de l'institut." + contact_hint()
    return None


def build_vocabulary():
    words = set()
    for chunk in CHUNKS:
        words.update(normalize(f"{chunk.title} {chunk.answer}").split())
        words.update(chunk.search.split())
    for text in (*STOP_WORDS, *MONTHS, *WEEKDAYS, *INFO_KEYWORDS.values(), *SYNONYM_GROUPS, *MENTION_ALIASES.values(),
                 *CANCEL_PHRASES, *RESTART_PHRASES, *SKIP_PHRASES, *BACK_PHRASES, *INTENT_PHRASES,
                 "moyenne calculer calcul credit credits coefficient matiere matieres semestre filiere note notes"):
        words.update(normalize(text).split())
    return {word for word in words if len(word) >= 3 and not word.isdigit()}


VOCABULARY = build_vocabulary()

VOCABULARY_BY_INITIAL = {}
for _word in VOCABULARY:
    VOCABULARY_BY_INITIAL.setdefault(_word[0], []).append(_word)

TYPO_CUTOFF = 0.84

TYPO_PATTERN = re.compile(r"[A-Za-zÀ-ÿ]{5,}")


def correct_typos(message):
    """Corrige les fautes de frappe en s'appuyant uniquement sur le vocabulaire des fichiers CSV."""
    def fix(match):
        word = normalize(match.group(0))
        if not word or word in VOCABULARY or word.rstrip("s") in VOCABULARY or word.endswith("s") and word[:-1] in VOCABULARY:
            return match.group(0)
        pool = VOCABULARY_BY_INITIAL.get(word[0], [])
        close = difflib.get_close_matches(word, pool, n=1, cutoff=TYPO_CUTOFF)
        return close[0] if close else match.group(0)
    return TYPO_PATTERN.sub(fix, message)


INTERRUPTED_NOTE = "\n\n(Calcul de moyenne interrompu — écrivez « moyenne » pour le reprendre.)"


def answer(message, session_id="default"):
    message = correct_typos(message)
    reply = grade_flow(message, session_id)
    if reply is BREAKOUT:
        text, intent, source, score = answer_core(message)
        return text + INTERRUPTED_NOTE, intent, source, score
    if reply:
        return reply, "grade", FORMATIONS_SOURCE, None
    return answer_core(message)


def answer_core(message):
    if not re.search(r"[^\W\d_]", message):
        return EMPTY_MESSAGE_REPLY, "unknown", None, None

    reply = who_answer(message)
    if reply:
        return reply, "council", COUNCIL_SOURCE, None

    query_words = set(tokens(message))
    if query_words and query_words <= IDENTITY_WORDS and IDENTITY_CARD:
        return IDENTITY_CARD.answer, "info", IDENTITY_CARD.source, None

    reply = date_answer(message)
    if reply:
        return reply, "calendar", CALENDAR_SOURCE, None

    reply = program_answer(message)
    if reply:
        return reply, "formation", FORMATIONS_SOURCE, None

    intent = smalltalk_intent(message)
    if intent:
        return REPLIES[intent], intent, None, None

    reply = out_of_scope_answer(message)
    if reply:
        return reply, "unknown", None, None

    results = [(chunk, score, coverage(message, chunk)) for chunk, score in retrieve(message)]
    results.sort(key=lambda item: item[1] * (0.3 + 0.7 * item[2]), reverse=True)
    if results and results[0][1] >= MIN_SCORE and results[0][2] >= MIN_COVERAGE:
        best, score, _ = results[0]
        text = best.answer
        if len(results) > 1:
            second, second_score, second_coverage = results[1]
            if second_score >= 0.9 * score and second_coverage >= MIN_COVERAGE and second.source != best.source:
                text += "\n\n" + second.answer
        return text, best.kind, best.source, round(score, 3)

    fallback = keyword_fallback(message)
    if fallback:
        return fallback.answer, fallback.kind, fallback.source, None

    return NOT_FOUND_REPLY + contact_hint() + "\n" + SUGGESTIONS, "unknown", None, None


app = FastAPI(title="Assistant ISET Kébili")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


@app.post("/chat")
async def chat(request: ChatRequest):
    message = request.message.strip()
    if not message:
        return {"reply": EMPTY_MESSAGE_REPLY}
    reply, intent, source, score = answer(message, request.session_id[:64])
    return {"reply": reply, "intent": intent, "source": source, "score": score}


@app.get("/health")
async def health():
    return {"chunks": len(CHUNKS), "events": len(EVENTS), "data_dir": str(DATA_DIRECTORY)}
