import json


# lecture d'un fichier json
def read_json(fichier):
    with open(fichier) as nom_fichier:
        donnee = json.load(nom_fichier)
    nom_fichier.close()
    return donnee


# ecriture d'un fichier json
def write_json(fichier, donnee):
    with open(fichier, 'w') as nom_fichier:
        json.dump(donnee, nom_fichier)
