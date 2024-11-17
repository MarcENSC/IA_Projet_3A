import logging

# Configuration du logger
log_file = 'game_log.txt'

logging.basicConfig(
    level=logging.INFO,  # Le niveau de base des messages à afficher (INFO, DEBUG, WARNING, etc.)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Format du message de log
    handlers=[
        logging.StreamHandler(),  # Affiche les logs dans la console
        logging.FileHandler(log_file)  # Enregistre les logs dans un fichier
    ]
)

def clear():
    """ Vider le contenu du fichier de log sans le supprimer """
    with open(log_file, 'w'):  # Ouvre le fichier en mode écriture et ferme immédiatement
        pass  # Ne rien faire, juste vider le fichier
    logging.info("Log file cleared.")  # Confirmation dans les logs

# Exemple de fonction pour logguer un message
def log(message, level=logging.INFO):
    """ Log une action à un niveau donné """
    if level == logging.DEBUG:
        logging.debug(message)
    elif level == logging.INFO:
        logging.info(message)
    elif level == logging.WARNING:
        logging.warning(message)
    elif level == logging.ERROR:
        logging.error(message)
    elif level == logging.CRITICAL:
        logging.critical(message) 
