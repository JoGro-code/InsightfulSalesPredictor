from InsightfulSalesPredictor.controllers.prediction_controller import run_training, run_prediction
import sys

def main():
    """
    Hauptfunktion zur Ausführung der Anwendung.
    Unterstützt zwei Modi: Training und Vorhersage.
    """
    if len(sys.argv) != 2:
        print("Falsche Anzahl an Argumenten. Benutzung: main.py [train|predict]")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == 'train':
        print("Trainingsmodus gestartet...")
        run_training()
        print("Training abgeschlossen.")
    elif mode == 'predict':
        print("Vorhersagemodus gestartet...")
        run_prediction()
        print("Vorhersagen wurden erfolgreich erstellt und gespeichert.")
    else:
        print("Unbekannter Modus. Bitte wählen Sie 'train' oder 'predict'.")
        sys.exit(1)

if __name__ == "__main__":
    main() 

#python main.py train
#python main.py predict
#python -m main train
#python -m main predict
#python -m controllers.prediction_controller
