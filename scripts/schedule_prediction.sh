# Navigieren zum Projektverzeichnis
cd /Users/janine.ullrich/Desktop/python/InsightfulSalesPredictor/config

# Aktivierung der virtuellen Umgebung
source venv/bin/activate

# Ausführen des Python-Skripts für das Training und die Vorhersage
echo "Starte den Vorhersageprozess: $(date)"
python main.py
echo "Vorhersageprozess abgeschlossen: $(date)"

# Deaktivierung der virtuellen Umgebung
deactivate

#crontab -e
#0 3 * * * /pfad/zum/InsightfulSalesPredictor/scripts/schedule_prediction.sh
