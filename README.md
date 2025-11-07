# CarPriceML - Car Price Prediction ML Pipeline

Un pipeline complet de Machine Learning pour estimer le prix des voitures d'occasion en dirhams marocains (MAD).

## 🚀 Fonctionnalités

- ✅ Pipeline complet de ML (préprocessing, entraînement, prédiction)
- ✅ API FastAPI avec endpoints `/predict` et `/health`
- ✅ Interface Streamlit pour interagir avec le modèle
- ✅ Dockerisation complète avec Docker Compose
- ✅ Redis pour la mise en cache des prédictions
- ✅ Prometheus et Grafana pour le monitoring

## 📋 Prérequis

- Python 3.11+
- Docker et Docker Compose
- (Optionnel) pip pour installation locale

## 🛠️ Installation et utilisation

### Méthode 1: Docker Compose (Recommandé)

1. **Cloner le projet** (si ce n'est pas déjà fait)

2. **S'assurer que le dataset est présent**

   ```bash
   ls data/car-details.csv
   ```

3. **Entraîner le modèle** (première fois seulement)

   ```bash
   # Localement ou dans un conteneur
   python -m src.train --n_estimators 100
   ```

   Le modèle sera sauvegardé dans `models/rf_model.joblib`

4. **Lancer tous les services avec Docker Compose**

   ```bash
   docker-compose up --build
   ```

5. **Accéder aux services**:
   - Frontend Streamlit: http://localhost:8501
   - API Backend: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000 (admin/admin)

### Méthode 2: Installation locale

source c:/Users/lenovo/Documents/CarPriceML/.venv311/Scripts/activate

1. **Installer les dépendances**

   ```bash
   pip install -r requirements.txt
   ```

2. **Entraîner le modèle**

   ```bash
   python -m src.train --n_estimators 100
   ```

3. **Lancer le backend**

   ```bash
   python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
   ```
   ```powershell
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000
   ```


4. **Lancer le frontend** (dans un autre terminal)
   ```bash
   streamlit run src/frontend/app.py
   ```

## 📁 Structure du projet

```
CarPriceML/
├── src/
│   ├── api/
│   │   └── main.py          # API FastAPI
│   ├── frontend/
│   │   └── app.py           # Interface Streamlit
│   ├── config.py            # Configuration
│   ├── preprocessing.py     # Préprocessing des données
│   ├── pipeline.py          # Pipeline ML
│   └── train.py             # Script d'entraînement
├── data/
│   └── car-details.csv      # Dataset
├── models/                  # Modèles sauvegardés
├── tests/
│   └── test_api.py          # Tests unitaires
├── docker/
│   └── prometheus.yml       # Configuration Prometheus
├── Dockerfile               # Dockerfile backend
├── Dockerfile.frontend      # Dockerfile frontend
├── docker-compose.yml       # Orchestration Docker
└── requirements.txt         # Dépendances Python
```

## 🧪 Tests

```bash
pytest tests/
```

## 📊 Endpoints API

- `GET /health` - Vérifier le statut de l'API
- `POST /predict` - Prédire le prix d'une voiture
- `GET /metrics` - Métriques Prometheus

### Exemple de requête `/predict`

```json
{
  "company": "Maruti",
  "model": "Swift",
  "fuel": "Petrol",
  "seller_type": "Individual",
  "transmission": "Manual",
  "km_driven": 50000,
  "mileage_mpg": 45.0,
  "engine_cc": 1200.0,
  "max_power_bhp": 80.0,
  "seats": 5
}
```

## 🐳 Services Docker

- **backend**: API FastAPI (port 8000)
- **frontend**: Interface Streamlit (port 8501)
- **redis**: Cache Redis (port 6379)
- **prometheus**: Monitoring (port 9090)
- **grafana**: Visualisation (port 3000)

## 📝 Notes

- Les prix dans le dataset original sont convertis de INR vers MAD (1 INR = 0.10 MAD)
- Le modèle utilise un RandomForestRegressor
- Les prédictions sont mises en cache dans Redis pendant 1 heure
- Les métriques Prometheus sont disponibles sur `/metrics`

## 🔧 Configuration

Les variables d'environnement peuvent être modifiées dans `.env`:

```
API_HOST=0.0.0.0
API_PORT=8000
REDIS_URL=redis://redis:6379/0
```

## 👨‍💻 Auteur

Achraf CHAIR | MLOps & IA Project 🚀
