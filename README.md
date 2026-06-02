# Credit Risk Probability Model

**Repository**: Bati Bank Credit Scoring System  
**Implementation**: End-to-end machine learning pipeline with MLOps

---

## Project Structure

```
credit-risk-model/
├── .github/
│   └── workflows/
│       └── ci.yml                    # CI/CD pipeline
├── data/
│   ├── data.csv                      # Transaction data
│   └── Xente_Variable_Definitions.csv # Feature definitions
├── notebooks/
│   └── eda.ipynb                     # Exploratory data analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py            # Feature engineering pipeline
│   ├── train.py                      # Model training & tracking
│   ├── predict.py                    # Inference utilities
│   └── api/
│       ├── main.py                   # FastAPI application
│       └── pydantic_models.py        # Request/response schemas
├── tests/
│   └── test_data_processing.py       # Unit tests
├── Dockerfile                         # Container image
├── docker-compose.yml                # Local deployment
├── requirements.txt                  # Python dependencies
└── .gitignore                        # Git rules
```

---

## Technology Stack

- **Data Processing**: pandas, numpy, scikit-learn
- **Feature Engineering**: xverse (WoE/IV), scikit-learn Pipeline
- **ML Modeling**: scikit-learn, XGBoost, LightGBM
- **Experiment Tracking**: MLflow
- **API**: FastAPI, Pydantic, uvicorn
- **Containerization**: Docker, docker-compose
- **CI/CD**: GitHub Actions
- **Testing**: pytest

---

## Setup

```bash
# Clone repository
git clone https://github.com/Ertibn/Credit-risk-model.git
cd Credit-risk-model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running Experiments

```bash
# Run EDA
jupyter notebook notebooks/eda.ipynb

# Train models with MLflow tracking
python src/train.py

# Start API service
uvicorn src.api.main:app --reload

# Docker deployment
docker-compose up
```

---

## CI/CD Pipeline

GitHub Actions workflow in `.github/workflows/ci.yml`:
- Linting: flake8, black
- Testing: pytest
- Docker build validation

---

## License

10 Academy Week 4 Challenge


