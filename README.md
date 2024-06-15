# Dynamic-Risk-Assessment-System

## Description
The objective of this project is to develop, deploy, and monitor a machine learning model for assessing the attrition risk of the company's clients. The project includes setting up processes for model re-training, re-deployment, monitoring, and reporting.

## Prerequisites

- Python 3 required.
- A Linux environment may be necessary, which can be set up within Windows using WSL.
  
## Dependencies

All project dependencies are listed in the `requirements.txt` file.

## Installation

To install the dependencies, use pip. It's recommended to install them in a separate virtual environment:

```bash
pip install -r requirements.txt
```

## Project Structure
- **data/**
  - **ingesteddata/**
    - `finaldata.csv`: Final processed data file.
    - `ingestedfiles.txt`: List of ingested files.
  - **practicedata/**: Data used for initial practice mode.
    - `dataset1.csv`, `dataset2.csv`: Practice dataset files.
  - **sourcedata/**: Data used for production mode.
    - `dataset3.csv`, `dataset4.csv`: Production dataset files.
  - **testdata/**: Test data.
    - `testdata.csv`: Dataset for testing.

- **model/**: Contains models and related artifacts.
  - **models/**: Models and reports for production mode.
    - `apireturns.txt`: API returns.
    - `confusionmatrix.png`: Confusion matrix.
    - `latestscore.txt`: Latest model score.
    - `summary_report.pdf`: Summary report in PDF.
    - `trainedmodel.pkl`: Trained model.
  - **practicemodels/**: Models and reports for practice mode.
    - `apireturns2.txt`: API returns.
    - `confusionmatrix2.png`: Confusion matrix.
    - `latestscore.txt`: Latest model score.
    - `summary_report.pdf`: Summary report in PDF.
    - `trainedmodel.pkl`: Trained model.
  - **production_deployment/**: Deployment artifacts for production.
    - `ingestedfiles.txt`: Ingested files.
    - `latestscore.txt`: Latest score.
    - `trainedmodel.pkl`: Trained model.

- **src/**: Project source code.
  - `apicalls.py`: Script for API calls.
  - `app.py`: Flask application.
  - `config.py`: Configuration file.
  - `deployment.py`: Model deployment script.
  - `diagnostics.py`: Model diagnostics script.
  - `fullprocess.py`: Process automation script.
  - `ingestion.py`: Data ingestion script.
  - `reporting.py`: Report generation script.
  - `scoring.py`: Model scoring script.
  - `training.py`: Model training script.
  - `wsgi.py`: WSGI file for deployment.

- `config.json`: JSON configuration file.
- `cronjob.txt`: Cronjob for automation.
- `README.md`: Project README file.
- `requirements.txt`: Project dependencies file.