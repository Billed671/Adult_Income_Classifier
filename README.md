ADULT INCOME MLOPS PIPELINE
============================

This repository contains an automated machine learning pipeline for the Adult Income dataset. The pipeline has three stages: data engineering, model engineering, and deployment. It runs automatically every five minutes with Apache Airflow. The model is served through a FastAPI application, and a Streamlit web application lets the user enter data and get a prediction. Both the API and the app run in separate Docker containers.

The goal of this project is to show a simple but complete MLOps workflow. The dataset is small, the models are standard gradient boosting models, and the focus is on the pipeline working from start to finish without manual steps.


WHAT THE APPLICATION DOES
=========================

The application predicts whether a person earns more than 50 thousand dollars per year. The input is a set of personal and work related features, such as age, education, occupation, working hours, and so on. The output is a class label (either "<=50K" or ">50K") and a probability between 0 and 1.

The user opens the Streamlit web page in a browser. The page shows several input fields. After filling the fields, the user presses the "Predict" button. The Streamlit app sends the data to the FastAPI service using an HTTP POST request. The API loads the trained model, makes a prediction, and sends the result back. The Streamlit app then shows the predicted label and the probability on the page.

The API and the web app are two separate services. They run in two separate Docker containers. The Streamlit container talks to the FastAPI container through the internal Docker network. This is the same pattern that is used in real production systems, where the front end and the model service are deployed separately.


PIPELINE STAGES
===============

Stage 1. Data Engineering
-------------------------

The script code/datasets/data_engineering.py downloads the raw Adult Income dataset from the UCI Machine Learning Repository. Then it cleans the data. Cleaning means the following steps:

- missing values, which are stored as "?" in the raw file, are removed;
- rows with duplicate values are removed;
- outliers in the "age" and "hours-per-week" columns are removed using the interquartile range method;
- the target column "income" is converted to a binary value (0 for "<=50K", 1 for ">50K").

After cleaning, the script splits the data into a training set (80 percent) and a test set (20 percent). The split keeps the same proportion of the target classes in both parts. The two files are saved to the data/processed folder as train.csv and test.csv.

Stage 2. Model Engineering
--------------------------

The script code/models/model_engineering.py loads the training and test files from Stage 1. Then it builds a preprocessing pipeline. Numerical features are scaled with StandardScaler. Categorical features are encoded with OneHotEncoder.

After preprocessing, three gradient boosting models are trained and tuned with Optuna:

- CatBoost
- XGBoost
- LightGBM

For each model, Optuna runs a number of trials and searches for the best hyperparameters. The tuning metric is ROC AUC on the test set. Each trial is logged to MLflow together with its parameters and metrics.

After tuning, the best model is refit on the full training set, evaluated on the test set, and saved to models/best_model.joblib. The metrics for all three models are saved to models/metrics.json. The best model is the one with the highest ROC AUC.

Stage 3. Deployment
-------------------

Stage 3 uses Docker and Docker Compose. Two Docker images are built:

- the API image, based on FastAPI and Uvicorn;
- the web app image, based on Streamlit.

The Docker Compose file code/deployment/docker-compose.yml starts both containers. The FastAPI container exposes port 8000. The Streamlit container exposes port 8501. The trained model file from Stage 2 is mounted into the API container as a read only volume. The Streamlit container sends requests to the API using the internal Docker hostname "api".

Automation with Airflow
-----------------------

The Airflow DAG services/airflow/dags/ml_pipeline_dag.py connects all three stages. It has three tasks:

1. data_engineering, which runs the Stage 1 script;
2. model_engineering, which runs the Stage 2 script;
3. deployment, which runs "docker compose up -d --build" inside code/deployment.

The tasks run in order. The DAG is scheduled with the cron expression "*/5 * * * *", so it runs every five minutes. If a run takes longer than five minutes, the interval can be increased in the DAG file.


REPOSITORY STRUCTURE
====================

The repository has the following layout.

code/datasets/data_engineering.py
    Stage 1 script. Loads, cleans, and splits the data.

code/models/model_engineering.py
    Stage 2 script. Trains, evaluates, and packages the model.

code/deployment/api/main.py
    FastAPI application. Loads the model and serves predictions.

code/deployment/api/Dockerfile
    Dockerfile for the API container.

code/deployment/api/requirements.txt
    Python dependencies for the API container.

code/deployment/app/app.py
    Streamlit application. Provides the input form and shows predictions.

code/deployment/app/Dockerfile
    Dockerfile for the Streamlit container.

code/deployment/app/requirements.txt
    Python dependencies for the Streamlit container.

code/deployment/docker-compose.yml
    Docker Compose file that starts both containers.

services/airflow/dags/ml_pipeline_dag.py
    Airflow DAG that connects all three stages and runs every five minutes.

data/raw/
    Raw dataset files. The dataset is downloaded here by Stage 1.

data/processed/
    Cleaned train and test files. Written by Stage 1.

models/
    Trained model file and metrics. Written by Stage 2.

requirements.txt
    Python dependencies for Stages 1 and 2.

README.txt
    This file.


REQUIREMENTS
============

Before running the pipeline, make sure the following software is installed on your computer.

- Python 3.10 or newer. You can check the version with "python --version".
- pip. It is usually installed together with Python.
- Git. You can check with "git --version".
- Docker Desktop. You can check with "docker --version".
- Docker Compose. It is included in recent versions of Docker Desktop. You can check with "docker compose version".

Docker Desktop must be running before you start Stage 3. On Windows and macOS you should open the Docker Desktop application and wait until the icon turns green or shows that the engine is running. On Linux you can start the Docker daemon with "sudo systemctl start docker".


STEP BY STEP GUIDE TO RUN THE PIPELINE LOCALLY
=============================================

Step 1. Clone the repository
----------------------------

Open a terminal and run:

    git clone https://github.com/<your-username>/adult-income-mlops.git
    cd adult-income-mlops

Replace <your-username> with your GitHub username. After this command, the working directory of your terminal is the root of the repository.

Step 2. Create a virtual environment
------------------------------------

A virtual environment keeps the project dependencies separate from the rest of the system. Run:

    python -m venv .venv

Then activate it:

On macOS or Linux:

    source .venv/bin/activate

On Windows with Git Bash:

    source .venv/Scripts/activate

On Windows with PowerShell:

    .venv\Scripts\Activate.ps1

After activation, your shell prompt usually shows "(.venv)" at the beginning. All further Python commands should be run inside this environment.

Step 3. Install the dependencies
--------------------------------

Run:

    pip install --upgrade pip
    pip install -r requirements.txt

This installs pandas, numpy, scikit-learn, CatBoost, XGBoost, LightGBM, Optuna, MLflow, and joblib. The first installation can take several minutes.

Step 4. Run Stage 1 (data engineering)
--------------------------------------

Run:

    python code/datasets/data_engineering.py

What happens:

- The script downloads the Adult Income dataset from the UCI repository and saves it to data/raw/adult.csv. If the file is already there, the download is skipped.
- The script cleans the data and removes missing values, duplicates, and outliers.
- The script splits the data into a training set and a test set.
- The two files are saved to data/processed/train.csv and data/processed/test.csv.

You can check the result with:

    ls -lh data/processed
    head -2 data/processed/train.csv

Step 5. Run Stage 2 (model engineering)
---------------------------------------

Run:

    python code/models/model_engineering.py

What happens:

- The script loads the training and test data.
- It builds the preprocessing pipeline.
- It tunes CatBoost, XGBoost, and LightGBM with Optuna. Each model runs several trials. The default is 15 trials per model. You can increase this number in the script if you want better quality, but tuning will then take more time.
- Each trial is logged to MLflow in the local mlruns folder.
- The best model is saved to models/best_model.joblib.
- All metrics are saved to models/metrics.json.

Depending on your computer, this step can take between five and twenty minutes. When the script finishes, you should see a line similar to:

    Best model: xgboost (AUC=0.91) saved to models/best_model.joblib

Optional. You can view the MLflow results in a browser:

    mlflow ui --port 5000

Then open http://localhost:5000. Press Ctrl+C in the terminal to stop the MLflow UI when you are done.

Step 6. Run Stage 3 (deployment)
--------------------------------

Make sure Docker Desktop is running. Then run:

    cd code/deployment
    docker compose up -d --build

The first build takes several minutes because the images need to be created and the Python packages need to be installed inside them. When the build is finished, two containers start:

- adult_api on port 8000;
- adult_app on port 8501.

You can check the status with:

    docker compose ps

Both containers should show the state "running".

Step 7. Use the web application
-------------------------------

Open a browser and go to:

    http://localhost:8501

The Streamlit page shows the input form. Fill the fields and press the "Predict" button. The prediction appears below the form as a label and a probability. The label is either "<=50K" or ">50K".

If you want to test the API directly, open:

    http://localhost:8000/docs

This is the automatic FastAPI documentation page. You can send a test request from there.

Step 8. Stop the containers
---------------------------

To stop the API and the web app, run:

    docker compose down

This stops and removes the containers. The built images stay on disk, so the next "docker compose up" is faster.

Step 9. Run the full pipeline with Airflow
------------------------------------------

Airflow is used to run all three stages automatically every five minutes. The easiest way is to install Airflow locally with pip.

First, go back to the root of the repository:

    cd ../..

Set the Airflow home directory to the services/airflow folder inside the repository:

On macOS or Linux:

    export AIRFLOW_HOME=$(pwd)/services/airflow
    export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/services/airflow/dags

On Windows with PowerShell:

    $env:AIRFLOW_HOME = "$(Get-Location)/services/airflow"
    $env:AIRFLOW__CORE__DAGS_FOLDER = "$(Get-Location)/services/airflow/dags"

Install Airflow with a constraint file to avoid version conflicts:

    pip install "apache-airflow==2.9.3" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.3/constraints-3.11.txt"

Initialize the Airflow database:

    airflow db init

Create an admin user:

    airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com

Before starting Airflow, open the DAG file services/airflow/dags/ml_pipeline_dag.py and check the PROJECT_DIR variable. It must point to the absolute path of the repository on your computer. For example, on macOS it might look like this:

    PROJECT_DIR = "/Users/yourname/adult-income-mlops"

On Windows it might look like this:

    PROJECT_DIR = "C:/Users/yourname/adult-income-mlops"

Save the file after the change.

Now start Airflow:

    airflow standalone

This command starts the webserver, the scheduler, and the triggerer in one process. When it starts, it prints an admin username and password in the terminal. Use these to log in.

Open a browser and go to:

    http://localhost:8080

Find the DAG with the id "adult_income_pipeline". Turn it on with the toggle on the left. The DAG will start running every five minutes. You can click on the DAG name to see the task states. Green means success. If a task fails, click on it and open the log to see what happened.

Keep in mind that the deployment task inside the DAG runs "docker compose up", so Docker Desktop must be running while Airflow runs the DAG.

To stop Airflow, go back to the terminal and press Ctrl+C.


COMMON PROBLEMS AND SOLUTIONS
=============================

Problem: "docker: command not found".
Solution: Docker Desktop is not installed or not in the PATH. Install Docker Desktop and restart the terminal.

Problem: "Cannot connect to the Docker daemon".
Solution: Docker Desktop is not running. Start the application and wait until it says the engine is ready.

Problem: "ModuleNotFoundError: No module named 'catboost'".
Solution: The virtual environment is not active, or the dependencies are not installed. Activate the environment and run "pip install -r requirements.txt" again.

Problem: The browser shows nothing at localhost:8501.
Solution: The container may not be running. Run "docker compose ps" inside code/deployment to check the state, and "docker compose logs app" to see the log messages.

Problem: The Streamlit app shows "Error calling API".
Solution: The API container is not running or not reachable. Check with "docker compose logs api". Make sure the "API_URL" environment variable in docker-compose.yml points to "http://api:8000/predict".

Problem: The Airflow DAG does not appear in the UI.
Solution: Check that the DAG file is inside services/airflow/dags and that the AIRFLOW__CORE__DAGS_FOLDER variable points to the same folder. Restart Airflow after any change to the DAG file.

Problem: The Airflow task "deployment" fails.
Solution: This task calls Docker, so Docker Desktop must be running. Also check that the PROJECT_DIR variable in the DAG file points to the correct absolute path.

Problem: The port 8000, 8501, or 8080 is already in use.
Solution: Another process is using the port. You can stop that process, or change the port mapping in docker-compose.yml and in the Streamlit URL.


HOW THE COMPONENTS TALK TO EACH OTHER
=====================================

The following sequence describes what happens when the user presses the "Predict" button.

1. The Streamlit app collects the values from the input fields into a Python dictionary.
2. The app sends an HTTP POST request to the URL http://api:8000/predict. The "api" part is the service name in docker-compose.yml. Docker resolves it to the internal IP address of the API container.
3. The FastAPI application receives the request. It validates the input against the Pydantic model. If a field is missing or has the wrong type, FastAPI returns an error.
4. The API converts the input into a pandas DataFrame with the same column names as the training data.
5. The API calls the "predict_proba" method of the trained model pipeline. The pipeline applies the same preprocessing steps that were used during training, and then calls the classifier.
6. The API takes the probability for the positive class (income > 50K), applies a threshold of 0.5, and produces a class label.
7. The API returns a JSON response with three fields: "prediction" (0 or 1), "label" ("<=50K" or ">50K"), and "probability" (a float between 0 and 1).
8. The Streamlit app reads the JSON response and displays the label and the probability on the page.

This design follows the separation of concerns principle. The model logic lives in the API. The user interface lives in the app. Both can be developed, tested, and deployed separately.


AUTOMATION
==========

The complete pipeline is automated with Airflow. The DAG runs every five minutes. On each run it does the following:

1. Runs the data engineering script. The script downloads, cleans, and splits the data.
2. Runs the model engineering script. The script trains, evaluates, and saves the model.
3. Runs "docker compose up -d --build" inside code/deployment. Docker rebuilds the API and app images, so the new model is included, and restarts the containers.

Because the model file is mounted into the API container from the host, the API sees the new model after the container restarts. The Streamlit app does not need to change, because it always talks to the API.

If a run takes longer than five minutes, the interval can be increased. Edit the "schedule_interval" value in the DAG file. For example, "*/15 * * * *" runs every fifteen minutes. The cron expression uses the standard five field format: minute, hour, day of month, month, day of week.


NOTES
=====

The trained model file is not committed to the repository. It is produced by Stage 2 and it can be large. To recreate it, run Stage 2 as described above.

The raw dataset file is also not committed. It is downloaded by Stage 1 and can be recreated by running Stage 1.

If the pipeline is demonstrated to a teacher or a reviewer, the full sequence is:

- clone the repository;
- create and activate a virtual environment;
- install the requirements;
- run Stage 1;
- run Stage 2;
- run "docker compose up -d --build" inside code/deployment;
- open http://localhost:8501 and make a prediction;
- start Airflow and show that the DAG runs every five minutes.
