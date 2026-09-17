ADULT INCOME MLOPS PIPELINE
============================

This repository contains an automated machine learning pipeline for the Adult Income dataset. The pipeline has three stages: data engineering, model engineering, and deployment. It runs automatically every five minutes with Apache Airflow. The model is served through a FastAPI application, and a Streamlit web application lets the user enter data and get a prediction. Both the API and the app run in separate Docker containers.

The goal of this project is to show a simple but complete MLOps workflow. The dataset is small, the models are standard gradient boosting models, and the focus is on the pipeline working from start to finish without manual steps.

Airflow itself also runs in a Docker container. It talks to the host Docker daemon through the mounted Docker socket, so it can build and start the API and the app containers during the deployment stage.

WHAT THE APPLICATION DOES
=========================

The application predicts whether a person earns more than 50 thousand dollars per year. The input is a set of personal and work related features, such as age, education, occupation, working hours, and so on. The output is a class label (either "<=50K" or ">50K") and a probability between 0 and 1.

The user opens the Streamlit web page in a browser. The page shows several input fields. After filling the fields, the user presses the "Predict" button. The Streamlit app sends the data to the FastAPI service using an HTTP POST request. The API loads the trained model, makes a prediction, and sends the result back. The Streamlit app then shows the predicted label and the probability on the page.

The API and the web app are two separate services. They run in two separate Docker containers. The Streamlit container talks to the FastAPI container through the internal Docker network. This is the same pattern that is used in real production systems, where the front end and the model service are deployed separately.

PIPELINE STAGES
===============

Stage 1. Data Engineering

The script code/datasets/data_engineering.py downloads the raw Adult Income dataset from the UCI Machine Learning Repository. Then it cleans the data. Cleaning means the following steps:

missing values, which are stored as "?" in the raw file, are removed;

rows with duplicate values are removed;

outliers in the "age" and "hours-per-week" columns are removed using the interquartile range method;

the target column "income" is converted to a binary value (0 for "<=50K", 1 for ">50K").

After cleaning, the script splits the data into a training set (80 percent) and a test set (20 percent). The split keeps the same proportion of the target classes in both parts. The two files are saved to the data/processed folder as train.csv and test.csv.

Stage 2. Model Engineering

The script code/models/model_engineering.py loads the training and test files from Stage 1. Then it builds a preprocessing pipeline. Numerical features are scaled with StandardScaler. Categorical features are encoded with OneHotEncoder.

After preprocessing, three gradient boosting models are trained and tuned with Optuna:

CatBoost

XGBoost

LightGBM

For each model, Optuna runs a number of trials and searches for the best hyperparameters. The tuning metric is ROC AUC on the test set. Each run is logged to MLflow, which uses a local SQLite database (mlflow.db in the project root).

After tuning, the best model is refit on the full training set, evaluated on the test set, and saved to models/best_model.joblib. The metrics for all three models are saved to models/metrics.json. The best model is the one with the highest ROC AUC.

Stage 3. Deployment

Stage 3 uses Docker and Docker Compose. Two Docker images are built:

the API image, based on FastAPI and Uvicorn;

the web app image, based on Streamlit.

The Docker Compose file code/deployment/docker-compose.yaml starts both containers. The FastAPI container exposes port 8000. The Streamlit container exposes port 8501. The trained model file from Stage 2 is mounted into the API container as a read only volume. The Streamlit container sends requests to the API using the internal Docker hostname "api".

Automation with Airflow

The Airflow DAG services/airflow/dags/ml_pipeline_dag.py connects all three stages. It has three tasks:

data_engineering, which runs the Stage 1 script;

model_engineering, which runs the Stage 2 script;

deployment, which runs "docker compose up -d --build" inside code/deployment.

The tasks run in order. The DAG is scheduled with the cron expression "*/5 * * * *", so it runs every five minutes. If a run takes longer than five minutes, the interval can be increased in the DAG file.

Airflow runs inside its own Docker container, built from services/airflow/Dockerfile. The container has the Docker CLI installed and the host Docker socket mounted, so it can manage the API and app containers directly.

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

code/deployment/docker-compose.yaml
Docker Compose file that starts the API and app containers.

services/airflow/Dockerfile
Dockerfile for the Airflow container. Installs Docker CLI and the
Python dependencies needed by Stages 1 and 2.

services/airflow/docker-compose.yml
Compose file that starts the Airflow container.

services/airflow/requirements.txt
Python dependencies installed inside the Airflow image.

services/airflow/.env
Local file with the HOST_PROJECT_DIR variable. Not committed to git.

services/airflow/dags/ml_pipeline_dag.py
Airflow DAG that connects all three stages and runs every five minutes.

data/raw/
Raw dataset files. The dataset is downloaded here by Stage 1.

data/processed/
Cleaned train and test files. Written by Stage 1.

models/
Trained model file and metrics. Written by Stage 2.

requirements.txt
Python dependencies for local runs of Stages 1 and 2.

README.txt
This file.

REQUIREMENTS
============

Before running the pipeline, make sure the following software is installed on your computer.

Python 3.10 or newer, for running Stages 1 and 2 locally. You can check with "python --version".

pip. It is usually installed together with Python.

Git. You can check with "git --version".

Docker Desktop. You can check with "docker --version".

Docker Compose. It is included in recent versions of Docker Desktop. You can check with "docker compose version".

Docker Desktop must be running before you start Stage 3 or Airflow. On Windows and macOS you should open the Docker Desktop application and wait until the icon turns green or shows that the engine is running. On Linux you can start the Docker daemon with "sudo systemctl start docker".

If you run the project inside WSL, make sure the project folder is placed under a path without Cyrillic characters or spaces. For example, use /mnt/c/Users/<name>/Adult_Income_Classifier instead of a path with non-ASCII letters.

STEP BY STEP GUIDE TO RUN THE PIPELINE LOCALLY
=============================================

Step 1. Clone the repository

Open a terminal and run:

git clone https://github.com/Billed671/Adult_Income_Classifier.git
cd Adult_Income_Classifier

After this command, the working directory of your terminal is the root of the repository.

Step 2. Create a virtual environment

A virtual environment keeps the project dependencies separate from the rest of the system. Run:

python -m venv .venv

Then activate it:

On macOS or Linux:

source .venv/bin/activate

On Windows with Git Bash:

source .venv/Scripts/activate

On Windows with PowerShell:

.venv\Scripts\Activate.ps1

After activation, your shell prompt usually shows "(.venv)" at the beginning.

Step 3. Install the dependencies

Run:

pip install --upgrade pip
pip install -r requirements.txt

This installs pandas, numpy, scikit-learn, CatBoost, XGBoost, LightGBM, Optuna, MLflow, and joblib. The first installation can take several minutes.

Note. The requirements.txt in the project root is meant for local runs of Stages 1 and 2. The Docker containers use their own requirements.txt files inside code/deployment/api and code/deployment/app. The Airflow container uses services/airflow/requirements.txt.

Step 4. Run Stage 1 (data engineering)

Run:

python code/datasets/data_engineering.py

What happens:

The script downloads the Adult Income dataset from the UCI repository and saves it to data/raw/adult.csv. If the file is already there, the download is skipped.

The script cleans the data and removes missing values, duplicates, and outliers.

The script splits the data into a training set and a test set.

The two files are saved to data/processed/train.csv and data/processed/test.csv.

You can check the result with:

ls -lh data/processed
head -2 data/processed/train.csv

Step 5. Run Stage 2 (model engineering)

Run:

python code/models/model_engineering.py

What happens:

The script loads the training and test data.

It builds the preprocessing pipeline.

It tunes CatBoost, XGBoost, and LightGBM with Optuna. Each model runs several trials. The default is 15 trials per model. You can change this number in the script.

Each run is logged to MLflow in a local SQLite database.

The best model is saved to models/best_model.joblib.

All metrics are saved to models/metrics.json.

Depending on your computer, this step can take between five and twenty minutes. When the script finishes, you should see a line similar to:

Best model: catboost (AUC=0.918) saved to models/best_model.joblib

Optional. You can view the MLflow results in a browser:

mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

Then open http://localhost:5000. Press Ctrl+C in the terminal to stop the MLflow UI when you are done.

Step 6. Run Stage 3 (deployment) manually

Make sure Docker Desktop is running. Then run:

cd code/deployment
docker compose up -d --build

The first build takes several minutes because the images need to be created and the Python packages need to be installed inside them. When the build is finished, two containers start:

adult_api on port 8000;

adult_app on port 8501.

You can check the status with:

docker compose ps

Both containers should show the state "running".

Step 7. Use the web application

Open a browser and go to:

http://localhost:8501

The Streamlit page shows the input form. Fill the fields and press the "Predict" button. The prediction appears below the form as a label and a probability. The label is either "<=50K" or ">50K".

If you want to test the API directly, open:

http://localhost:8000/docs

This is the automatic FastAPI documentation page. You can send a test request from there.

Step 8. Stop the containers

To stop the API and the web app, run:

docker compose down

This stops and removes the containers. The built images stay on disk, so the next "docker compose up" is faster.

Step 9. Run the full pipeline with Airflow in Docker

Airflow runs in its own Docker container. It has all the Python dependencies needed by Stages 1 and 2, and it has the Docker CLI installed, so it can run "docker compose" for the deployment stage. The host Docker socket is mounted into the container.

Step 9.1. Set the HOST_PROJECT_DIR variable

The Airflow container mounts the project twice: once at /opt/airflow/project for reading files, and once at the same absolute path as on the host, so that the host Docker daemon can resolve volume paths for the API and app containers. The path is stored in a small .env file.

From the services/airflow folder, run:

cd services/airflow
echo "HOST_PROJECT_DIR=$(cd ../.. && pwd)" > .env
cat .env

You should see something like:

HOST_PROJECT_DIR=/mnt/c/Users/Bille/Adult_Income_Classifier

or on macOS:

HOST_PROJECT_DIR=/Users/yourname/Adult_Income_Classifier

Step 9.2. Check the DOCKER_GID value

The Airflow image adds the airflow user to a group with the same GID as the docker.sock file on the host. On Linux and WSL, find the GID of the docker group:

getent group docker

The output looks like "docker:x:1001:". The number after the second colon is the GID. If it is not 1001, open services/airflow/docker-compose.yml and change the DOCKER_GID value in the build args, and open services/airflow/Dockerfile and change the ARG DOCKER_GID value in the same way.

Step 9.3. Build and start Airflow

From the services/airflow folder, run:

docker compose up -d --build

The first build takes several minutes. After it finishes, check the log:

docker logs airflow -f

Wait for a line like "Listening at: http://0.0.0.0:8080" or "Airflow is ready". Press Ctrl+C to stop following the log.

Step 9.4. Reset the Airflow admin password

On the first start, the standalone mode of Airflow creates an admin user with a random password. To set a known password, run:

docker exec -it airflow airflow users reset-password --username admin --password admin

Step 9.5. Open the Airflow UI

Open a browser and go to:

http://localhost:8080

Log in with username "admin" and password "admin".

Find the DAG with the id "adult_income_pipeline". Turn it on with the toggle on the left. The DAG will start running every five minutes. To run it right now without waiting, press the "Trigger DAG" button (the play icon) next to the DAG name.

Click on the DAG name to see task states. Green means success. If a task fails, click on it and open the Log to see what happened.

Step 9.6. Verify that the deployment task works

Inside the Airflow container, the deployment task runs:

cd "$HOST_PROJECT_DIR/code/deployment" && docker compose up -d --build

For this to work, the Docker socket must be accessible. You can check it manually:

docker exec -it airflow docker ps

The command should list the running containers on the host, including the airflow container itself. If it prints a permission error, see the section "Common problems and solutions" below.

Step 9.7. Stop Airflow

To stop the Airflow container:

docker compose down

from the services/airflow folder.

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
Solution: The API container is not running or not reachable. Check with "docker compose logs api". Make sure the "API_URL" environment variable in docker-compose.yaml points to "http://api:8000/predict".

Problem: The Airflow DAG does not appear in the UI.
Solution: Check that the DAG file is inside services/airflow/dags. Restart the Airflow container after any change to the DAG file: "docker compose restart airflow".

Problem: The Airflow admin password does not work after a rebuild.
Solution: The standalone command creates a new admin with a random password on each fresh start. Reset the password manually:
docker exec -it airflow airflow users reset-password --username admin --password admin

Problem: The Airflow task "deployment" fails with "permission denied while trying to connect to the docker API".
Solution: The airflow user inside the container is not in the group that owns /var/run/docker.sock. Check the GID of the docker group on the host with "getent group docker", then set DOCKER_GID in services/airflow/docker-compose.yml and services/airflow/Dockerfile to that value, and rebuild:
docker compose down
docker compose build --no-cache
docker compose up -d
Alternatively, for a quick temporary fix, run "sudo chmod 666 /var/run/docker.sock" on the host. This resets after a reboot.

Problem: The Airflow task "deployment" fails with "unable to prepare context: path ... not found".
Solution: The mirror mount of the project is missing or the HOST_PROJECT_DIR value is wrong. Check that services/airflow/.env contains the correct absolute path to the project, and that services/airflow/docker-compose.yml mounts the project at the same path inside the container, next to the /opt/airflow/project mount. Then restart the Airflow container.

Problem: The Airflow task "deployment" fails with "compose file ... is invalid: no such file or directory".
Solution: The compose file inside code/deployment is named docker-compose.yaml, not docker-compose.yml. Make sure the DAG uses the correct name, or rename the file to docker-compose.yml.

Problem: The Airflow DAG runs but the model_engineering task fails with "libgomp.so.1: cannot open shared object file".
Solution: The Airflow image is missing the OpenMP runtime that LightGBM and XGBoost need. Rebuild the image after making sure that services/airflow/Dockerfile installs libgomp1.

Problem: The port 8000, 8501, or 8080 is already in use.
Solution: Another process is using the port. You can stop that process, or change the port mapping in the corresponding docker-compose file.

HOW THE COMPONENTS TALK TO EACH OTHER
=====================================

The following sequence describes what happens when the user presses the "Predict" button.

The Streamlit app collects the values from the input fields into a Python dictionary.

The app sends an HTTP POST request to the URL http://api:8000/predict. The "api" part is the service name in docker-compose.yaml. Docker resolves it to the internal IP address of the API container.

The FastAPI application receives the request. It validates the input against the Pydantic model. If a field is missing or has the wrong type, FastAPI returns an error.

The API converts the input into a pandas DataFrame with the same column names as the training data.

The API calls the "predict_proba" method of the trained model pipeline. The pipeline applies the same preprocessing steps that were used during training, and then calls the classifier.

The API takes the probability for the positive class (income > 50K), applies a threshold of 0.5, and produces a class label.

The API returns a JSON response with three fields: "prediction" (0 or 1), "label" ("<=50K" or ">50K"), and "probability" (a float between 0 and 1).

The Streamlit app reads the JSON response and displays the label and the probability on the page.

This design follows the separation of concerns principle. The model logic lives in the API. The user interface lives in the app. Both can be developed, tested, and deployed separately.

The Airflow container runs beside the API and the app. It does not receive HTTP requests. It only triggers the pipeline on a schedule and starts the deployment containers through the Docker socket.

AUTOMATION
==========

The complete pipeline is automated with Airflow. The DAG runs every five minutes. On each run it does the following:

Runs the data engineering script. The script downloads, cleans, and splits the data.

Runs the model engineering script. The script trains, evaluates, and saves the model.

Runs "docker compose up -d --build" inside code/deployment. Docker rebuilds the API and app images, so the new model is included, and restarts the containers.

Because the model file is mounted into the API container from the host, the API sees the new model after the container restarts. The Streamlit app does not need to change, because it always talks to the API.

The deployment task runs the command inside the Airflow container, but the Docker daemon that actually performs the build and starts the containers is the host daemon. The two sides agree on the file paths because the project is mounted at the same absolute path in both places.

If a run takes longer than five minutes, the interval can be increased. Edit the "schedule_interval" value in the DAG file. For example, "*/15 * * * *" runs every fifteen minutes. The cron expression uses the standard five field format: minute, hour, day of month, month, day of week.

NOTES
=====

The trained model file is not committed to the repository. It is produced by Stage 2 and it can be large. To recreate it, run Stage 2 as described above.

The raw dataset file is also not committed. It is downloaded by Stage 1 and can be recreated by running Stage 1.

The file services/airflow/.env is not committed. Each user creates it after cloning the repository with a single command:
echo "HOST_PROJECT_DIR=$(cd ../.. && pwd)" > .env
inside the services/airflow folder.

If the pipeline is demonstrated to a teacher or a reviewer, the full sequence is:

clone the repository;

create and activate a virtual environment;

install the requirements;

run Stage 1;

run Stage 2;

(optional) run "docker compose up -d --build" inside code/deployment to start the API and the app manually;

create services/airflow/.env with the HOST_PROJECT_DIR variable;

run "docker compose up -d --build" inside services/airflow to start Airflow;

reset the Airflow admin password and open http://localhost:8080;

turn on the DAG adult_income_pipeline and press "Trigger DAG";

wait for all three tasks to turn green;

open http://localhost:8501 and make a prediction;

show the DAG page in Airflow to demonstrate that it runs every five minutes.
