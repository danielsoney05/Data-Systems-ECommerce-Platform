# Data-Systems-Retail-Sales-Analytics-Platform
An E-Commerce Sales Analytics Platform developed for the Data Systems Project, Autumn 2026

----------------------------------------------------------------------
HOW TO RUN APP.PY VIA DOCKER (INTENDED WAY)
----------------------------------------------------------------------

## Prerequisites

Before running the app, make sure you have:

- Docker installed
- A Google Cloud service account JSON key
- Access to the required Google Cloud Storage bucket
- Access to the required BigQuery dataset/tables
- The repository cloned locally

---

## 1. Clone the Repository

```bash
git clone <your-repository-url>
cd Data-Systems-Retail-Sales-Analytics-Platform


## 2. Add the Google Cloud Key

Place your Google Cloud service account key inside:

input_app/key.json

The file should look like a standard Google Cloud service account JSON key.

Example structure:

Data-Systems-Retail-Sales-Analytics-Platform/
├── input_app/
│   ├── app.py
│   ├── key.json
│   └── templates/
├── Dockerfile
└── requirements.txt

Important: do not commit key.json to GitHub.

Make sure .gitignore includes:

input_app/key.json
key.json
*.json


## 3. Build the Docker Image

From the repository root, run:

docker build -t olist-input-app .

This creates a Docker image called:

olist-input-app


## 4. Run the Docker Container
Windows PowerShell / CMD:
docker run -p 5000:5000 ^
-v "%cd%\input_app\key.json:/app/input_app/key.json" ^
--name olist-app ^
olist-input-app

Mac/Linux:
docker run -p 5000:5000 \
-v "$(pwd)/input_app/key.json:/app/input_app/key.json" \
--name olist-app \
olist-input-app

This starts the Flask app and maps it to:

http://localhost:5000


## 5. Open the App

Once the container is running, open:

http://localhost:5000

You should be redirected to the login page.



----------------------------------------------------------------------
NAVIGATING THE WEBSITE
----------------------------------------------------------------------
## 6. Login

Enter a customer_id.

You can use:

an existing customer ID from the Olist dataset
or a new customer ID to simulate a new user

Existing customers will show profile details and previous orders if they exist in the dataset.


## 7. Useful Test Pages
Test BigQuery Connection
http://localhost:5000/test-bq

This checks whether the app can query BigQuery.

Test GCS Output Folder
http://localhost:5000/test-gcs

This checks whether the app can access the Google Cloud Storage output folder.

Debug App Config
http://localhost:5000/debug-config

This shows the active bucket, output path, BigQuery project, dataset, and table config.


## 8. Where New Orders Are Saved

New orders created through the app are saved to Google Cloud Storage as a daily CSV file.

Example:

gs://olist-494110_bucket/daily_orders/orders_2026-05-06.csv

The file name is generated from the current date:

orders_YYYY-MM-DD.csv


## 9. Stopping the Container

To stop the app:

docker stop olist-app

To remove the container:

docker rm olist-app



----------------------------------------------------------------------
COMMON ISSUES
----------------------------------------------------------------------
## FileNotFoundError: key.json

The container cannot find the Google Cloud key.

Check that:

input_app/key.json

exists locally and that the Docker volume mount path is correct.