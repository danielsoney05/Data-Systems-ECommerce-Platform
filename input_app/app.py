from flask import Flask, render_template, request, redirect, session, url_for
from google.oauth2 import service_account
from google.cloud import storage, bigquery
import pandas as pd
from io import BytesIO
import uuid
from datetime import datetime, timedelta, date
import os

# ============================================================
# CONFIG
# ============================================================

# ----------------------------
# Google credentials
# ----------------------------
# Local fallback:
#   input_app/key.json
#
# Docker:
#   either mount the key here:
#   /app/input_app/key.json
#
# or set:
#   GOOGLE_APPLICATION_CREDENTIALS=/app/input_app/key.json
KEY_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "input_app/key.json")

# ----------------------------
# GCS output config
# ----------------------------
# Bucket name ONLY.
# Do NOT include folders here.
BUCKET_NAME = "olist-494110_bucket"

# Output folder for app-created orders.
# Example final path:
# gs://olist-494110_bucket/app_output/orders_2026-05-06.csv
OUTPUT_PREFIX = "daily_orders"

# ----------------------------
# BigQuery config
# ----------------------------
# Your project ID can be pulled from the key file, but you can override here.
BQ_PROJECT_ID = os.getenv("BQ_PROJECT_ID", None)

# CHANGE THIS to your actual BigQuery dataset name.
# Example:
# BQ_DATASET_ID = "olist"
BQ_DATASET_ID = "olist"

# These are the table names BigQuery should have.
# They should match the CSVs you uploaded/loaded.
BQ_CUSTOMERS_TABLE = "customer"
BQ_ORDERS_TABLE = "orders"
BQ_ORDER_ITEMS_TABLE = "order_items"
BQ_PAYMENTS_TABLE = "order_payments"
BQ_PRODUCTS_TABLE = "products"
BQ_SELLERS_TABLE = "sellers"
BQ_CATEGORY_TRANSLATION_TABLE = "product_category_name_translation"

# Flask session secret.
SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-later")


# ============================================================
# AUTH + CLIENTS
# ============================================================

def create_google_clients():
    print("Current working directory:", os.getcwd())
    print("Configured KEY_PATH:", KEY_PATH)
    print("Key exists:", os.path.exists(KEY_PATH))

    if os.path.exists(KEY_PATH):
        credentials = service_account.Credentials.from_service_account_file(KEY_PATH)

        project_id = BQ_PROJECT_ID or credentials.project_id

        storage_client = storage.Client(
            credentials=credentials,
            project=project_id
        )

        bq_client = bigquery.Client(
            credentials=credentials,
            project=project_id
        )

        return storage_client, bq_client, project_id

    print("⚠️ Key file not found. Falling back to default Google credentials.")

    storage_client = storage.Client()
    bq_client = bigquery.Client()
    project_id = BQ_PROJECT_ID or bq_client.project

    return storage_client, bq_client, project_id


storage_client, bq_client, ACTIVE_PROJECT_ID = create_google_clients()

if BQ_PROJECT_ID is None:
    BQ_PROJECT_ID = ACTIVE_PROJECT_ID


# ============================================================
# APP INIT
# ============================================================

app = Flask(__name__)
app.secret_key = SECRET_KEY


# ============================================================
# GCS OUTPUT HELPERS
# ============================================================

def get_output_file_name():
    """
    Daily output file name.

    Example:
    orders_2026-05-06.csv
    """
    return f"orders_{date.today().strftime('%Y-%m-%d')}.csv"


def output_gcs_path(file_name):
    """
    Builds output object path.

    Example:
    app_output/orders_2026-05-06.csv
    """
    if OUTPUT_PREFIX:
        return f"{OUTPUT_PREFIX}/{file_name}"

    return file_name


def read_output_csv(file_name):
    """
    Reads the app-created output CSV from GCS.

    This is only for new rows created by the Flask app.
    If the file does not exist yet, return an empty DataFrame.
    """
    object_path = output_gcs_path(file_name)

    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(object_path)

        print(f"🔍 Trying to read output: gs://{BUCKET_NAME}/{object_path}")

        if not blob.exists():
            print(f"⚠️ Output file not found yet: gs://{BUCKET_NAME}/{object_path}")
            return pd.DataFrame()

        data = blob.download_as_bytes()
        df = pd.read_csv(BytesIO(data))

        print(f"✅ Loaded output file {file_name}: {len(df)} rows")
        return df

    except Exception as e:
        print(f"❌ Error reading output gs://{BUCKET_NAME}/{object_path}: {e}")
        return pd.DataFrame()


def write_output_csv(df, file_name):
    """
    Writes app-created output CSV to GCS.
    """
    object_path = output_gcs_path(file_name)

    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(object_path)

        csv_data = df.to_csv(index=False)
        blob.upload_from_string(csv_data, content_type="text/csv")

        print(f"✅ Uploaded output to gs://{BUCKET_NAME}/{object_path}")

    except Exception as e:
        print(f"❌ Error writing output gs://{BUCKET_NAME}/{object_path}: {e}")


def load_data():
    """
    Loads today's generated order output.
    """
    return read_output_csv(get_output_file_name())


def save_data(df):
    """
    Saves today's generated order output.
    """
    write_output_csv(df, get_output_file_name())


# ============================================================
# BIGQUERY HELPERS
# ============================================================

def bq_table(table_name):
    """
    Builds a full BigQuery table reference.

    Example:
    `project.dataset.table`
    """
    return f"`{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{table_name}`"


def run_bq_query(query, parameters=None):
    """
    Runs a BigQuery query and returns list[dict].

    Uses parameters where needed to avoid string-injection issues.
    """
    try:
        job_config = None

        if parameters:
            job_config = bigquery.QueryJobConfig(
                query_parameters=parameters
            )

        result = bq_client.query(query, job_config=job_config).result()

        rows = []
        for row in result:
            rows.append(dict(row))

        return rows

    except Exception as e:
        print("❌ BigQuery query failed:")
        print(e)
        print("Query was:")
        print(query)
        return []


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_float(value):
    if pd.isna(value) or value == "":
        return 0.0

    try:
        return float(value)
    except Exception:
        return 0.0


def safe_int(value, default=1):
    if pd.isna(value) or value == "":
        return default

    try:
        return int(value)
    except Exception:
        return default


def now_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def future_date_string(days):
    return (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")


def clean_money(value):
    if value is None or pd.isna(value) or value == "":
        return "unknown"

    try:
        return f"${float(value):.2f}"
    except Exception:
        return str(value)


def clean_records(records):
    """
    Makes BigQuery records safe for Jinja tojson.
    Converts NaN-like values to None.
    """
    cleaned = []

    for record in records:
        item = {}

        for key, value in record.items():
            if pd.isna(value):
                item[key] = None
            else:
                item[key] = value

        cleaned.append(item)

    return cleaned


# ============================================================
# LOGIN HELPERS
# ============================================================

def get_logged_in_customer_id():
    return session.get("customer_id")


def require_login():
    return bool(get_logged_in_customer_id())


# ============================================================
# PRODUCT LOOKUP FROM BIGQUERY
# ============================================================

def build_product_display_name(row):
    product_id = row.get("product_id")

    category_en = row.get("product_category_name_english")
    category_raw = row.get("product_category_name")

    if category_en:
        category = category_en
    elif category_raw:
        category = category_raw
    else:
        category = "unknown category"

    price = clean_money(row.get("price"))
    freight = clean_money(row.get("freight_value"))

    return f"{category} — Price: {price} — Freight: {freight} — {product_id}"


def get_products():
    """
    Gets products for the create-order page from BigQuery.

    It joins:
    - products
    - category translation
    - cheapest listing from order_items

    The form still submits only product_id and quantity.
    """

    query = f"""
        WITH cheapest_listing AS (
            SELECT
                product_id,
                seller_id,
                price,
                freight_value
            FROM (
                SELECT
                    product_id,
                    seller_id,
                    price,
                    freight_value,
                    ROW_NUMBER() OVER (
                        PARTITION BY product_id
                        ORDER BY price ASC
                    ) AS rn
                FROM {bq_table(BQ_ORDER_ITEMS_TABLE)}
                WHERE product_id IS NOT NULL
            )
            WHERE rn = 1
        )

        SELECT
            p.product_id,
            p.product_category_name,
            t.product_category_name_english,
            p.product_photos_qty,
            p.product_weight_g,
            p.product_length_cm,
            p.product_height_cm,
            p.product_width_cm,
            c.seller_id,
            c.price,
            c.freight_value
        FROM {bq_table(BQ_PRODUCTS_TABLE)} p
        LEFT JOIN {bq_table(BQ_CATEGORY_TRANSLATION_TABLE)} t
            ON p.product_category_name = t.product_category_name
        LEFT JOIN cheapest_listing c
            ON p.product_id = c.product_id
        WHERE p.product_id IS NOT NULL
        LIMIT 100
    """

    products = run_bq_query(query)

    for product in products:
        product["display_name"] = build_product_display_name(product)

    return clean_records(products)


def get_product_enrichment(product_id):
    """
    Given a product_id, gets all enrichment data from BigQuery.

    Used when saving a new order row.
    """

    query = f"""
        WITH cheapest_listing AS (
            SELECT
                product_id,
                seller_id,
                price,
                freight_value
            FROM (
                SELECT
                    product_id,
                    seller_id,
                    price,
                    freight_value,
                    ROW_NUMBER() OVER (
                        PARTITION BY product_id
                        ORDER BY price ASC
                    ) AS rn
                FROM {bq_table(BQ_ORDER_ITEMS_TABLE)}
                WHERE product_id = @product_id
            )
            WHERE rn = 1
        )

        SELECT
            c.product_id,
            c.seller_id,
            c.price,
            c.freight_value,

            s.seller_zip_code_prefix,
            s.seller_city,
            s.seller_state,

            p.product_category_name,
            t.product_category_name_english,

            p.product_name_lenght AS product_name_length,
            p.product_description_lenght AS product_description_length,
            p.product_photos_qty,
            p.product_weight_g,
            p.product_length_cm,
            p.product_height_cm,
            p.product_width_cm
        FROM cheapest_listing c
        LEFT JOIN {bq_table(BQ_SELLERS_TABLE)} s
            ON c.seller_id = s.seller_id
        LEFT JOIN {bq_table(BQ_PRODUCTS_TABLE)} p
            ON c.product_id = p.product_id
        LEFT JOIN {bq_table(BQ_CATEGORY_TRANSLATION_TABLE)} t
            ON p.product_category_name = t.product_category_name
        LIMIT 1
    """

    rows = run_bq_query(
        query,
        parameters=[
            bigquery.ScalarQueryParameter("product_id", "STRING", product_id)
        ]
    )

    if not rows:
        return None

    row = rows[0]

    return {
        "seller_id": row.get("seller_id"),
        "price": row.get("price"),
        "freight_value": row.get("freight_value"),

        "seller_zip_code_prefix": row.get("seller_zip_code_prefix"),
        "seller_lat": None,
        "seller_lng": None,
        "seller_city": row.get("seller_city"),
        "seller_state": row.get("seller_state"),

        "product_category_name": row.get("product_category_name"),
        "product_category_name_english": row.get("product_category_name_english"),
        "product_name_length": row.get("product_name_length"),
        "product_description_length": row.get("product_description_length"),
        "product_photos_qty": row.get("product_photos_qty"),
        "product_weight_g": row.get("product_weight_g"),
        "product_length_cm": row.get("product_length_cm"),
        "product_height_cm": row.get("product_height_cm"),
        "product_width_cm": row.get("product_width_cm"),
    }


# ============================================================
# CUSTOMER LOOKUP FROM BIGQUERY
# ============================================================

def get_customer_profile(customer_id):
    """
    Gets customer details from BigQuery and order summary from olist_orders_dataset.
    """

    query = f"""
        SELECT
            c.customer_id,
            c.customer_unique_id,
            c.customer_zip_code_prefix,
            c.customer_city,
            c.customer_state,

            COUNT(o.order_id) AS order_count,
            MAX(o.order_purchase_timestamp) AS latest_order_date
        FROM {bq_table(BQ_CUSTOMERS_TABLE)} c
        LEFT JOIN {bq_table(BQ_ORDERS_TABLE)} o
            ON c.customer_id = o.customer_id
        WHERE c.customer_id = @customer_id
        GROUP BY
            c.customer_id,
            c.customer_unique_id,
            c.customer_zip_code_prefix,
            c.customer_city,
            c.customer_state
        LIMIT 1
    """

    rows = run_bq_query(
        query,
        parameters=[
            bigquery.ScalarQueryParameter("customer_id", "STRING", customer_id)
        ]
    )

    if rows:
        row = rows[0]

        return {
            "customer_id": row.get("customer_id"),
            "customer_unique_id": row.get("customer_unique_id"),
            "customer_zip_code_prefix": row.get("customer_zip_code_prefix"),
            "customer_city": row.get("customer_city"),
            "customer_state": row.get("customer_state"),
            "customer_lat": None,
            "customer_lng": None,
            "exists_in_customers_dataset": True,
            "order_count": row.get("order_count") or 0,
            "latest_order_date": row.get("latest_order_date"),
        }

    # New customer fallback
    return {
        "customer_id": customer_id,
        "customer_unique_id": None,
        "customer_zip_code_prefix": None,
        "customer_city": None,
        "customer_state": None,
        "customer_lat": None,
        "customer_lng": None,
        "exists_in_customers_dataset": False,
        "order_count": 0,
        "latest_order_date": None,
    }


def get_customer_info(customer_id):
    """
    Gets customer info for saving new app-created orders.
    """

    profile = get_customer_profile(customer_id)

    return {
        "customer_id": profile.get("customer_id"),
        "customer_unique_id": profile.get("customer_unique_id"),
        "customer_zip_code_prefix": profile.get("customer_zip_code_prefix"),
        "customer_city": profile.get("customer_city"),
        "customer_state": profile.get("customer_state"),
        "customer_lat": None,
        "customer_lng": None,
    }


# ============================================================
# ORDER VIEWER FROM BIGQUERY
# ============================================================

def get_existing_customer_orders(customer_id):
    """
    Gets existing historical customer orders from BigQuery.

    Main source:
    - olist_orders_dataset

    Joined with:
    - order_items
    - payments
    - products
    - category translation
    """

    query = f"""
        SELECT
            o.order_id,
            o.customer_id,
            o.order_status,
            o.order_purchase_timestamp,
            o.order_approved_at,
            o.order_delivered_carrier_date,
            o.order_delivered_customer_date,
            o.order_estimated_delivery_date,

            i.order_item_id,
            i.product_id,
            i.seller_id,
            i.price,
            i.freight_value,

            p.product_category_name,
            t.product_category_name_english,

            pay.payment_type,
            pay.payment_installments,
            pay.payment_value
        FROM {bq_table(BQ_ORDERS_TABLE)} o
        LEFT JOIN {bq_table(BQ_ORDER_ITEMS_TABLE)} i
            ON o.order_id = i.order_id
        LEFT JOIN {bq_table(BQ_PRODUCTS_TABLE)} p
            ON i.product_id = p.product_id
        LEFT JOIN {bq_table(BQ_CATEGORY_TRANSLATION_TABLE)} t
            ON p.product_category_name = t.product_category_name
        LEFT JOIN {bq_table(BQ_PAYMENTS_TABLE)} pay
            ON o.order_id = pay.order_id
        WHERE o.customer_id = @customer_id
        ORDER BY o.order_purchase_timestamp DESC, i.order_item_id ASC
    """

    rows = run_bq_query(
        query,
        parameters=[
            bigquery.ScalarQueryParameter("customer_id", "STRING", customer_id)
        ]
    )

    if not rows:
        return []

    orders_by_id = {}

    for row in rows:
        order_id = row.get("order_id")

        if order_id not in orders_by_id:
            orders_by_id[order_id] = {
                "order_id": order_id,
                "order_status": row.get("order_status"),
                "order_purchase_timestamp": row.get("order_purchase_timestamp"),
                "order_approved_at": row.get("order_approved_at"),
                "order_delivered_carrier_date": row.get("order_delivered_carrier_date"),
                "order_delivered_customer_date": row.get("order_delivered_customer_date"),
                "order_estimated_delivery_date": row.get("order_estimated_delivery_date"),
                "payment_type": row.get("payment_type") or "Unknown",
                "payment_installments": row.get("payment_installments") or "Unknown",
                "order_total": 0.0,
                "items": [],
                "source": "BigQuery olist_orders_dataset"
            }

        price = safe_float(row.get("price"))
        freight = safe_float(row.get("freight_value"))

        line_total = price + freight

        category = (
            row.get("product_category_name_english")
            or row.get("product_category_name")
            or "Unknown"
        )

        # Avoid adding empty item rows for orders with no order_items match.
        if row.get("product_id"):
            orders_by_id[order_id]["items"].append({
                "order_item_id": row.get("order_item_id"),
                "product_id": row.get("product_id"),
                "category": category,
                "seller_id": row.get("seller_id"),
                "price": price,
                "freight_value": freight,
                "payment_value": line_total,
                "status": row.get("order_status")
            })

        # Payment value appears repeated across item join rows.
        # So we use item totals here for display consistency.
        orders_by_id[order_id]["order_total"] += line_total

    return list(orders_by_id.values())


def get_generated_customer_orders(customer_id):
    """
    Gets app-created orders from today's output CSV in GCS.
    """

    df = load_data()

    if df.empty or "customer_id" not in df.columns:
        return []

    customer_rows = df[
        df["customer_id"].astype(str) == str(customer_id)
    ]

    if customer_rows.empty:
        return []

    if "order_purchase_timestamp" in customer_rows.columns:
        customer_rows = customer_rows.sort_values(
            "order_purchase_timestamp",
            ascending=False
        )

    orders = []

    for order_id, order_df in customer_rows.groupby("order_id", dropna=False):
        item_list = []
        order_total = 0.0

        for _, row in order_df.iterrows():
            price = safe_float(row.get("price"))
            freight = safe_float(row.get("freight_value"))
            payment_value = safe_float(row.get("payment_value"))

            if payment_value == 0:
                line_total = price + freight
            else:
                line_total = payment_value

            order_total += line_total

            category = row.get("product_category_name_english")

            if pd.isna(category) or not category:
                category = row.get("product_category_name")

            item_list.append({
                "order_item_id": row.get("order_item_id"),
                "product_id": row.get("product_id"),
                "category": category or "Unknown",
                "seller_id": row.get("seller_id"),
                "price": price,
                "freight_value": freight,
                "payment_value": line_total,
                "status": row.get("order_status")
            })

        first_row = order_df.iloc[0]

        orders.append({
            "order_id": order_id,
            "order_status": first_row.get("order_status"),
            "order_purchase_timestamp": first_row.get("order_purchase_timestamp"),
            "order_approved_at": first_row.get("order_approved_at"),
            "order_delivered_carrier_date": first_row.get("order_delivered_carrier_date"),
            "order_delivered_customer_date": first_row.get("order_delivered_customer_date"),
            "order_estimated_delivery_date": first_row.get("order_estimated_delivery_date"),
            "payment_type": first_row.get("payment_type"),
            "payment_installments": first_row.get("payment_installments"),
            "order_total": order_total,
            "items": item_list,
            "source": get_output_file_name()
        })

    return orders


def get_customer_orders(customer_id):
    """
    Combines:
    1. Historical orders from BigQuery
    2. App-created orders from today's GCS output CSV
    """

    existing_orders = get_existing_customer_orders(customer_id)
    generated_orders = get_generated_customer_orders(customer_id)

    all_orders = existing_orders + generated_orders

    all_orders.sort(
        key=lambda order: str(order.get("order_purchase_timestamp") or ""),
        reverse=True
    )

    return all_orders


# ============================================================
# ROUTES
# ============================================================

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        customer_id = request.form.get("customer_id")

        if not customer_id or customer_id.strip() == "":
            return render_template(
                "login.html",
                error="Please enter a customer ID."
            )

        session["customer_id"] = customer_id.strip()

        return redirect(url_for("form"))

    return render_template("login.html", error=None)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
def form():
    if not require_login():
        return redirect(url_for("login"))

    customer_id = get_logged_in_customer_id()
    customer_profile = get_customer_profile(customer_id)
    products = get_products()

    return render_template(
        "form.html",
        customer_id=customer_id,
        customer_profile=customer_profile,
        products=products
    )


@app.route("/orders")
def orders():
    if not require_login():
        return redirect(url_for("login"))

    customer_id = get_logged_in_customer_id()
    customer_profile = get_customer_profile(customer_id)
    customer_orders = get_customer_orders(customer_id)

    return render_template(
        "orders.html",
        customer_id=customer_id,
        customer_profile=customer_profile,
        orders=customer_orders
    )


@app.route("/submit", methods=["POST"])
def submit():
    if not require_login():
        return redirect(url_for("login"))

    df = load_data()
    form_data = request.form

    order_id = str(uuid.uuid4())
    customer_id = get_logged_in_customer_id()
    customer_info = get_customer_info(customer_id)

    payment_type = form_data.get("payment_type") or "credit_card"
    payment_installments = safe_int(
        form_data.get("payment_installments"),
        default=1
    )

    if payment_installments < 1:
        payment_installments = 1

    purchase_time = now_string()

    order_status = "processing"
    order_approved_at = now_string()
    order_delivered_carrier_date = future_date_string(2)
    order_delivered_customer_date = future_date_string(7)
    order_estimated_delivery_date = future_date_string(10)

    order_items = []
    i = 0

    while f"product_id_{i}" in form_data:
        product_id = form_data.get(f"product_id_{i}")
        quantity = safe_int(form_data.get(f"quantity_{i}"), default=1)

        if quantity < 1:
            quantity = 1

        if product_id:
            order_items.append({
                "product_id": product_id,
                "quantity": quantity
            })

        i += 1

    if not order_items:
        print("⚠️ No items submitted. Nothing saved.")
        return redirect(url_for("form"))

    rows = []
    payment_total = 0.0

    for item in order_items:
        product_id = item["product_id"]
        quantity = item["quantity"]

        enrichment = get_product_enrichment(product_id)

        if enrichment is None:
            print(f"⚠️ Product not found in BigQuery: {product_id}")
            continue

        price = safe_float(enrichment.get("price"))
        freight_value = safe_float(enrichment.get("freight_value"))

        for _ in range(quantity):
            payment_value = price + freight_value
            payment_total += payment_value

            row = {
                "order_id": order_id,
                "order_item_id": len(rows) + 1,
                "product_id": product_id,
                "seller_id": enrichment.get("seller_id"),
                "shipping_limit_date": future_date_string(3),
                "price": price,
                "freight_value": freight_value,

                "customer_id": customer_info.get("customer_id"),
                "order_status": order_status,
                "order_purchase_timestamp": purchase_time,
                "order_approved_at": order_approved_at,
                "order_delivered_carrier_date": order_delivered_carrier_date,
                "order_delivered_customer_date": order_delivered_customer_date,
                "order_estimated_delivery_date": order_estimated_delivery_date,

                "seller_zip_code_prefix": enrichment.get("seller_zip_code_prefix"),
                "seller_lat": enrichment.get("seller_lat"),
                "seller_lng": enrichment.get("seller_lng"),
                "seller_city": enrichment.get("seller_city"),
                "seller_state": enrichment.get("seller_state"),

                "customer_unique_id": customer_info.get("customer_unique_id"),
                "customer_zip_code_prefix": customer_info.get("customer_zip_code_prefix"),
                "customer_city": customer_info.get("customer_city"),
                "customer_state": customer_info.get("customer_state"),
                "customer_lat": customer_info.get("customer_lat"),
                "customer_lng": customer_info.get("customer_lng"),

                "product_category_name": enrichment.get("product_category_name"),
                "product_category_name_english": enrichment.get("product_category_name_english"),
                "product_name_length": enrichment.get("product_name_length"),
                "product_description_length": enrichment.get("product_description_length"),
                "product_photos_qty": enrichment.get("product_photos_qty"),
                "product_weight_g": enrichment.get("product_weight_g"),
                "product_length_cm": enrichment.get("product_length_cm"),
                "product_height_cm": enrichment.get("product_height_cm"),
                "product_width_cm": enrichment.get("product_width_cm"),

                "payment_sequential": 1,
                "payment_type": payment_type,
                "payment_installments": payment_installments,
                "payment_value": payment_value
            }

            rows.append(row)

    if not rows:
        print("⚠️ No valid rows were created. Nothing saved.")
        return redirect(url_for("form"))

    new_df = pd.DataFrame(rows)

    if df.empty:
        df = new_df
    else:
        df = pd.concat([df, new_df], ignore_index=True)

    save_data(df)

    print(f"✅ Created order {order_id} with {len(rows)} order item rows.")
    print(f"✅ Estimated payment total: {payment_total}")

    return redirect(url_for("orders"))


@app.route("/test-gcs")
def test_gcs():
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blobs = list(bucket.list_blobs(prefix=OUTPUT_PREFIX, max_results=50))

        if not blobs:
            return f"No files found under gs://{BUCKET_NAME}/{OUTPUT_PREFIX}/"

        return "<br>".join([blob.name for blob in blobs])

    except Exception as e:
        return f"Error: {e}"


@app.route("/test-bq")
def test_bq():
    query = f"""
        SELECT COUNT(*) AS order_count
        FROM {bq_table(BQ_ORDERS_TABLE)}
    """

    rows = run_bq_query(query)

    if not rows:
        return "BigQuery test failed. Check logs."

    return f"BigQuery connected. Orders table count: {rows[0].get('order_count')}"


@app.route("/debug-config")
def debug_config():
    return f"""
    <h3>Config</h3>
    <p><strong>Bucket:</strong> {BUCKET_NAME}</p>
    <p><strong>Output Prefix:</strong> {OUTPUT_PREFIX}</p>
    <p><strong>Output File:</strong> {get_output_file_name()}</p>
    <p><strong>Output Path:</strong> gs://{BUCKET_NAME}/{output_gcs_path(get_output_file_name())}</p>

    <p><strong>BigQuery Project:</strong> {BQ_PROJECT_ID}</p>
    <p><strong>BigQuery Dataset:</strong> {BQ_DATASET_ID}</p>
    <p><strong>Orders Table:</strong> {BQ_ORDERS_TABLE}</p>
    <p><strong>Products Table:</strong> {BQ_PRODUCTS_TABLE}</p>
    """


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)