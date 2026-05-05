from flask import Flask, render_template, request, redirect, session, url_for
from google.oauth2 import service_account
from google.cloud import storage, bigquery
import pandas as pd
from io import BytesIO
import uuid
from datetime import datetime, timedelta

# ----------------------------
# CONFIG
# ----------------------------
KEY_PATH = "input_app/key.json"

# Bucket name only. Folder path goes into DATA_PREFIX.
BUCKET_NAME = "olist-494110_bucket"
DATA_PREFIX = "archive (5)"

MAIN_FILE = "merged_data.csv"

CUSTOMERS_FILE = "olist_customers_dataset.csv"
GEOLOCATION_FILE = "olist_geolocation_dataset.csv"
ORDER_ITEMS_FILE = "olist_order_items_dataset.csv"
ORDER_PAYMENTS_FILE = "olist_order_payments_dataset.csv"
ORDER_REVIEWS_FILE = "olist_order_reviews_dataset.csv"
ORDERS_FILE = "olist_orders_dataset.csv"
PRODUCTS_FILE = "olist_products_dataset.csv"
SELLERS_FILE = "olist_sellers_dataset.csv"
CATEGORY_TRANSLATION_FILE = "product_category_name_translation.csv"

# Flask session secret.
# For assignment/demo this is fine. In production, use an environment variable.
SECRET_KEY = "dev-secret-key-change-later"

# ----------------------------
# AUTH
# ----------------------------
credentials = service_account.Credentials.from_service_account_file(KEY_PATH)

storage_client = storage.Client(credentials=credentials)
bq_client = bigquery.Client(credentials=credentials)

# ----------------------------
# APP
# ----------------------------
app = Flask(__name__)
app.secret_key = SECRET_KEY


# ----------------------------
# GCS HELPERS
# ----------------------------
def gcs_path(file_name):
    if DATA_PREFIX:
        return f"{DATA_PREFIX}/{file_name}"
    return file_name


def read_gcs_csv(file_name):
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(gcs_path(file_name))

    if not blob.exists():
        print(f"⚠️ File not found in GCS: gs://{BUCKET_NAME}/{gcs_path(file_name)}")
        return pd.DataFrame()

    try:
        data = blob.download_as_bytes()
        df = pd.read_csv(BytesIO(data))
        print(f"✅ Loaded {file_name}: {len(df)} rows")
        return df
    except Exception as e:
        print(f"❌ Error reading {file_name}: {e}")
        return pd.DataFrame()


def write_gcs_csv(df, file_name):
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(gcs_path(file_name))

    try:
        csv_data = df.to_csv(index=False)
        blob.upload_from_string(csv_data, content_type="text/csv")
        print(f"✅ Uploaded to gs://{BUCKET_NAME}/{gcs_path(file_name)}")
    except Exception as e:
        print(f"❌ Error writing {file_name}: {e}")


# ----------------------------
# MAIN DATA LOAD/SAVE
# ----------------------------
def load_data():
    return read_gcs_csv(MAIN_FILE)


def save_data(df):
    write_gcs_csv(df, MAIN_FILE)


# ----------------------------
# DISPLAY HELPERS
# ----------------------------
def clean_display_value(value, suffix=""):
    if pd.isna(value) or value == "":
        return "unknown"

    try:
        if float(value).is_integer():
            value = int(float(value))
    except Exception:
        pass

    return f"{value}{suffix}"


def clean_money(value):
    if pd.isna(value) or value == "":
        return "unknown"

    try:
        return f"${float(value):.2f}"
    except Exception:
        return str(value)


def build_product_display_name(row):
    product_id = row.get("product_id")

    category_en = row.get("product_category_name_english")
    category_raw = row.get("product_category_name")

    if pd.notna(category_en) and category_en:
        category = category_en
    elif pd.notna(category_raw) and category_raw:
        category = category_raw
    else:
        category = "unknown category"

    return f"{category} — {product_id}"


# ----------------------------
# LOAD REFERENCE DATA
# ----------------------------
def load_reference_data():
    order_items_df = read_gcs_csv(ORDER_ITEMS_FILE)
    products_df = read_gcs_csv(PRODUCTS_FILE)
    sellers_df = read_gcs_csv(SELLERS_FILE)
    customers_df = read_gcs_csv(CUSTOMERS_FILE)
    category_translation_df = read_gcs_csv(CATEGORY_TRANSLATION_FILE)

    return {
        "order_items": order_items_df,
        "products": products_df,
        "sellers": sellers_df,
        "customers": customers_df,
        "category_translation": category_translation_df
    }


# ----------------------------
# PRODUCT DROPDOWN DATA
# ----------------------------
def get_products():
    products_df = read_gcs_csv(PRODUCTS_FILE)
    translation_df = read_gcs_csv(CATEGORY_TRANSLATION_FILE)
    order_items_df = read_gcs_csv(ORDER_ITEMS_FILE)

    if products_df.empty or "product_id" not in products_df.columns:
        return []

    keep_cols = [
        "product_id",
        "product_category_name",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm"
    ]

    existing_cols = [col for col in keep_cols if col in products_df.columns]

    products_df = (
        products_df[existing_cols]
        .dropna(subset=["product_id"])
        .drop_duplicates()
    )

    if (
        not translation_df.empty
        and "product_category_name" in translation_df.columns
        and "product_category_name_english" in translation_df.columns
        and "product_category_name" in products_df.columns
    ):
        products_df = products_df.merge(
            translation_df,
            on="product_category_name",
            how="left"
        )
    else:
        products_df["product_category_name_english"] = None

    if (
        not order_items_df.empty
        and "product_id" in order_items_df.columns
        and "price" in order_items_df.columns
        and "freight_value" in order_items_df.columns
    ):
        price_df = (
            order_items_df
            .sort_values("price")
            .groupby("product_id", as_index=False)
            .first()[["product_id", "seller_id", "price", "freight_value"]]
        )

        products_df = products_df.merge(
            price_df,
            on="product_id",
            how="left"
        )
    else:
        products_df["seller_id"] = None
        products_df["price"] = None
        products_df["freight_value"] = None

    products_df["display_name"] = products_df.apply(
        build_product_display_name,
        axis=1
    )

    return products_df[[
        "product_id",
        "display_name",
        "price",
        "freight_value",
        "product_category_name_english",
        "product_category_name",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm"
    ]].head(100).to_dict("records")


# ----------------------------
# CUSTOMER HELPERS
# ----------------------------
def get_customer_info(customer_id, customers_df):
    fallback = {
        "customer_id": customer_id,
        "customer_unique_id": None,
        "customer_zip_code_prefix": None,
        "customer_city": None,
        "customer_state": None,
        "customer_lat": None,
        "customer_lng": None,
        "is_existing_customer": False
    }

    if customers_df.empty or "customer_id" not in customers_df.columns:
        return fallback

    matches = customers_df[customers_df["customer_id"] == customer_id]

    if matches.empty:
        return fallback

    customer = matches.iloc[0]

    return {
        "customer_id": customer.get("customer_id"),
        "customer_unique_id": customer.get("customer_unique_id"),
        "customer_zip_code_prefix": customer.get("customer_zip_code_prefix"),
        "customer_city": customer.get("customer_city"),
        "customer_state": customer.get("customer_state"),
        "customer_lat": None,
        "customer_lng": None,
        "is_existing_customer": True
    }


def get_logged_in_customer_id():
    return session.get("customer_id")


def require_login():
    if not get_logged_in_customer_id():
        return False
    return True

def get_customer_profile(customer_id):
    """
    Gets customer profile from olist_customers_dataset.csv
    and order summary from olist_orders_dataset.csv.
    """

    customers_df = read_gcs_csv(CUSTOMERS_FILE)
    orders_df = read_gcs_csv(ORDERS_FILE)

    profile = {
        "customer_id": customer_id,
        "customer_unique_id": None,
        "customer_zip_code_prefix": None,
        "customer_city": None,
        "customer_state": None,
        "exists_in_customers_dataset": False,
        "order_count": 0,
        "latest_order_date": None
    }

    # ----------------------------
    # CUSTOMER DETAILS
    # ----------------------------
    if not customers_df.empty and "customer_id" in customers_df.columns:
        matches = customers_df[
            customers_df["customer_id"].astype(str) == str(customer_id)
        ]

        if not matches.empty:
            customer = matches.iloc[0]

            profile.update({
                "customer_unique_id": customer.get("customer_unique_id"),
                "customer_zip_code_prefix": customer.get("customer_zip_code_prefix"),
                "customer_city": customer.get("customer_city"),
                "customer_state": customer.get("customer_state"),
                "exists_in_customers_dataset": True
            })

    # ----------------------------
    # ORDER SUMMARY
    # ----------------------------
    if not orders_df.empty and "customer_id" in orders_df.columns:
        customer_orders = orders_df[
            orders_df["customer_id"].astype(str) == str(customer_id)
        ]

        profile["order_count"] = len(customer_orders)

        if (
            not customer_orders.empty
            and "order_purchase_timestamp" in customer_orders.columns
        ):
            profile["latest_order_date"] = (
                customer_orders["order_purchase_timestamp"]
                .dropna()
                .sort_values(ascending=False)
                .iloc[0]
            )

    return profile


# ----------------------------
# PRODUCT / SELLER ENRICHMENT
# ----------------------------
def get_product_enrichment(product_id, ref_data):
    order_items_df = ref_data["order_items"]
    products_df = ref_data["products"]
    sellers_df = ref_data["sellers"]
    category_translation_df = ref_data["category_translation"]

    if order_items_df.empty or "product_id" not in order_items_df.columns:
        return None

    item_matches = order_items_df[order_items_df["product_id"] == product_id]

    if item_matches.empty:
        return None

    if "price" in item_matches.columns:
        chosen_item = item_matches.sort_values("price").iloc[0]
    else:
        chosen_item = item_matches.iloc[0]

    seller_id = chosen_item.get("seller_id")
    price = chosen_item.get("price")
    freight_value = chosen_item.get("freight_value")

    enrichment = {
        "seller_id": seller_id,
        "price": price,
        "freight_value": freight_value,

        "seller_zip_code_prefix": None,
        "seller_lat": None,
        "seller_lng": None,
        "seller_city": None,
        "seller_state": None,

        "product_category_name": None,
        "product_category_name_english": None,
        "product_name_length": None,
        "product_description_length": None,
        "product_photos_qty": None,
        "product_weight_g": None,
        "product_length_cm": None,
        "product_height_cm": None,
        "product_width_cm": None
    }

    if not sellers_df.empty and "seller_id" in sellers_df.columns:
        seller_matches = sellers_df[sellers_df["seller_id"] == seller_id]

        if not seller_matches.empty:
            seller = seller_matches.iloc[0]
            enrichment.update({
                "seller_zip_code_prefix": seller.get("seller_zip_code_prefix"),
                "seller_city": seller.get("seller_city"),
                "seller_state": seller.get("seller_state")
            })

    if not products_df.empty and "product_id" in products_df.columns:
        product_matches = products_df[products_df["product_id"] == product_id]

        if not product_matches.empty:
            product = product_matches.iloc[0]

            category_name = product.get("product_category_name")

            enrichment.update({
                "product_category_name": category_name,
                "product_name_length": product.get("product_name_lenght"),
                "product_description_length": product.get("product_description_lenght"),
                "product_photos_qty": product.get("product_photos_qty"),
                "product_weight_g": product.get("product_weight_g"),
                "product_length_cm": product.get("product_length_cm"),
                "product_height_cm": product.get("product_height_cm"),
                "product_width_cm": product.get("product_width_cm")
            })

            if (
                category_name is not None
                and not category_translation_df.empty
                and "product_category_name" in category_translation_df.columns
            ):
                translation_matches = category_translation_df[
                    category_translation_df["product_category_name"] == category_name
                ]

                if not translation_matches.empty:
                    translation = translation_matches.iloc[0]
                    enrichment["product_category_name_english"] = translation.get(
                        "product_category_name_english"
                    )

    return enrichment


# ----------------------------
# ORDER VIEW HELPERS
# ----------------------------
def get_customer_orders(customer_id):
    """
    Pulls existing customer orders from the real Olist dataset.

    Main source:
    - olist_orders_dataset.csv

    Related detail sources:
    - olist_order_items_dataset.csv
    - olist_order_payments_dataset.csv
    - olist_products_dataset.csv
    - product_category_name_translation.csv
    """

    orders_df = read_gcs_csv(ORDERS_FILE)
    order_items_df = read_gcs_csv(ORDER_ITEMS_FILE)
    payments_df = read_gcs_csv(ORDER_PAYMENTS_FILE)
    products_df = read_gcs_csv(PRODUCTS_FILE)
    translation_df = read_gcs_csv(CATEGORY_TRANSLATION_FILE)

    if orders_df.empty:
        print("⚠️ Could not load olist_orders_dataset.csv")
        return []

    if "customer_id" not in orders_df.columns:
        print("⚠️ olist_orders_dataset.csv has no customer_id column")
        return []

    customer_orders_df = orders_df[
        orders_df["customer_id"].astype(str) == str(customer_id)
    ]

    if customer_orders_df.empty:
        return []

    if "order_purchase_timestamp" in customer_orders_df.columns:
        customer_orders_df = customer_orders_df.sort_values(
            "order_purchase_timestamp",
            ascending=False
        )

    orders = []

    for _, order_row in customer_orders_df.iterrows():
        order_id = order_row.get("order_id")

        # ----------------------------
        # Get items for this order
        # ----------------------------
        if not order_items_df.empty and "order_id" in order_items_df.columns:
            items_for_order = order_items_df[
                order_items_df["order_id"].astype(str) == str(order_id)
            ]
        else:
            items_for_order = pd.DataFrame()

        # ----------------------------
        # Get payments for this order
        # ----------------------------
        if not payments_df.empty and "order_id" in payments_df.columns:
            payments_for_order = payments_df[
                payments_df["order_id"].astype(str) == str(order_id)
            ]
        else:
            payments_for_order = pd.DataFrame()

        payment_type = "Unknown"
        payment_installments = "Unknown"
        order_total = 0.0

        if not payments_for_order.empty:
            first_payment = payments_for_order.iloc[0]

            payment_type = first_payment.get("payment_type", "Unknown")
            payment_installments = first_payment.get("payment_installments", "Unknown")

            if "payment_value" in payments_for_order.columns:
                order_total = payments_for_order["payment_value"].apply(safe_float).sum()

        # ----------------------------
        # Build item list
        # ----------------------------
        items = []

        for _, item_row in items_for_order.iterrows():
            product_id = item_row.get("product_id")

            category_raw = None
            category_english = None

            # Product category lookup
            if not products_df.empty and "product_id" in products_df.columns:
                product_matches = products_df[
                    products_df["product_id"].astype(str) == str(product_id)
                ]

                if not product_matches.empty:
                    product = product_matches.iloc[0]
                    category_raw = product.get("product_category_name")

            # Category translation lookup
            if (
                category_raw is not None
                and not translation_df.empty
                and "product_category_name" in translation_df.columns
                and "product_category_name_english" in translation_df.columns
            ):
                translation_matches = translation_df[
                    translation_df["product_category_name"].astype(str) == str(category_raw)
                ]

                if not translation_matches.empty:
                    category_english = translation_matches.iloc[0].get(
                        "product_category_name_english"
                    )

            price = safe_float(item_row.get("price"))
            freight = safe_float(item_row.get("freight_value"))
            line_total = price + freight

            items.append({
                "order_item_id": item_row.get("order_item_id"),
                "product_id": product_id,
                "category": category_english or category_raw or "Unknown",
                "seller_id": item_row.get("seller_id"),
                "price": price,
                "freight_value": freight,
                "payment_value": line_total,
                "status": order_row.get("order_status")
            })

        # If payment table had no total, fallback to item totals.
        if order_total == 0:
            order_total = sum(item["payment_value"] for item in items)

        orders.append({
            "order_id": order_id,
            "order_status": order_row.get("order_status"),
            "order_purchase_timestamp": order_row.get("order_purchase_timestamp"),
            "order_approved_at": order_row.get("order_approved_at"),
            "order_delivered_carrier_date": order_row.get("order_delivered_carrier_date"),
            "order_delivered_customer_date": order_row.get("order_delivered_customer_date"),
            "order_estimated_delivery_date": order_row.get("order_estimated_delivery_date"),
            "payment_type": payment_type,
            "payment_installments": payment_installments,
            "order_total": order_total,
            "items": items
        })

    return orders


def safe_float(value):
    if pd.isna(value) or value == "":
        return 0.0

    try:
        return float(value)
    except Exception:
        return 0.0


# ----------------------------
# DATE HELPERS
# ----------------------------
def now_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def future_date_string(days):
    return (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")


# ----------------------------
# ROUTES
# ----------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        customer_id = request.form.get("customer_id")

        if not customer_id or customer_id.strip() == "":
            return render_template(
                "login.html",
                error="Please enter a customer ID."
            )

        customer_id = customer_id.strip()
        session["customer_id"] = customer_id

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

    products = get_products()
    customer_id = get_logged_in_customer_id()
    customer_profile = get_customer_profile(customer_id)

    return render_template(
        "form.html",
        products=products,
        customer_id=customer_id,
        customer_profile=customer_profile
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
    ref_data = load_reference_data()
    form = request.form

    order_id = str(uuid.uuid4())

    customer_id = get_logged_in_customer_id()
    customer_info = get_customer_info(customer_id, ref_data["customers"])

    payment_type = form.get("payment_type") or "credit_card"

    try:
        payment_installments = int(form.get("payment_installments") or 1)
    except ValueError:
        payment_installments = 1

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

    while f"product_id_{i}" in form:
        product_id = form.get(f"product_id_{i}")
        quantity_raw = form.get(f"quantity_{i}")

        if product_id:
            try:
                quantity = int(quantity_raw) if quantity_raw else 1
            except ValueError:
                quantity = 1

            if quantity < 1:
                quantity = 1

            order_items.append({
                "product_id": product_id,
                "quantity": quantity
            })

        i += 1

    if not order_items:
        print("⚠️ No items submitted. Nothing saved.")
        return redirect(url_for("form"))

    rows = []
    payment_total = 0

    for item in order_items:
        product_id = item["product_id"]
        quantity = item["quantity"]

        enrichment = get_product_enrichment(product_id, ref_data)

        if enrichment is None:
            print(f"⚠️ Product not found in reference data: {product_id}")
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
        blobs = list(bucket.list_blobs(prefix=DATA_PREFIX, max_results=50))

        if not blobs:
            return f"No files found under gs://{BUCKET_NAME}/{DATA_PREFIX}/"

        return "<br>".join([blob.name for blob in blobs])

    except Exception as e:
        return f"Error: {e}"


@app.route("/debug-gcs")
def debug_gcs():
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blobs = list(bucket.list_blobs(prefix=DATA_PREFIX, max_results=100))

        if not blobs:
            return f"""
            <h3>No files found</h3>
            <p>Bucket: {BUCKET_NAME}</p>
            <p>Prefix: {DATA_PREFIX}</p>
            <p>Looking under: gs://{BUCKET_NAME}/{DATA_PREFIX}/</p>
            """

        output = [
            f"<h3>Files found under gs://{BUCKET_NAME}/{DATA_PREFIX}/</h3>"
        ]

        for blob in blobs:
            output.append(f"{blob.name}<br>")

        return "".join(output)

    except Exception as e:
        return f"<h3>Error</h3><pre>{e}</pre>"


# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)