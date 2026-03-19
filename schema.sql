-- =============================================================================
-- Olist E-Commerce Database Schema
-- SQLite version (PostgreSQL equivalents noted in comments)
-- =============================================================================

-- Geolocation reference table
CREATE TABLE IF NOT EXISTS geolocation (
    geolocation_zip_code_prefix TEXT,
    geolocation_lat REAL,           -- PostgreSQL: DOUBLE PRECISION
    geolocation_lng REAL,
    geolocation_city TEXT,
    geolocation_state TEXT
);

-- Customers
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    customer_unique_id TEXT NOT NULL,
    customer_zip_code_prefix TEXT,
    customer_city TEXT,
    customer_state TEXT
);

-- Sellers
CREATE TABLE IF NOT EXISTS sellers (
    seller_id TEXT PRIMARY KEY,
    seller_zip_code_prefix TEXT,
    seller_city TEXT,
    seller_state TEXT
);

-- Products
CREATE TABLE IF NOT EXISTS products (
    product_id TEXT PRIMARY KEY,
    product_category_name TEXT,
    product_name_lenght INTEGER,        -- original typo in dataset
    product_description_lenght INTEGER, -- original typo in dataset
    product_photos_qty INTEGER,
    product_weight_g INTEGER,
    product_length_cm INTEGER,
    product_height_cm INTEGER,
    product_width_cm INTEGER
);

-- Category name translation
CREATE TABLE IF NOT EXISTS category_translation (
    product_category_name TEXT PRIMARY KEY,
    product_category_name_english TEXT
);

-- Orders (central fact table)
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    order_status TEXT,                  -- PostgreSQL: VARCHAR(20)
    order_purchase_timestamp TEXT,      -- PostgreSQL: TIMESTAMP
    order_approved_at TEXT,
    order_delivered_carrier_date TEXT,
    order_delivered_customer_date TEXT,
    order_estimated_delivery_date TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Order Items (bridge table: orders ↔ products ↔ sellers)
CREATE TABLE IF NOT EXISTS order_items (
    order_id TEXT NOT NULL,
    order_item_id INTEGER NOT NULL,
    product_id TEXT NOT NULL,
    seller_id TEXT NOT NULL,
    shipping_limit_date TEXT,
    price REAL NOT NULL,
    freight_value REAL NOT NULL,
    PRIMARY KEY (order_id, order_item_id),
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id),
    FOREIGN KEY (seller_id) REFERENCES sellers(seller_id)
);

-- Order Payments
CREATE TABLE IF NOT EXISTS order_payments (
    order_id TEXT NOT NULL,
    payment_sequential INTEGER NOT NULL,
    payment_type TEXT,
    payment_installments INTEGER,
    payment_value REAL,
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

-- Order Reviews
CREATE TABLE IF NOT EXISTS order_reviews (
    review_id TEXT,
    order_id TEXT NOT NULL,
    review_score INTEGER,
    review_comment_title TEXT,
    review_comment_message TEXT,
    review_creation_date TEXT,
    review_answer_timestamp TEXT,
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(order_status);
CREATE INDEX IF NOT EXISTS idx_orders_purchase ON orders(order_purchase_timestamp);
CREATE INDEX IF NOT EXISTS idx_order_items_order ON order_items(order_id);
CREATE INDEX IF NOT EXISTS idx_order_items_product ON order_items(product_id);
CREATE INDEX IF NOT EXISTS idx_order_items_seller ON order_items(seller_id);
CREATE INDEX IF NOT EXISTS idx_payments_order ON order_payments(order_id);
CREATE INDEX IF NOT EXISTS idx_reviews_order ON order_reviews(order_id);
CREATE INDEX IF NOT EXISTS idx_geolocation_zip ON geolocation(geolocation_zip_code_prefix);
CREATE INDEX IF NOT EXISTS idx_customers_state ON customers(customer_state);
CREATE INDEX IF NOT EXISTS idx_sellers_state ON sellers(seller_state);
