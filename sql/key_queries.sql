-- =============================================================================
-- Key SQL Queries — Olist E-Commerce Analysis
-- Run these against the SQLite database (data/olist.db)
-- =============================================================================

-- ─────────────────────────────────────────────────────────────────────────────
-- 1. Monthly Revenue Trend
-- Shows revenue trajectory over time — the most fundamental business metric.
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    strftime('%Y-%m', o.order_purchase_timestamp) AS month,
    ROUND(SUM(p.payment_value), 2) AS revenue,
    COUNT(DISTINCT o.order_id) AS order_count
FROM orders o
JOIN order_payments p ON o.order_id = p.order_id
WHERE o.order_status = 'delivered'
GROUP BY 1
ORDER BY 1;


-- ─────────────────────────────────────────────────────────────────────────────
-- 2. Delivery Delay Impact on Review Score
-- Proves that operational quality directly affects customer satisfaction.
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    CASE
        WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date
        THEN 'Late'
        ELSE 'On-time'
    END AS delivery_status,
    ROUND(AVG(r.review_score), 2) AS avg_review_score,
    COUNT(*) AS order_count
FROM orders o
JOIN order_reviews r ON o.order_id = r.order_id
WHERE o.order_delivered_customer_date IS NOT NULL
  AND o.order_estimated_delivery_date IS NOT NULL
GROUP BY 1;


-- ─────────────────────────────────────────────────────────────────────────────
-- 3. Top 10 Revenue Categories (CTE — demonstrates advanced SQL)
-- Combines joins across 4 tables and uses a CTE for readability.
-- ─────────────────────────────────────────────────────────────────────────────
WITH cat_revenue AS (
    SELECT
        COALESCE(t.product_category_name_english, p.product_category_name) AS category,
        ROUND(SUM(i.price), 2) AS revenue,
        COUNT(DISTINCT o.order_id) AS orders,
        ROUND(AVG(i.price), 2) AS avg_price
    FROM order_items i
    JOIN products p ON i.product_id = p.product_id
    LEFT JOIN category_translation t ON p.product_category_name = t.product_category_name
    JOIN orders o ON i.order_id = o.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY 1
)
SELECT * FROM cat_revenue
ORDER BY revenue DESC
LIMIT 10;


-- ─────────────────────────────────────────────────────────────────────────────
-- 4. Seller Performance Ranking (Window Function)
-- Demonstrates RANK(), aggregation, and multi-table joins.
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    i.seller_id,
    s.seller_city,
    s.seller_state,
    ROUND(SUM(i.price), 2) AS total_revenue,
    COUNT(DISTINCT i.order_id) AS order_count,
    ROUND(AVG(r.review_score), 2) AS avg_rating,
    RANK() OVER (ORDER BY SUM(i.price) DESC) AS revenue_rank
FROM order_items i
JOIN sellers s ON i.seller_id = s.seller_id
JOIN orders o ON i.order_id = o.order_id
JOIN order_reviews r ON o.order_id = r.order_id
WHERE o.order_status = 'delivered'
GROUP BY i.seller_id, s.seller_city, s.seller_state
ORDER BY revenue_rank
LIMIT 20;
