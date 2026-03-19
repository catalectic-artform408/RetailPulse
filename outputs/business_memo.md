# Business Memo — Olist Customer Intelligence & Revenue Optimization

## Situation

Olist, Brazil's largest marketplace aggregator, processes 100K+ orders across 9 relational datasets (2016–2018). With a median 90-day repurchase rate under 1%, the customer base is predominantly one-time buyers. This analysis examines the drivers of churn, identifies high-value segments for targeted retention investment, and quantifies the revenue opportunity from reducing customer attrition.

## Key Findings

- **Champions segment** (top 7% of customers by RFM score) **generates 13.1% of total revenue** (R$2.0M), demonstrating the outsized value of repeat buyers.

- **At-Risk segment** comprises **22,229 customers with R$3.7M in lifetime value** — these were once active buyers who have gone quiet. Combined with the "Can't Lose Them" segment (8,671 customers, R$2.1M), **R$5.8M in revenue is at risk** of permanent loss.

- **Late deliveries directly depress review scores**: on-time orders average **4.1★** vs late orders at **2.5★** — a 1.6-point drop. Late delivery rate stands at **8.1%**, with 206 of 1,238 active sellers flagged as underperformers.

- São Paulo state accounts for **41% of all orders** and **R$5.8M in revenue** but has the lowest average cart value (R$142), suggesting room for upselling. Bahia has the highest average cart value at R$182.

- **Churn prediction model** (ROC-AUC: 1.00) identified **R$12.3M in total revenue at risk** from 74.8K churned customers. Top churn drivers: days since first order, total spend, and delivery delay.

## Recommendations

1. **Launch a reactivation campaign** targeting the At-Risk segment (22K customers) with a 10% discount voucher. If even 8% convert, recovered revenue = **R$297K** — ROI-positive at any reasonable campaign cost.

2. **Flag sellers with on-time rate below 85%** for performance review — they are directly linked to 40%+ of 1-star reviews. Implement delivery SLA enforcement with progressive penalties.

3. **Introduce loyalty tiers** for Champions and Loyal Customers (33K combined) to increase repeat purchase frequency from the current avg of 1.05× to 1.3×/year — potential uplift of **R$1.4M annually**.

4. **Target São Paulo for upsell campaigns** — highest volume (41% of orders) but lowest AOV (R$142 vs R$182 in Bahia). Category-specific bundle offers on Health & Beauty and Watches (highest revenue but lowest repeat rate).

## Next Steps

- Deploy churn model scoring pipeline to flag new at-risk customers weekly
- A/B test reactivation email campaign for At-Risk segment in Q2
- Pilot seller performance tiers with delivery SLA enforcement
- Monitor cohort retention heatmap monthly to measure campaign impact
- Expand analysis to include seasonal forecasting for inventory optimization
