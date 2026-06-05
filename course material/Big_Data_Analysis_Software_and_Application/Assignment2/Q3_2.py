#!/usr/bin/env python3
"""
Q3_2: Find the difference in average annual consumption between segments
for customers with minimum consumption in 2013 (CZK currency).
Uses Spark SQL API.

Formula: avg_consumption = total_consumption_of_segment / count_of_min_customers_in_segment

Run with: spark-submit --jars ./sqlite-jdbc-3.51.3.0.jar Q3_2.py
"""

import os
from pyspark.sql import SparkSession

def main():
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    jdbc_jar = os.path.join(script_dir, "sqlite-jdbc-3.51.3.0.jar")
    sqlite_db = os.path.join(script_dir, "debit_card_specializing/debit_card_specializing.sqlite")
    
    spark = SparkSession.builder \
        .appName("Q3_2_MinConsumption_SQL") \
        .config("spark.jars", jdbc_jar) \
        .getOrCreate()
    
    jdbc_url = f"jdbc:sqlite:{sqlite_db}"
    jdbc_driver = "org.sqlite.JDBC"
    
    # Load tables
    df_customers = spark.read.format("jdbc") \
        .option("url", jdbc_url) \
        .option("dbtable", "customers") \
        .option("driver", jdbc_driver) \
        .load()
    
    df_yearmonth = spark.read.format("jdbc") \
        .option("url", jdbc_url) \
        .option("dbtable", "yearmonth") \
        .option("driver", jdbc_driver) \
        .load()
    
    # Register as temporary views for SQL
    df_customers.createOrReplaceTempView("customers")
    df_yearmonth.createOrReplaceTempView("yearmonth")
    
    print("=" * 80)
    print("Q3_2: Minimum Consumption Analysis for 2013 (CZK) - SQL API")
    print("=" * 80)
    
    # SQL query to solve the problem
    query = """
    WITH 
    -- Step 1: Filter 2013 data and join with customers (CZK only)
    data_2013_czk AS (
        SELECT 
            y.CustomerID,
            y.Date,
            y.Consumption,
            c.Segment,
            c.Currency
        FROM yearmonth y
        INNER JOIN customers c ON y.CustomerID = c.CustomerID
        WHERE y.Date >= '201301' 
          AND y.Date <= '201312'
          AND c.Currency = 'CZK'
    ),
    
    -- Step 2: Calculate total consumption per customer in 2013
    customer_totals AS (
        SELECT 
            CustomerID,
            Segment,
            SUM(Consumption) AS total_consumption
        FROM data_2013_czk
        GROUP BY CustomerID, Segment
    ),
    
    -- Step 3: Find minimum consumption value for each segment
    segment_min_vals AS (
        SELECT 
            Segment,
            MIN(total_consumption) AS min_consumption_val
        FROM customer_totals
        GROUP BY Segment
    ),
    
    -- Step 4: Find customers who have the minimum consumption in their segment
    min_customers AS (
        SELECT 
            ct.CustomerID,
            ct.Segment,
            ct.total_consumption
        FROM customer_totals ct
        INNER JOIN segment_min_vals smv 
            ON ct.Segment = smv.Segment 
            AND ct.total_consumption = smv.min_consumption_val
    ),
    
    -- Step 5: Count min-customers per segment
    min_customer_counts AS (
        SELECT 
            Segment,
            COUNT(CustomerID) AS min_customer_count
        FROM min_customers
        GROUP BY Segment
    ),
    
    -- Step 6: Calculate total consumption for each segment (all customers)
    segment_totals AS (
        SELECT 
            Segment,
            SUM(total_consumption) AS segment_total_consumption
        FROM customer_totals
        GROUP BY Segment
    ),
    
    -- Step 7: Calculate average = segment_total / count_of_min_customers
    avg_consumption AS (
        SELECT 
            st.Segment,
            st.segment_total_consumption,
            mcc.min_customer_count,
            st.segment_total_consumption / mcc.min_customer_count AS avg_consumption
        FROM segment_totals st
        INNER JOIN min_customer_counts mcc ON st.Segment = mcc.Segment
    )
    
    SELECT * FROM avg_consumption
    ORDER BY Segment
    """
    
    # Execute the query
    result = spark.sql(query)
    
    print("\n--- Average Annual Consumption for Min Customers per Segment ---")
    print("Formula: segment_total_consumption / min_customer_count")
    result.show()
    
    # Collect results for calculation
    results = result.collect()
    
    segment_values = {}
    for row in results:
        segment_values[row["Segment"]] = row["avg_consumption"]
    
    sme_val = segment_values.get("SME", 0)
    lam_val = segment_values.get("LAM", 0)
    kam_val = segment_values.get("KAM", 0)
    
    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)
    print(f"SME average: {sme_val:.2f}")
    print(f"LAM average: {lam_val:.2f}")
    print(f"KAM average: {kam_val:.2f}")
    print()
    print(f"SME - LAM difference: {sme_val - lam_val:.2f}")
    print(f"LAM - KAM difference: {lam_val - kam_val:.2f}")
    print(f"KAM - SME difference: {kam_val - sme_val:.2f}")
    print("=" * 80)
    
    # Save results
    output_dir = os.path.join(script_dir, "../Q3_2_output")
    result.write.mode("overwrite").csv(
        output_dir,
        header=True
    )
    print(f"\nOutput saved to {output_dir}")
    
    spark.stop()

if __name__ == "__main__":
    main()
