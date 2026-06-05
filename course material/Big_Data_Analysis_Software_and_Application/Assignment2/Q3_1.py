#!/usr/bin/env python3
"""
Q3_1: Find the difference in average annual consumption between segments
for customers with minimum consumption in 2013 (CZK currency).

Formula: avg_consumption = total_consumption_of_segment / count_of_min_customers_in_segment

Where "min customers" = customers with the minimum total_consumption in that segment

Run with: spark-submit --jars ./sqlite-jdbc-3.51.3.0.jar Q3_1.py
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as _sum, count, min as _min

def main():
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    jdbc_jar = os.path.join(script_dir, "sqlite-jdbc-3.51.3.0.jar")
    sqlite_db = os.path.join(script_dir, "debit_card_specializing/debit_card_specializing.sqlite")
    
    spark = SparkSession.builder \
        .appName("Q3_1_MinConsumption") \
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
    
    print("=" * 80)
    print("Q3_1: Minimum Consumption Analysis for 2013 (CZK)")
    print("=" * 80)
    
    # Step 1: Filter 2013 data from yearmonth (Date BETWEEN 201301 AND 201312)
    df_2013 = df_yearmonth.filter(
        (col("Date") >= "201301") & (col("Date") <= "201312")
    )
    
    # Step 2: Join with customers to get Segment and Currency
    df_joined = df_2013.join(df_customers, on="CustomerID", how="inner")
    
    # Step 3: Filter CZK currency
    df_czk = df_joined.filter(col("Currency") == "CZK")
    
    print("\n--- Data Summary ---")
    print(f"Total records in 2013: {df_2013.count()}")
    print(f"Total records with CZK in 2013: {df_czk.count()}")
    
    # Step 4: Calculate total consumption per customer in 2013
    df_customer_total = df_czk.groupBy("CustomerID", "Segment") \
        .agg(_sum("Consumption").alias("total_consumption"))
    
    print("\n--- Total Consumption per Customer (2013, CZK) - Sample ---")
    df_customer_total.show(10)
    
    # Step 5: Find the minimum total_consumption value for each segment
    df_segment_min = df_customer_total.groupBy("Segment") \
        .agg(_min("total_consumption").alias("min_consumption_val"))
    
    print("\n--- Minimum Consumption Value per Segment ---")
    df_segment_min.show()
    
    # Step 6: Find customers who have the minimum consumption in their segment
    df_min_customers = df_customer_total.join(df_segment_min, on="Segment") \
        .filter(col("total_consumption") == col("min_consumption_val"))
    
    print("\n--- Customers with Minimum Consumption per Segment ---")
    df_min_customers.show()
    
    # Step 7: Count how many min-customers per segment
    df_min_count = df_min_customers.groupBy("Segment") \
        .agg(count("CustomerID").alias("min_customer_count"))
    
    print("\n--- Count of Min Consumption Customers per Segment ---")
    df_min_count.show()
    
    # Step 8: Calculate total consumption for each segment (all customers)
    df_segment_total = df_customer_total.groupBy("Segment") \
        .agg(_sum("total_consumption").alias("segment_total_consumption"))
    
    print("\n--- Total Consumption per Segment ---")
    df_segment_total.show()
    
    # Step 9: Calculate average = segment_total / count_of_min_customers
    df_avg = df_segment_total.join(df_min_count, on="Segment") \
        .withColumn("avg_consumption", col("segment_total_consumption") / col("min_customer_count"))
    
    print("\n--- Average Annual Consumption for Min Customers per Segment ---")
    print("Formula: segment_total_consumption / min_customer_count")
    df_avg.show()
    
    # Step 10: Calculate differences between segments
    results = df_avg.collect()
    
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
    output_dir = os.path.join(script_dir, "../Q3_1_output")
    df_avg.write.mode("overwrite").csv(
        output_dir,
        header=True
    )
    print(f"\nOutput saved to {output_dir}")
    
    spark.stop()

if __name__ == "__main__":
    main()
