#!/usr/bin/env python3
"""
Q2_2: Find instructors whose courses about "spark" (created after 2018-01-01)
have the highest rating. Use PySpark SQL.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace

def main():
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("Q2_2_TopSparkInstructor") \
        .getOrCreate()
    
    # Read CSV files
    instructors_df = spark.read.csv(
        "/data/assignment2/Q2_data/instructors.csv",
        header=True,
        inferSchema=False
    )
    
    courses_df = spark.read.csv(
        "/data/assignment2/Q2_data/courses.csv",
        header=True,
        inferSchema=False
    )
    
    # Register as temporary views for SQL
    instructors_df.createOrReplaceTempView("instructors")
    courses_df.createOrReplaceTempView("courses")
    
    query = """
    WITH spark_courses AS (
        SELECT 
            c.id AS course_id,
            c.title AS course_title,
            c.rating AS course_rating,
            c.created AS course_created,
            c.instructors_id,
            i.display_name,
            i.job_title
        FROM courses c
        INNER JOIN instructors i ON c.instructors_id = i.id
        WHERE 
            LOWER(c.title) LIKE '%spark%'
            AND c.created > '2018-01-01T00:00:00'
    )
    SELECT 
        display_name,
        job_title,
        course_rating,
        course_title,
        course_created
    FROM spark_courses
    ORDER BY CAST(course_rating AS DOUBLE) DESC
    LIMIT 1
    """
    
    result = spark.sql(query)
    
    print("\nSQL Query executed.")
    print("\n--- Result: Top-Rated Spark Instructor ---")
    result.show(truncate=False)
    
    # Also show all matching courses for reference
    print("\n--- All matching Spark courses (for reference) ---")
    all_spark_query = """
    SELECT 
        i.display_name,
        i.job_title,
        c.title AS course_title,
        c.rating AS course_rating,
        c.created AS course_created
    FROM courses c
    INNER JOIN instructors i ON c.instructors_id = i.id
    WHERE 
        LOWER(c.title) LIKE '%spark%'
        AND c.created > '2018-01-01T00:00:00'
    ORDER BY CAST(c.rating AS DOUBLE) DESC
    """
    all_results = spark.sql(all_spark_query)
    all_results.show(truncate=False)
    
    # Save the top result
    result.write.mode("overwrite").csv(
        "/data/assignment2/Q2_2_output",
        header=True
    )
    
    print("Output saved to /data/assignment2/Q2_2_output")
    
    spark.stop()

if __name__ == "__main__":
    main()
