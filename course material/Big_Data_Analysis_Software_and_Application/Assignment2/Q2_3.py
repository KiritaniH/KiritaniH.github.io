#!/usr/bin/env python3

from pyspark.sql import SparkSession

def main():
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("Q2_3_InterviewCourses") \
        .getOrCreate()
    
    # Read CSV files
    courses_df = spark.read.csv(
        "/data/assignment2/Q2_data/courses.csv",
        header=True,
        inferSchema=False
    )
    
    # Register as temporary view for SQL
    courses_df.createOrReplaceTempView("courses")
    
    query = """
    SELECT 
        id AS course_id,
        title AS course_title,
        ROUND(CAST(rating AS DOUBLE), 1) AS course_rating,
        created AS course_created,
        num_reviews,
        duration,
        instructors_id
    FROM courses
    WHERE 
        LOWER(title) LIKE '%interview%'
    ORDER BY 
        course_rating DESC,
        course_created DESC
    """
    
    result = spark.sql(query)
    
    print("\nSQL Query executed.")
    print(f"\nTotal matching courses: {result.count()}")
    
    print("\n--- All Interview Courses (sorted) ---")
    result.show(truncate=False)
    
    # Save output
    result.write.mode("overwrite").csv(
        "/data/assignment2/Q2_3_output",
        header=True
    )

    print("Output saved to /data/assignment2/Q2_3_output")
    
    spark.stop()

if __name__ == "__main__":
    main()
