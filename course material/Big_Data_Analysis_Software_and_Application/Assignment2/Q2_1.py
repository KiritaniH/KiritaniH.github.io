#!/usr/bin/env python3

from pyspark.sql import SparkSession

def main():
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("Q2_1_InnerJoin") \
        .getOrCreate()
    
    # Read CSV files with headers (all as strings to avoid type inference issues)
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
    
    # Perform inner join on instructors_id
    # instructors.id == courses.instructors_id
    joined_df = courses_df.join(
        instructors_df,
        courses_df.instructors_id == instructors_df.id,
        how="inner"
    )
    
    print(f"\nInstructors schema: {instructors_df.columns}")
    print(f"Courses schema: {courses_df.columns}")
    
    print(f"\nTotal joined records: {joined_df.count()}")
    
    print("\n--- Sample joined records (first 5) ---")
    joined_df.select(
        courses_df.instructors_id,
        courses_df.title.alias("course_title"),
        instructors_df.name,
        instructors_df.display_name,
        courses_df.rating,
        courses_df.num_reviews
    ).show(5, truncate=False)
    
    # Save output
    joined_df.write.mode("overwrite").csv(
        "/data/assignment2/Q2_1_output",
        header=True
    )
    
    print("Output saved to /data/assignment2/Q2_1_output")
    
    spark.stop()

if __name__ == "__main__":
    main()
