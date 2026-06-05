#!/usr/bin/env python3
"""
Q1: Partition data by origin column using Spark RDD API.
- All rows with origin = ATL go to one partition
- All other rows are randomly distributed to 3 other partitions
"""

from pyspark import SparkContext, SparkConf
import random

def main():
    # Initialize Spark context
    conf = SparkConf().setAppName("Q1_PartitionByOrigin")
    sc = SparkContext(conf=conf)
    
    # Read CSV file
    csv_file = "/data/assignment2/Q1_data/departuredelays.csv"
    lines = sc.textFile(csv_file)
    
    # Get header and data
    header = lines.first()
    data = lines.filter(lambda line: line != header)
    
    # Parse CSV line and extract origin (index 3)
    def parse_line(line):
        parts = line.split(",")
        origin = parts[3]
        return (parts, origin)
    
    # Create key-value pairs for partitioning
    # Key: 0 for ATL, random 1-3 for others
    # Value: the full row data
    def create_partition_key(parsed):
        parts, origin = parsed
        if origin == "ATL":
            key = 0
        else:
            key = random.randint(1, 3)
        return (key, parts)
    
    # Parse and create key-value pairs
    parsed_data = data.map(parse_line)
    keyed_data = parsed_data.map(create_partition_key)
    
    # Partition into 4 partitions
    partitioned = keyed_data.partitionBy(4)
    
    # Collect results and verify partitioning
    # Get partition ID for each record
    def get_partition_info(kv):
        # This will be executed on workers, we need to track partitions differently
        return kv
    
    # Use mapPartitionsWithIndex to see which records are in which partition
    partition_contents = partitioned.mapPartitionsWithIndex(
        lambda idx, iterator: [(idx, list(iterator))]
    ).collect()
    
    # Print results
    print(f"Header: {header}")
    print(f"\nTotal partitions: {partitioned.getNumPartitions()}")
    
    for partition_idx, records in partition_contents:
        origins = [r[1][3] for r in records]  # Get origin from each record
        atl_count = origins.count("ATL")
        other_count = len(origins) - atl_count
        print(f"\nPartition {partition_idx}:")
        print(f"  Total records: {len(records)}")
        print(f"  ATL records: {atl_count}")
        print(f"  Other records: {other_count}")
        if len(records) > 0:
            print(f"  Sample origins: {origins[:10]}...")
    
    # Save output
    partitioned.map(lambda x: x[1]).map(lambda parts: ",".join(parts)) \
        .saveAsTextFile("/data/assignment2/Q1_output")
    print("\nOutput saved to /data/assignment2/Q1_output")
    
    sc.stop()

if __name__ == "__main__":
    main()
