#!/bin/bash

INPUT="SRR12326775_1_Light_Bulk.csv"
WORKDIR="q2_3_work"
CHUNKDIR="$WORKDIR/chunks"
CLEANCHUNKDIR="$WORKDIR/clean_chunks"
FINAL_OUTPUT="SRR12326775_1_Light_Bulk_final.csv"

mkdir -p "$WORKDIR" "$CHUNKDIR" "$CLEANCHUNKDIR"

echo "Step 1: extract header (line 2)"
sed -n '2p' "$INPUT" > "$WORKDIR/header.txt"

echo "Step 2: extract data rows (from line 3 onward)"
tail -n +3 "$INPUT" > "$WORKDIR/data_only.csv"

echo "Step 3: split data into 8 chunks"
split -n l/8 -d --additional-suffix=.csv "$WORKDIR/data_only.csv" "$CHUNKDIR/chunk_"

echo "Step 4: add header to each chunk"
for f in "$CHUNKDIR"/chunk_*.csv
do
    tmpfile="${f}.tmp"
    cat "$WORKDIR/header.txt" "$f" > "$tmpfile"
    mv "$tmpfile" "$f"
done

echo "Step 5: process each chunk in parallel"
for f in "$CHUNKDIR"/chunk_*.csv
do
    base=$(basename "$f" .csv)
    py process_chunk.py "$f" "$CLEANCHUNKDIR/${base}_clean.csv" &
done
wait

echo "Step 6: merge all cleaned chunk outputs"
first_file=$(ls "$CLEANCHUNKDIR"/*_clean.csv | head -n 1)

cat "$first_file" > "$FINAL_OUTPUT"

for f in "$CLEANCHUNKDIR"/*_clean.csv
do
    if [ "$f" != "$first_file" ]; then
        tail -n +2 "$f" >> "$FINAL_OUTPUT"
    fi
done

echo "Done. Final output saved to $FINAL_OUTPUT"