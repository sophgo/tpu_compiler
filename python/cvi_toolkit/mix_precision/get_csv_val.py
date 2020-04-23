#!/usr/bin/env python
import csv
import sys

def get_csv_val(csv_file, FIELDS, row_idx):
    r = ""
    with open(csv_file, "r") as f:
        mycsv = csv.DictReader(f)
        for r_idx, row in enumerate(mycsv):
            if r_idx == row_idx:
                for i, col in enumerate(row):
                    if col.strip() in FIELDS:
                        try:
                            r = (row[col].strip())
                            break
                        except KeyError:
                            pass
    return r

def get_rows_by_column(csv_file, included_cols):
    with open(csv_file, 'r') as infile:
        # read the file as a dictionary for each row ({header : value})
        reader = csv.DictReader(infile)
        data = []
        for row in reader:
            for header, value in row.items():
                if header in included_cols:
                    data.append(value)

    return data

if __name__ == '__main__':
    if len(sys.argv) < 3:
      print("Usage: %s csv_file column_name row_idx " % sys.argv[0])
      print("  output: value by column_name and row_idx")
      exit(-1)

    csv_file = sys.argv[1]
    column_name = (sys.argv[2])
    row_idx = int(sys.argv[3])

    FIELDS = []
    FIELDS.append(column_name)

    v = get_csv_val(csv_file, FIELDS, row_idx);
    print(v)

    v = get_rows_by_column(csv_file, [column_name])
    print(v)

