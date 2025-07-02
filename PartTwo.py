import csv
from pprint import pprint


def rename_labour_co_op(csv_reader):
    updated_rows = []
    for (_, row) in enumerate(csv_reader):
        if row['party'] == "Labour (Co-op)":
            row.update({'party': "Labour"})
            updated_rows.append(row)
        else:
            updated_rows.append(row)
    return updated_rows

def main():
    
    with open('p2-texts/hansard40000.csv', 'r', encoding="utf-8") as file:
        reader = csv.DictReader(file)
        # rename_labour_co_op(reader)
        for (idx, row) in enumerate(rename_labour_co_op(reader)):
            if idx == 16:
                pprint(row)

main()