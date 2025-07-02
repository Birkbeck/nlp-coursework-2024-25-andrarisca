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

def count_parties(csvRows):
    partyCount = dict()
    
    for row in csvRows:
        party_name = row['party']
        if partyCount.get(party_name, None) == None:
            partyCount.update({party_name: 1})
        elif partyCount.get(party_name, None) != None:
            partyCount[party_name] += 1
            
    return partyCount

most_common_party_name = {
    'Conservative': 25079,
    'Labour': 8038,
    'Scottish National Party': 2303,
    'Liberal Democrat': 803,
}

def delete_uncommon_parties(csvRows):
    updated_rows = []
    for row in csvRows:
        if row['party'] in most_common_party_name.keys():
            updated_rows.append(row)
    return updated_rows

def main():
    
    with open('p2-texts/hansard40000.csv', 'r', encoding="utf-8") as file:
        reader = csv.DictReader(file)
        # rename_labour_co_op(reader)
        # for (idx, row) in enumerate(rename_labour_co_op(reader)):
        #     if idx == 16:
        #         pprint(row)
        # pprint(count_parties(rename_labour_co_op(reader)))
        renamed = rename_labour_co_op(reader)
        print(len(renamed))
        cleaned_up = delete_uncommon_parties(renamed)
        print(len(cleaned_up))
            # if idx == 16:
            #     pprint(row)

main()