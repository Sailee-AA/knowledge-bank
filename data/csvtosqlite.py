import sqlite3
import csv

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('data.db')
cursor = conn.cursor()

# Create a table
    # Incident_Id INTEGER,
    # Issue_Type TEXT,
    # RCA TEXT,
    # Steps_to_Resolve TEXT,
    # Tags TEXT,
    # Positive_Counter INTEGER DEFAULT 0,
    # Negative_Counter INTEGER DEFAULT 0
cursor.execute('''
    CREATE TABLE IF NOT EXISTS data (             
    Summary	TEXT,
    Description	TEXT,
    Custom_field_Root_Cause	TEXT,
    Comment	TEXT,
    Issue_Type	TEXT,
    Status	TEXT,
    Priority	TEXT,
    Resolution	TEXT,
    Assignee	TEXT,
    Assignee_Id	TEXT,
    Reporter	TEXT,
    Reporter_Id	TEXT,
    Creator	TEXT,
    Creator_Id	TEXT,
    Watchers	TEXT,
    Custom_field_Approvers	TEXT,
    Custom_field_Approvers_Id	TEXT,
    Custom_field_Epic_Link	TEXT,
    Epic_Link_Summary	TEXT,
    Sprint	TEXT,
    Parent INTEGER,
    Parent_summary TEXT

)
''')

# Open the CSV file and read its contents
with open('incident_data.csv', 'r') as file:
    reader = csv.reader(file)
    header = next(reader)  # Skip the header row
    rows = [row for row in reader]

# Insert the rows into the database
cursor.executemany('INSERT INTO data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ? ,?)', rows)

# Commit the transaction and close the connection
conn.commit()
conn.close()

print("CSV file has been successfully loaded into the SQLite database.")
