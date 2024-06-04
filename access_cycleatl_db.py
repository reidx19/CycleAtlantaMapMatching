import sqlite3
from pathlib import Path

def create_database(sql_file, db_file):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Read the SQL file
    with open(sql_file, 'r') as sql_file:
        sql_commands = sql_file.read()

    # Execute the SQL commands to create tables and insert data
    cursor.executescript(sql_commands)

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

def query_database(db_file):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Query the database and print the results
    cursor.execute("SELECT * FROM your_table_name")  # Replace your_table_name with the actual table name
    rows = cursor.fetchall()
    for row in rows:
        print(row)

    # Close the connection
    conn.close()

if __name__ == "__main__":
    # Provide the path to your .sql file and the desired name for the SQLite database
    sql_file = Path.home() / "Documents/ridership_data/fromchris/Cycle Atlanta Dump/cycleatl_production.DreamHostBackup.20170412.sql"
    db_file = Path.home() / "Downloads/test_db.db"

    # Create the database and populate it with data from the SQL file
    create_database(sql_file, db_file)

    # Query the database and display its contents
    query_database(db_file)
