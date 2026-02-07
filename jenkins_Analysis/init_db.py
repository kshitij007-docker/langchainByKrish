import sqlite3

# This line CREATES the .db file if it doesn't exist
conn = sqlite3.connect("jenkins_ai.db")

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS build_failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_name TEXT,
    build_number INTEGER,
    node_name TEXT,
    category TEXT,
    severity TEXT,
    summary TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()

print("SQLite database created successfully")
