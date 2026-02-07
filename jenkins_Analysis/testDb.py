import sqlite3

conn = sqlite3.connect("jenkins_ai.db")
cursor = conn.cursor()

cursor.execute("""
INSERT INTO build_failures
(job_name, build_number, node_name, category, severity, summary)
VALUES (?, ?, ?, ?, ?, ?)
""", (
    "TestJob",
    1,
    "local-node",
    "infra",
    "blocker",
    "Test failure summary"
))

conn.commit()
conn.close()

print("Test row inserted")
