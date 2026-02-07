from flask import Flask, jsonify
import sqlite3
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "jenkins_ai.db")


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@app.route("/api/failures")
def get_failures():
    conn = get_db_connection()
    rows = conn.execute("""
        SELECT job_name, build_number, node_name,
               category, severity, summary, created_at
        FROM build_failures
        ORDER BY created_at DESC
        LIMIT 20
    """).fetchall()
    conn.close()

    return jsonify([dict(r) for r in rows])


@app.route("/api/metrics")
def get_metrics():
    conn = get_db_connection()

    total = conn.execute(
        "SELECT COUNT(*) FROM build_failures"
    ).fetchone()[0]

    by_category = conn.execute("""
        SELECT category, COUNT(*) as count
        FROM build_failures
        GROUP BY category
    """).fetchall()

    conn.close()

    categories = {}
    for row in by_category:
        categories[row["category"]] = round(
            (row["count"] / total) * 100, 1
        ) if total > 0 else 0

    return jsonify({
        "total_failures": total,
        "top_failure_categories": categories
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
