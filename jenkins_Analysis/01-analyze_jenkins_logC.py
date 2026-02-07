import sys
import os
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
import sqlite3
from datetime import datetime

load_dotenv()

# -------------------------
# Input validation
# -------------------------
if len(sys.argv) < 2:
    print("Usage: python analyze_jenkins_log.py <jenkins_log_file>")
    sys.exit(1)

log_file_path = sys.argv[1]

if not os.path.exists(log_file_path):
    print(f"ERROR: Log file not found: {log_file_path}")
    sys.exit(1)

# -------------------------
# Read Jenkins log
# -------------------------
with open(log_file_path, "r", errors="ignore") as f:
    log_text = f.read()

# Keep it bounded
log_text = log_text[-6000:]

# -------------------------
# LLM setup (DEMO: Ollama)
# -------------------------
llm = ChatOllama(
    model="gemma3:1b",
    temperature=0
)

# -------------------------
# Jenkins-specific prompt
# -------------------------
prompt = f"""
You are a senior Software Reliabilty engineer.

TASK:
Analyze the Jenkins build log below.

STRICT RULES:
- Base your analysis ONLY on information explicitly present in the log.
- Do NOT assume the technology (Git, database, API, etc.) unless clearly mentioned.
- If the log is ambiguous, say "Insufficient information in log".
- Do NOT guess the system or component.
- Do NOT introduce new entities not present in the log.
- If a component is not explicitly named in the log, you MUST describe it generically (e.g., "a remote service", "a network endpoint").

OUTPUT FORMAT (DO NOT CHANGE HEADINGS):

BUILD FAILURE ANALYSIS
----------------------
Failure Reason:
<what is explicitly visible in the log>

Root Cause (Evidence-Based):
- State only what can be concluded directly from the log.
- If the log does not identify the component, explicitly say so.
- Do NOT name databases, repositories, services, or tools unless explicitly mentioned.

Suggested Fix:
<generic next step without naming specific systems>

Next Debugging Steps:
- Step 1
- Step 2
- Step 3

LOG:
{log_text}

"""

# -------------------------
# Run analysis
# -------------------------
try:
    response = llm.invoke(prompt)
    print("\n🔍 AI BUILD FAILURE ANALYSIS\n")
    print(response.content)
except Exception as e:
    print("ERROR: Failed to analyze Jenkins log")
    print(str(e))



job_name = os.getenv("JOB_NAME", "unknown-job")
build_number = os.getenv("BUILD_NUMBER", "0")
node_name = os.getenv("NODE_NAME", "unknown-node")

print(response)

save_failure_to_sqlite(
    job_name=job_name,
    build_number=int(build_number),
    node_name=node_name,
    category=category,
    severity=severity,
    summary=response
)

print("Failure summary saved to SQLite")

def save_failure_to_sqlite(
    job_name,
    build_number,
    node_name,
    category,
    severity,
    summary
):
    
    conn = sqlite3.connect("jenkins_ai.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO build_failures
        (job_name, build_number, node_name, category, severity, summary)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        job_name,
        build_number,
        node_name,
        category,
        severity,
        summary
    ))

    conn.commit()
    conn.close()
