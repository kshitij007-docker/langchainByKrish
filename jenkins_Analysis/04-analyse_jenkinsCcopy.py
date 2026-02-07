import sys
import os
import sqlite3
import re
from datetime import datetime
from langchain_ollama import ChatOllama

# --- CONFIG: Use relative path so it matches the API Server ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "jenkins_ai.db")

print(f"📂 Using Database: {DB_PATH}")

# -------------------------
# Helper: Rule-based classification
# -------------------------
def classify_by_rules(log_text: str):
    text = log_text.lower()
    
    # Defaults
    severity = "major"
    category = "build"
    confident = False

    if "warning" in text or "flaky" in text:
        severity = "minor"
        category = "test"
        confident = True
    elif "timeout" in text or "slow" in text:
        severity = "major"
        category = "performance"
        confident = True
    elif "maven" in text or "dependency" in text:
        severity = "major"
        category = "build"
        confident = True
    elif "database" in text or "connection refused" in text:
        severity = "blocker"
        category = "infra"
        confident = True
    elif "disk" in text or "no space" in text:
        severity = "blocker"
        category = "infra"
        confident = True

    return severity, category, confident

# -------------------------
# Helper: Parse AI Output
# -------------------------
def parse_ai_response(ai_text):
    # This regex handles both "Severity: minor" and "Severity:\nminor"
    # It looks for the label, consumes whitespace (including newlines), and captures the next word.
    sev_match = re.search(r"Severity:\s*(\w+)", ai_text, re.IGNORECASE)
    cat_match = re.search(r"Category:\s*(\w+)", ai_text, re.IGNORECASE)

    parsed_sev = sev_match.group(1).lower() if sev_match else None
    parsed_cat = cat_match.group(1).lower() if cat_match else None

    return parsed_sev, parsed_cat

# -------------------------
# Helper: Save to SQLite
# -------------------------
def save_failure_to_sqlite(job_name, build_number, node_name,
                           severity, category, summary):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO build_failures
        (job_name, build_number, node_name,
         severity, category, summary, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        job_name,
        build_number,
        node_name,
        severity,
        category,
        summary,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()

# -------------------------
# Main Execution
# -------------------------
if len(sys.argv) < 2:
    print("Usage: python 04-Analyse_JenkinsC.py <log_file>")
    sys.exit(1)

log_file = sys.argv[1]

if not os.path.exists(log_file):
    print(f"❌ ERROR: Log file not found: {log_file}")
    sys.exit(1)

with open(log_file, "r", errors="ignore") as f:
    log_text = f.read()[-6000:]  # Keep last 6000 chars

# Metadata
job_name = os.environ.get("JOB_NAME", "demo-job-1")
build_number = int(os.environ.get("BUILD_NUMBER", "42"))
node_name = os.environ.get("NODE_NAME", "built-in")

# Step 1: Try Rules (Fast Path)
severity, category, confident = classify_by_rules(log_text)

# Step 2: Ask AI (Using your strict prompt)
llm = ChatOllama(model="gemma3:1b", temperature=0)

prompt = f"""
You are a senior Software Reliability Engineer.

TASK:
Analyze the Jenkins build log provided below.

STRICT RULES:
- Base your analysis ONLY on information explicitly present in the log.
- Do NOT assume the technology (Git, database, API, etc.) unless it is clearly mentioned in the log.
- Do NOT guess the system, service, or component.
- Do NOT introduce entities, tools, or technologies not present in the log.
- If the log is ambiguous, explicitly state: "Insufficient information in log".
- If a component is not explicitly named, describe it generically
  (e.g., "a remote service", "a network endpoint", "a build dependency").

OUTPUT FORMAT (DO NOT CHANGE HEADINGS):

BUILD FAILURE ANALYSIS
----------------------
Failure Reason:
<Describe only what is explicitly visible in the log>

Root Cause (Evidence-Based):
- State only what can be directly concluded from the log.
- If the log does not identify the component, explicitly say so.
- Do NOT name systems or tools unless explicitly mentioned.

Suggested Fix:
<Provide a generic next step without naming specific systems or technologies>

Next Debugging Steps:
- Step 1
- Step 2
- Step 3

Log:
{log_text}
"""

print("⏳ Asking AI...")
ai_summary = llm.invoke(prompt).content

# Step 3: Parse AI response
# Even if rules were confident, we check if AI found something different/better
# or we can stick to rules if they are preferred. 
# Here, we update from AI if rules were NOT confident, or if we want AI to override.
# Current logic: If rules were not confident, trust AI.
if not confident:
    ai_sev, ai_cat = parse_ai_response(ai_summary)
    if ai_sev and ai_sev != "unknown": 
        severity = ai_sev
    if ai_cat and ai_cat != "unknown": 
        category = ai_cat

# Step 4: Save
save_failure_to_sqlite(
    job_name=job_name,
    build_number=build_number,
    node_name=node_name,
    severity=severity,
    category=category,
    summary=ai_summary
)

print("\n🔍 AI BUILD FAILURE ANALYSIS\n")
print(f"Severity: {severity.upper()}")
print(f"Category: {category.upper()}")
print(ai_summary)
print(f"✅ Saved to db: {job_name}")
