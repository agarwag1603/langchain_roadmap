import sqlite3
from datetime import datetime
import os
import sqlite3

# Dynamically get the path of this Python file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "SQLite", "chat_history.db")

def init_db():
    os.makedirs(os.path.join(BASE_DIR, "SQLite"), exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        role TEXT,
        content TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

def add_message(session_id, role, content):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO conversations (session_id, role, content) VALUES (?, ?, ?)",
        (session_id, role, content)
    )
    conn.commit()
    conn.close()

def get_conversation(session_id):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content FROM conversations WHERE session_id=? ORDER BY id",
        (session_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1]} for r in rows]

def list_sessions():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT session_id FROM conversations ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows]