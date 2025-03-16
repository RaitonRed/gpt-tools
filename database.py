import sqlite3

# Database path
DATABASE_PATH = 'database.db'

conn = sqlite3.connect(DATABASE_PATH)

def create_db():
    """
    Create or reset the database tables.
    """
    
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS inputs (
            id INTEGER PRIMARY KEY,
            input_text TEXT,
            selected_model TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            username TEXT NOT NULL,
            user_message TEXT NOT NULL,
            ai_response TEXT NOT NULL
        )
    """)
    conn.commit()

def insert_chat(chat_id, username, user_message, ai_response):
    """
    Insert a chat into the chats table.
    """
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO chats (chat_id, username, user_message, ai_response)
            VALUES (?, ?, ?, ?)
        """, (str(chat_id), str(username), str(user_message), str(ai_response)))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error inserting chat: {e}")

def insert_into_db(input_text, selected_model):
    """
    Insert data into the inputs table.
    """

    try:
        c = conn.cursor()
        c.execute("""
            INSERT INTO inputs (input_text, selected_model)
            VALUES (?, ?)
        """, (str(input_text), str(selected_model)))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error inserting into inputs: {e}")

def clear_database():
    """
    Clear all data from the inputs table.
    """

    try:
        c = conn.cursor()
        c.execute("DELETE FROM inputs")
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error clearing database: {e}")

def fetch_all_inputs():
    """
    Fetch all inputs from the inputs table.
    """

    try:
        c = conn.cursor()
        c.execute("SELECT input_text, selected_model FROM inputs")
        results = c.fetchall()
        return results
    except sqlite3.Error as e:
        print(f"Error fetching inputs from database: {e}")
        return []

def fetch_chats_by_id(chat_id):
    """
    Fetch messages and responses associated with a chat_id.
    """

    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT user_message, ai_response FROM chats
            WHERE chat_id = ?
        """, (str(chat_id),))
        rows = cursor.fetchall()
        return rows
    except sqlite3.Error as e:
        print(f"Error fetching chats by ID: {e}")
        return []

def fetch_ids_by_user(username):
    """
    Fetch chat IDs for a specific user.
    """

    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT chat_id FROM chats
            WHERE username = ?
        """, (str(username),))
        rows = cursor.fetchall()
        return rows
    except sqlite3.Error as e:
        print(f"Error fetching chat IDs by username: {e}")
        return []

def clear_chats_by_username(username):
    """
    Delete chats associated with a specific user.
    """

    try:
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM chats
            WHERE username = ?
        """, (str(username),))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error clearing chats by username: {e}")