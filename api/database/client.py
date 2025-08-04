import sqlite3
from pathlib import Path
from contextlib import contextmanager

SQLITE_DB_PATH = Path(__file__).parent / ".sqlite" / "database.db"

class DatabaseClient:
    def __init__(self):
        self.db_path = SQLITE_DB_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
       
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.init_database()
        
    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  
        conn.execute("PRAGMA foreign_keys = ON") 
        try:
            yield conn
        finally:
            conn.close()
            
    def init_database(self):
        schema_path = Path(__file__).parent / "schemas.sql"
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        
        with self.get_connection() as conn:
            conn.executescript(schema_sql)
            conn.commit()
    
    # Image operations
    def create_image(self, url: str) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO images (url) VALUES (?)",
                (url,)
            )
            conn.commit()
            return cursor.lastrowid
    
    def get_image(self, image_id: int) -> dict:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM images WHERE id = ?", (image_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    # Class operations
    def create_class(self, name: str, description: str) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO classes (name, description) VALUES (?, ?)",
                (name, description)
            )
            conn.commit()
            return cursor.lastrowid
    
    def get_class_by_name(self, name: str) -> dict:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM classes WHERE name = ?", (name,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_all_classes(self) -> list:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM classes ORDER BY name")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    # Prediction operations
    def create_prediction(self, image_id: int, confidence: float) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO predictions (image_id, confidence) VALUES (?, ?)",
                (image_id, confidence)
            )
            conn.commit()
            return cursor.lastrowid
    
    def get_prediction(self, prediction_id: int) -> dict:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def link_prediction_to_class(self, prediction_id: int, class_id: int) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO predictions_classes (prediction_id, class_id) VALUES (?, ?)",
                (prediction_id, class_id)
            )
            conn.commit()
            return cursor.lastrowid
    
    def get_prediction_class(self, prediction_id: int) -> dict:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT c.* FROM classes c
                JOIN predictions_classes pc ON c.id = pc.class_id
                WHERE pc.prediction_id = ?
            """, (prediction_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    # Chat operations
    def create_chat(self, prediction_id: int) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO chats (prediction_id) VALUES (?)",
                (prediction_id,)
            )
            conn.commit()
            return cursor.lastrowid
    
    def get_chat(self, chat_id: int) -> dict:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chats WHERE id = ?", (chat_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_chats_by_prediction(self, prediction_id: int) -> list:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chats WHERE prediction_id = ?", (prediction_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    # Message operations
    def create_message(self, chat_id: int, role: str, content: str) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
                (chat_id, role, content)
            )
            conn.commit()
            return cursor.lastrowid
    
    def get_messages_by_chat(self, chat_id: int) -> list:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM messages WHERE chat_id = ? ORDER BY created_at",
                (chat_id,)
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def get_message(self, message_id: int) -> dict:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM messages WHERE id = ?", (message_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        
    def get_predictions(self) -> list:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM predictions ORDER BY created_at DESC")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        
    def get_chat_by_prediction(self, prediction_id: int) -> dict:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chats WHERE prediction_id = ?", (prediction_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
            
database_client = DatabaseClient()