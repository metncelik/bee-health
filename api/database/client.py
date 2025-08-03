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
            
database_client = DatabaseClient()