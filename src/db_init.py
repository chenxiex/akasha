from dotenv import load_dotenv
load_dotenv()

import psycopg2
import os
import logging
from pgvector.psycopg2 import register_vector

def get_db_connection() -> psycopg2.extensions.connection:
    """创建并返回数据库连接"""
    db_host = os.getenv("POSTGRES_HOST")
    db_name = os.getenv("POSTGRES_DB")
    db_user = os.getenv("POSTGRES_USER")
    db_password = os.getenv("POSTGRES_PASSWORD")
    
    conn = psycopg2.connect(
        host=db_host,
        database=db_name,
        user=db_user,
        password=db_password
    )
    return conn

def check_table_exists(table_name:str, cursor:psycopg2.extensions.cursor) -> bool:
    # 检查表是否存在
    cursor.execute(f"""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public'
            AND table_name = %s
        );
    """, (table_name,))

    fetch_result = cursor.fetchone()
    table_exists = fetch_result[0] if fetch_result is not None else False
    return table_exists

def initialize_database(recreate:bool=False) -> None:
    """
    初始化数据库表结构

    :param recreate: 是否重新创建表结构，默认为False

    :raises Exception: 如果数据库连接或表创建失败
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('CREATE EXTENSION IF NOT EXISTS vector')
    register_vector(cursor)

    if recreate:
        try:
            # 删除现有表
            cursor.execute("DROP TABLE IF EXISTS embeddings;")
            cursor.execute("DROP TABLE IF EXISTS settings;")
            conn.commit()
        except Exception as e:
            logging.error(f"删除表时出错: {e}")
            conn.rollback()
            cursor.close()
            conn.close()
            return

    try:
        # 创建需要的表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                file_name VARCHAR(255) PRIMARY KEY,
                embedding vector(1024) NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key VARCHAR(255) PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        conn.commit()

    except Exception as e:
        logging.error(f"初始化数据库时出错: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    initialize_database(recreate=True)