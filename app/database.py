import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Optional, Dict, Any, List
from app.config import config
import contextlib
from datetime import datetime


class Database:
    def __init__(self):
        self.connection_string = config.DATABASE_URL

    @contextlib.contextmanager
    def get_connection(self):
        """Context manager para abrir/fechar conexões"""
        conn = psycopg2.connect(
            self.connection_string,
            cursor_factory=RealDictCursor
        )
        try:
            yield conn
        finally:
            conn.close()

    def init_db(self):
        """Inicializa tabelas (se necessário)."""
        
        print("INFO: Database tables already exist, skipping table creation")

    # -----------------------------
    # CRUD for Jobs
    # -----------------------------
    def create_job(self, repo_name: str) -> int:
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO public.jobs (repo_name, status, created_at)
                    VALUES (%s, %s, %s)
                    RETURNING id
                    """,
                    (repo_name, "pending", datetime.utcnow())
                )
                job_id = cur.fetchone()["id"]
                conn.commit()
                return job_id

    def update_job(self, job_id: int, status: str,
                   repo_url: Optional[str] = None,
                   error: Optional[str] = None) -> None:
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE public.jobs
                    SET status = %s,
                        github_repo_url = %s,
                        error = %s,
                        updated_at = %s
                    WHERE id = %s
                    """,
                    (status, repo_url, error, datetime.utcnow(), job_id)
                )
                conn.commit()

    def get_job(self, job_id: int) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, repo_name, status, github_repo_url, error,
                           created_at, updated_at
                    FROM public.jobs
                    WHERE id = %s
                    """,
                    (job_id,)
                )
                return cur.fetchone()

    def list_jobs(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, repo_name, status, github_repo_url, error,
                           created_at, updated_at
                    FROM public.jobs
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,)
                )
                return cur.fetchall()

    # -----------------------------
    # CRUD for files generated
    # -----------------------------
    def insert_generated_file(self, job_id: int, path: str) -> None:
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO public.generated_files (job_id, path, created_at)
                    VALUES (%s, %s, %s)
                    """,
                    (job_id, path, datetime.utcnow())
                )
                conn.commit()

    def list_generated_files(self, job_id: int) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, job_id, path, created_at
                    FROM public.generated_files
                    WHERE job_id = %s
                    ORDER BY created_at ASC
                    """,
                    (job_id,)
                )
                return cur.fetchall()



db = Database()
