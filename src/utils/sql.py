from __future__ import annotations

import os
from typing import Any, List, Tuple

import pymysql


def get_mysql_conn():
    return pymysql.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
        database=os.getenv("MYSQL_DB", "test"),
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )


def run_query(sql: str, params: Tuple[Any, ...] | None = None) -> List[dict]:
    with get_mysql_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            if cur.description:
                return list(cur.fetchall())
            return []
