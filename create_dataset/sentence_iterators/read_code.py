"""
https://www.kaggle.com/datasets/simiotic/github-code-snippets
"""

import sqlite3

DATASET_NAME = "code"

def iter_code_snippets(database_path: str):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    cursor.execute('SELECT snippet FROM snippets')

    row = cursor.fetchone()
    while row is not None:
        row = cursor.fetchone()[0]
        for command in row.split("\n"):
            yield command

    cursor.close()
    conn.close()