from pathlib import Path
import sqlite3


# ============================================================
# DATABASE PATH
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parent.parent

DATABASE_DIR = BACKEND_DIR / "database"

DATABASE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DATABASE_PATH = DATABASE_DIR / "medical_predictions.db"


# ============================================================
# CREATE DATABASE TABLE
# ============================================================

def create_database():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            filename TEXT NOT NULL,

            prediction TEXT NOT NULL,

            confidence REAL NOT NULL,

            medical_report TEXT,

            heatmap_path TEXT,

            gradcam_path TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        )
        """
    )

    connection.commit()

    connection.close()


# ============================================================
# SAVE PREDICTION
# ============================================================

def save_prediction(

    filename,

    prediction,

    confidence,

    medical_report,

    heatmap_path=None,

    gradcam_path=None

):

    connection = sqlite3.connect(

        DATABASE_PATH

    )

    cursor = connection.cursor()


    cursor.execute(

        """
        INSERT INTO predictions
        (
            filename,
            prediction,
            confidence,
            medical_report,
            heatmap_path,
            gradcam_path
        )

        VALUES (?, ?, ?, ?, ?, ?)
        """,

        (

            filename,

            prediction,

            confidence,

            medical_report,

            heatmap_path,

            gradcam_path

        )

    )


    prediction_id = cursor.lastrowid


    connection.commit()

    connection.close()


    return prediction_id


# ============================================================
# GET PREDICTION HISTORY
# ============================================================

def get_prediction_history():

    connection = sqlite3.connect(

        DATABASE_PATH

    )

    connection.row_factory = (

        sqlite3.Row

    )

    cursor = connection.cursor()


    cursor.execute(

        """
        SELECT
            id,
            filename,
            prediction,
            confidence,
            medical_report,
            heatmap_path,
            gradcam_path,
            created_at

        FROM predictions

        ORDER BY created_at DESC
        """
    )


    records = cursor.fetchall()


    connection.close()


    return [

        dict(record)

        for record in records

    ]


# ============================================================
# GET SINGLE PREDICTION
# ============================================================

def get_prediction_by_id(

    prediction_id

):

    connection = sqlite3.connect(

        DATABASE_PATH

    )

    connection.row_factory = (

        sqlite3.Row

    )

    cursor = connection.cursor()


    cursor.execute(

        """
        SELECT
            id,
            filename,
            prediction,
            confidence,
            medical_report,
            heatmap_path,
            gradcam_path,
            created_at

        FROM predictions

        WHERE id = ?

        """,

        (

            prediction_id,

        )

    )


    record = cursor.fetchone()


    connection.close()


    if record is None:

        return None


    return dict(record)


# ============================================================
# INITIALIZE DATABASE
# ============================================================

create_database()