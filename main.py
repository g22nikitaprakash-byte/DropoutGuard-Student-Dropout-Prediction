from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
from pathlib import Path
import sqlite3
from datetime import datetime
import shap
import pandas as pd
import numpy as np


# =========================================================
# APP
# =========================================================

app = FastAPI(title="DropoutGuard API")


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR.parent / "ml" / "prediction_model.pkl"

DB_PATH = BASE_DIR / "students.db"


# =========================================================
# LOAD ML MODEL
# =========================================================

model = joblib.load(MODEL_PATH)


# =========================================================
# SQLITE DATABASE
# =========================================================

def init_db():

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            student_name TEXT NOT NULL,

            student_id TEXT UNIQUE NOT NULL,

            department TEXT,

            age INTEGER,

            admission_grade REAL,

            attendance REAL,

            semester1_grade REAL,

            semester2_grade REAL,

            approved1 INTEGER,

            approved2 INTEGER,

            scholarship INTEGER,

            debtor INTEGER,

            tuition INTEGER,

            academic_average REAL,

            dropout_probability REAL,

            risk_level TEXT,

            prediction TEXT,

            created_at TEXT

        )
    """)
        # =====================================================
    # INTERVENTION TABLE
    # =====================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interventions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            student_id TEXT NOT NULL,

            student_name TEXT NOT NULL,

            intervention_type TEXT NOT NULL,

            description TEXT,

            status TEXT DEFAULT 'OPEN',

            created_at TEXT,

            updated_at TEXT

        )
    """)

    conn.commit()

    conn.close()


# Create database/table when server starts
init_db()


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# =========================================================
# PREDICTION INPUT
# =========================================================

class StudentInput(BaseModel):

    age: int

    admission_grade: float

    attendance: float

    semester1_grade: float

    semester2_grade: float

    approved1: int

    approved2: int

    scholarship: int

    debtor: int

    tuition: int


# =========================================================
# STUDENT DATABASE RECORD
# =========================================================

class StudentRecord(BaseModel):

    student_name: str

    student_id: str

    department: str

    age: int = 0

    admission_grade: float = 0

    attendance: float = 0

    semester1_grade: float = 0

    semester2_grade: float = 0

    approved1: int = 0

    approved2: int = 0

    scholarship: int = 0

    debtor: int = 0

    tuition: int = 0

    academic_average: float = 0

    dropout_probability: float = 0

    risk_level: str = "Low Risk"

    prediction: str = "Unknown"
# =========================================================
# INTERVENTION RECORD
# =========================================================

class InterventionRecord(BaseModel):

    student_id: str

    student_name: str

    intervention_type: str

    description: str = ""

    status: str = "OPEN"

# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {
        "project": "DropoutGuard",
        "status": "Running",
        "version": "1.0"
    }


# =========================================================
# ML PREDICTION
# =========================================================

@app.post("/predict")
def predict(student: StudentInput):

    input_data = [[

        student.age,

        student.admission_grade,

        student.approved1,

        student.semester1_grade,

        student.approved2,

        student.semester2_grade,

        student.scholarship,

        student.debtor,

        student.tuition

    ]]


    # ML prediction

    prediction = model.predict(input_data)[0]


    # Probability

    probabilities = model.predict_proba(input_data)[0]


    # Classes

    classes = model.classes_


    # Dropout probability

    dropout_probability = 0


    for class_name, probability in zip(
        classes,
        probabilities
    ):

        if class_name == "Dropout":

            dropout_probability = probability * 100


    # Risk level

    if dropout_probability >= 70:

        risk = "High Risk"

    elif dropout_probability >= 40:

        risk = "Medium Risk"

    else:

        risk = "Low Risk"


    # Academic average

    academic_average = (

        student.semester1_grade +
        student.semester2_grade

    ) / 2


    return {

        "prediction": prediction,

        "dropout_probability": round(
            dropout_probability,
            2
        ),

        "risk_level": risk,

        "academic_average": round(
            academic_average,
            2
        )

    }

# =========================================================
# EXPLAINABLE AI - SHAP
# =========================================================

@app.post("/explain")
def explain_prediction(student: StudentInput):

    try:

        # Same feature order used by prediction model
        feature_names = [

            "Age at enrollment",

            "Admission grade",

            "Curricular units 1st sem (approved)",

            "Curricular units 1st sem (grade)",

            "Curricular units 2nd sem (approved)",

            "Curricular units 2nd sem (grade)",

            "Scholarship holder",

            "Debtor",

            "Tuition fees up to date"

        ]

        # -------------------------------------------------
        # Create model input
        # -------------------------------------------------

        input_values = np.array([[
            student.age,
            student.admission_grade,
            student.approved1,
            student.semester1_grade,
            student.approved2,
            student.semester2_grade,
            student.scholarship,
            student.debtor,
            student.tuition
        ]], dtype=float)

        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

        prediction = model.predict(input_values)[0]

        probabilities = model.predict_proba(input_values)[0]

        classes = list(model.classes_)

        if "Dropout" not in classes:
            raise Exception(
                "Dropout class was not found in model classes"
            )

        dropout_index = classes.index("Dropout")

        dropout_probability = (
            float(probabilities[dropout_index]) * 100
        )

        # -------------------------------------------------
        # Risk level
        # -------------------------------------------------

        if dropout_probability >= 70:
            risk_level = "High Risk"

        elif dropout_probability >= 40:
            risk_level = "Medium Risk"

        else:
            risk_level = "Low Risk"

        # -------------------------------------------------
        # SHAP
        # -------------------------------------------------

        explainer = shap.TreeExplainer(model)

        shap_result = explainer(input_values)

        shap_values = np.asarray(
            shap_result.values
        )

        # -------------------------------------------------
        # Handle SHAP output
        # -------------------------------------------------

        if shap_values.ndim == 3:

            # (samples, features, classes)

            feature_values = shap_values[
                0,
                :,
                dropout_index
            ]

        elif shap_values.ndim == 2:

            # (samples, features)

            feature_values = shap_values[0]

        else:

            feature_values = shap_values.flatten()

        # -------------------------------------------------
        # Create explanations
        # -------------------------------------------------

        explanations = []

        for feature, shap_value in zip(
            feature_names,
            feature_values
        ):

            value = float(shap_value)

            if value > 0:
                impact = "Increases dropout risk"

            elif value < 0:
                impact = "Decreases dropout risk"

            else:
                impact = "No significant impact"

            explanations.append({

                "feature": feature,

                "shap_value": round(
                    value,
                    4
                ),

                "impact": impact

            })

        # -------------------------------------------------
        # Sort by importance
        # -------------------------------------------------

        explanations.sort(

            key=lambda x:
                abs(x["shap_value"]),

            reverse=True

        )

        # -------------------------------------------------
        # Return response
        # -------------------------------------------------

        return {

            "prediction": str(
                prediction
            ),

            "dropout_probability": round(
                dropout_probability,
                2
            ),

            "risk_level": risk_level,

            "explanations": explanations,

            "top_features": explanations[:5]

        }

    except Exception as e:

        print("===================================")
        print("SHAP ERROR:", str(e))
        print("===================================")

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )
    # =========================================================
# SAVE STUDENT
# =========================================================

@app.post("/students")
def add_student(student: StudentRecord):

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:

        cursor.execute("""
            INSERT INTO students (
                student_name,
                student_id,
                department,
                age,
                admission_grade,
                attendance,
                semester1_grade,
                semester2_grade,
                approved1,
                approved2,
                scholarship,
                debtor,
                tuition,
                academic_average,
                dropout_probability,
                risk_level,
                prediction,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (

            student.student_name,
            student.student_id,
            student.department,
            student.age,
            student.admission_grade,
            student.attendance,
            student.semester1_grade,
            student.semester2_grade,
            student.approved1,
            student.approved2,
            student.scholarship,
            student.debtor,
            student.tuition,
            student.academic_average,
            student.dropout_probability,
            student.risk_level,
            student.prediction,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        ))

        conn.commit()

        return {
            "message": "Student saved successfully",
            "student_id": student.student_id
        }

    except sqlite3.IntegrityError:

        raise HTTPException(
            status_code=409,
            detail="Student ID already exists"
        )

    finally:

        conn.close()


# =========================================================
# GET ALL STUDENTS
# =========================================================

@app.get("/students")
def get_students():

    conn = sqlite3.connect(DB_PATH)

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM students
        ORDER BY id DESC
    """)

    students = [
        dict(row)
        for row in cursor.fetchall()
    ]

    conn.close()

    return students


# =========================================================
# CREATE INTERVENTION
# =========================================================

@app.post("/interventions")
def create_intervention(
    intervention: InterventionRecord
):

    allowed_status = [
        "OPEN",
        "IN PROGRESS",
        "RESOLVED"
    ]

    if intervention.status not in allowed_status:

        raise HTTPException(
            status_code=400,
            detail="Invalid intervention status"
        )

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    now = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    cursor.execute("""
        INSERT INTO interventions (
            student_id,
            student_name,
            intervention_type,
            description,
            status,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (

        intervention.student_id,
        intervention.student_name,
        intervention.intervention_type,
        intervention.description,
        intervention.status,
        now,
        now

    ))

    intervention_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return {
        "message": "Intervention created successfully",
        "intervention_id": intervention_id,
        "status": intervention.status
    }


# =========================================================
# GET STUDENT INTERVENTIONS
# =========================================================

@app.get("/interventions/{student_id}")
def get_interventions(student_id: str):

    conn = sqlite3.connect(DB_PATH)

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM interventions
        WHERE student_id = ?
        ORDER BY id DESC
    """, (student_id,))

    interventions = [
        dict(row)
        for row in cursor.fetchall()
    ]

    conn.close()

    return interventions


# =========================================================
# UPDATE INTERVENTION STATUS
# =========================================================

@app.put("/interventions/{intervention_id}/status")
def update_intervention_status(
    intervention_id: int,
    status: str
):

    allowed_status = [
        "OPEN",
        "IN PROGRESS",
        "RESOLVED"
    ]

    if status not in allowed_status:

        raise HTTPException(
            status_code=400,
            detail="Status must be OPEN, IN PROGRESS or RESOLVED"
        )

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE interventions
        SET
            status = ?,
            updated_at = ?
        WHERE id = ?
    """, (

        status,
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        intervention_id

    ))

    if cursor.rowcount == 0:

        conn.close()

        raise HTTPException(
            status_code=404,
            detail="Intervention not found"
        )

    conn.commit()
    conn.close()

    return {
        "message": "Intervention status updated",
        "intervention_id": intervention_id,
        "status": status
    }


# =========================================================
# GET ALL INTERVENTIONS
# =========================================================

@app.get("/interventions")
def get_all_interventions():

    conn = sqlite3.connect(DB_PATH)

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM interventions
        ORDER BY id DESC
    """)

    interventions = [
        dict(row)
        for row in cursor.fetchall()
    ]

    conn.close()

    return interventions