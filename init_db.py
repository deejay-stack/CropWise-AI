"""Create any missing tables once after verifying DATABASE_URL.

Does not migrate existing tables or transfer existing local PostgreSQL data.
"""
from sqlalchemy import text
from app import app, db

with app.app_context():
    db.session.execute(text("SELECT 1"))
    db.create_all()
    print("Database connection succeeded; missing tables created.")
