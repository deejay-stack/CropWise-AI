
import os
import pickle
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from dotenv import load_dotenv

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify,
)

from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user,
)

from flask_bcrypt import Bcrypt

from sqlalchemy.engine import URL, make_url
from sqlalchemy.pool import NullPool
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from db import db


# ============================================================
# 1. ENVIRONMENT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

logging.basicConfig(level=logging.INFO)

IS_VERCEL = bool(os.getenv("VERCEL"))

IS_PRODUCTION = (
    os.getenv("APP_ENV", "development").lower() == "production"
    or IS_VERCEL
)


# ============================================================
# 2. APPLICATION SECURITY
# ============================================================

SECRET_KEY = os.getenv("SECRET_KEY")

if not SECRET_KEY:
    raise RuntimeError(
        "SECRET_KEY is missing. "
        "Add SECRET_KEY to your .env file locally "
        "or to the Vercel Environment Variables."
    )

app.config["SECRET_KEY"] = SECRET_KEY

app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = IS_PRODUCTION

app.config["REMEMBER_COOKIE_HTTPONLY"] = True
app.config["REMEMBER_COOKIE_SECURE"] = IS_PRODUCTION

app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024

app.config["JSON_SORT_KEYS"] = False


# ============================================================
# 3. SUPABASE POSTGRESQL DATABASE CONFIGURATION
# ============================================================

DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:

    # Support older PostgreSQL URL formats.
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace(
            "postgres://",
            "postgresql+psycopg://",
            1,
        )

    elif DATABASE_URL.startswith("postgresql://"):
        DATABASE_URL = DATABASE_URL.replace(
            "postgresql://",
            "postgresql+psycopg://",
            1,
        )

    elif DATABASE_URL.startswith("postgresql+psycopg2://"):
        DATABASE_URL = DATABASE_URL.replace(
            "postgresql+psycopg2://",
            "postgresql+psycopg://",
            1,
        )

else:

    # Optional support for separate PostgreSQL credentials.
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME")

    if not all([
        DB_USER,
        DB_PASSWORD,
        DB_HOST,
        DB_NAME,
    ]):
        raise RuntimeError(
            "Database configuration is missing. "
            "Configure DATABASE_URL or the DB_* "
            "environment variables."
        )

    DATABASE_URL = URL.create(
        drivername="postgresql+psycopg",
        username=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=int(DB_PORT),
        database=DB_NAME,
    )


app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False


# ============================================================
# 4. SERVERLESS DATABASE CONNECTION HANDLING
# ============================================================

parsed_database_url = make_url(DATABASE_URL)

database_port = parsed_database_url.port

is_transaction_pooler = database_port == 6543

engine_options = {
    "pool_pre_ping": True,
    "connect_args": {
        "connect_timeout": 10,
    },
}

# Supabase's transaction pooler does not support
# Psycopg automatic prepared statements.
if is_transaction_pooler:

    engine_options["connect_args"][
        "prepare_threshold"
    ] = None


# Avoid maintaining persistent SQLAlchemy connection
# pools across Vercel serverless function instances.
if IS_VERCEL:

    engine_options["poolclass"] = NullPool

else:

    engine_options["pool_recycle"] = 300

    engine_options["pool_size"] = 3

    engine_options["max_overflow"] = 2


app.config[
    "SQLALCHEMY_ENGINE_OPTIONS"
] = engine_options


# ============================================================
# 5. INITIALIZE FLASK EXTENSIONS
# ============================================================

db.init_app(app)

bcrypt = Bcrypt(app)

login_manager = LoginManager()

login_manager.init_app(app)

login_manager.login_view = "login"

login_manager.login_message_category = "danger"

login_manager.session_protection = "strong"


# ============================================================
# 6. DATABASE MODELS
# ============================================================

class User(db.Model, UserMixin):

    __tablename__ = "users"

    id = db.Column(
        db.Integer,
        primary_key=True,
    )

    fullname = db.Column(
        db.String(120),
        nullable=False,
    )

    email = db.Column(
        db.String(120),
        unique=True,
        nullable=False,
    )

    password = db.Column(
        db.String(255),
        nullable=False,
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
    )

    histories = db.relationship(
        "PredictionHistory",
        backref="user",
        lazy=True,
        cascade="all, delete-orphan",
    )


class PredictionHistory(db.Model):

    __tablename__ = "prediction_history"

    id = db.Column(
        db.Integer,
        primary_key=True,
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=False,
    )

    nitrogen = db.Column(
        db.Float,
        nullable=False,
    )

    phosphorus = db.Column(
        db.Float,
        nullable=False,
    )

    potassium = db.Column(
        db.Float,
        nullable=False,
    )

    temperature = db.Column(
        db.Float,
        nullable=False,
    )

    humidity = db.Column(
        db.Float,
        nullable=False,
    )

    ph = db.Column(
        db.Float,
        nullable=False,
    )

    rainfall = db.Column(
        db.Float,
        nullable=False,
    )

    predicted_crop = db.Column(
        db.String(100),
        nullable=False,
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
    )


# ============================================================
# 7. FLASK LOGIN USER LOADER
# ============================================================

@login_manager.user_loader
def load_user(user_id):

    try:

        return db.session.get(
            User,
            int(user_id),
        )

    except (TypeError, ValueError):

        return None


# ============================================================
# 8. MACHINE LEARNING MODEL
# ============================================================

MODEL_PATH = BASE_DIR / "RandomForest.pkl"

model = None

try:

    if MODEL_PATH.is_file():

        with MODEL_PATH.open("rb") as file:

            # Only load a trusted model created by your project.
            model = pickle.load(file)

        app.logger.info(
            "Random Forest model loaded successfully."
        )

    else:

        app.logger.error(
            "RandomForest.pkl is missing from the project root."
        )

except Exception:

    app.logger.exception(
        "Failed to load the Random Forest model."
    )

    model = None


# ============================================================
# 9. CROP INFORMATION
# ============================================================

CROP_INFO = {

    "rice": {
        "image": "rice.jpg",
        "reason": "Rice grows well in warm temperatures, high humidity, and areas with sufficient rainfall. It is suitable for soil conditions that can hold water.",
        "fertilizer": "Use nitrogen-rich fertilizer such as urea, and apply phosphorus and potassium based on soil test results.",
        "tips": "Maintain proper water levels in the field and avoid over-flooding during early growth stages."
    },

    "maize": {
        "image": "maize.jpg",
        "reason": "Maize is suitable for moderately warm temperatures, balanced nutrients, and well-drained soil.",
        "fertilizer": "Apply nitrogen fertilizer during early growth and add phosphorus for strong root development.",
        "tips": "Plant in rows with proper spacing and ensure enough sunlight exposure."
    },

    "chickpea": {
        "image": "chickpea.jpg",
        "reason": "Chickpea performs well in dry and cool conditions with moderate soil nutrients.",
        "fertilizer": "Use phosphorus-based fertilizer and avoid too much nitrogen because chickpea can fix nitrogen naturally.",
        "tips": "Avoid waterlogging and plant in well-drained soil."
    },

    "kidneybeans": {
        "image": "kidneybeans.jpg",
        "reason": "Kidney beans grow best in fertile, well-drained soil with moderate temperature and moisture.",
        "fertilizer": "Use compost and phosphorus-rich fertilizer to support pod formation.",
        "tips": "Avoid excessive watering and provide support if needed."
    },

    "pigeonpeas": {
        "image": "pigeonpeas.jpg",
        "reason": "Pigeon peas tolerate dry conditions and grow well in moderate soil fertility.",
        "fertilizer": "Apply phosphorus fertilizer and organic compost before planting.",
        "tips": "Plant in sunny areas and avoid heavy waterlogged soil."
    },

    "mothbeans": {
        "image": "mothbeans.jpg",
        "reason": "Moth beans are drought-tolerant and suitable for dry soil conditions.",
        "fertilizer": "Use organic manure and small amounts of phosphorus fertilizer.",
        "tips": "Best grown in warm climates with minimal irrigation."
    },

    "mungbean": {
        "image": "mungbean.jpg",
        "reason": "Mungbean grows well in warm temperatures with moderate rainfall and well-drained soil.",
        "fertilizer": "Apply phosphorus and potassium fertilizer. Avoid too much nitrogen.",
        "tips": "Use good-quality seeds and avoid planting during heavy rain."
    },

    "blackgram": {
        "image": "blackgram.jpg",
        "reason": "Blackgram is suitable for warm climates and moderate soil moisture.",
        "fertilizer": "Use compost and phosphorus fertilizer for better root and pod growth.",
        "tips": "Avoid waterlogging and control weeds during early growth."
    },

    "lentil": {
        "image": "lentil.jpg",
        "reason": "Lentil grows best in cool weather and well-drained soil with moderate fertility.",
        "fertilizer": "Apply phosphorus fertilizer and organic matter.",
        "tips": "Do not over-irrigate because lentils are sensitive to excess water."
    },

    "pomegranate": {
        "image": "pomegranate.jpg",
        "reason": "Pomegranate is suitable for warm and dry climates with well-drained soil.",
        "fertilizer": "Use compost, potassium, and balanced NPK fertilizer.",
        "tips": "Prune regularly and avoid excessive irrigation."
    },

    "banana": {
        "image": "banana.jpg",
        "reason": "Banana needs high humidity, warm temperature, and nutrient-rich soil.",
        "fertilizer": "Apply potassium-rich fertilizer, compost, and nitrogen during vegetative growth.",
        "tips": "Keep soil moist and protect plants from strong winds."
    },

    "mango": {
        "image": "mango.jpg",
        "reason": "Mango grows well in warm climates with moderate rainfall and well-drained soil.",
        "fertilizer": "Apply organic compost and balanced NPK fertilizer.",
        "tips": "Avoid waterlogging and prune trees to improve fruit production."
    },

    "grapes": {
        "image": "grapes.jpg",
        "reason": "Grapes prefer warm temperatures, good sunlight, and well-drained soil.",
        "fertilizer": "Use potassium and phosphorus fertilizer during flowering and fruiting.",
        "tips": "Provide trellis support and prune vines regularly."
    },

    "watermelon": {
        "image": "watermelon.jpg",
        "reason": "Watermelon grows well in warm soil, high sunlight, and moderate rainfall.",
        "fertilizer": "Use compost and potassium-rich fertilizer for fruit development.",
        "tips": "Provide wide spacing and avoid overhead watering."
    },

    "muskmelon": {
        "image": "muskmelon.jpg",
        "reason": "Muskmelon is suitable for warm and dry conditions with well-drained soil.",
        "fertilizer": "Apply compost and potassium fertilizer.",
        "tips": "Use mulching to conserve soil moisture."
    },

    "apple": {
        "image": "apple.jpg",
        "reason": "Apple grows best in cooler climates with balanced nutrients and proper soil moisture.",
        "fertilizer": "Use nitrogen, phosphorus, and potassium depending on tree age and soil test.",
        "tips": "Prune yearly and protect from pests and fungal diseases."
    },

    "orange": {
        "image": "orange.jpg",
        "reason": "Orange prefers warm climates, moderate rainfall, and slightly acidic to neutral soil.",
        "fertilizer": "Use citrus fertilizer with nitrogen, potassium, and micronutrients.",
        "tips": "Water regularly but avoid waterlogging."
    },

    "papaya": {
        "image": "papaya.jpg",
        "reason": "Papaya grows well in warm temperatures, high humidity, and fertile soil.",
        "fertilizer": "Apply compost and balanced NPK fertilizer regularly.",
        "tips": "Plant in sunny areas and protect from strong winds."
    },

    "coconut": {
        "image": "coconut.jpg",
        "reason": "Coconut is suitable for humid tropical areas with high rainfall and sandy or loamy soil.",
        "fertilizer": "Use potassium-rich fertilizer and organic manure.",
        "tips": "Maintain proper spacing and provide irrigation during dry months."
    },

    "cotton": {
        "image": "cotton.jpg",
        "reason": "Cotton grows well in warm climates with moderate rainfall and fertile soil.",
        "fertilizer": "Apply nitrogen, phosphorus, and potassium fertilizer based on soil needs.",
        "tips": "Control weeds and pests early to protect boll formation."
    },

    "jute": {
        "image": "jute.jpg",
        "reason": "Jute grows well in warm, humid conditions with high rainfall.",
        "fertilizer": "Use nitrogen and phosphorus fertilizer with organic compost.",
        "tips": "Keep the field moist and harvest at the right maturity stage."
    },

    "coffee": {
        "image": "coffee.jpg",
        "reason": "Coffee grows best in moderate temperatures, good rainfall, and slightly acidic soil.",
        "fertilizer": "Use organic compost, nitrogen, and potassium fertilizer.",
        "tips": "Provide partial shade and maintain proper pruning."
    }

}


# ============================================================
# 10. CROP IMAGE HELPER
# ============================================================

def get_crop_image_url(image_name):

    image_path = f"images/crops/{image_name}"

    local_image = (
        BASE_DIR
        / "static"
        / image_path
    )

    public_image = (
        BASE_DIR
        / "public"
        / "static"
        / image_path
    )

    if local_image.is_file() or public_image.is_file():

        return url_for(
            "static",
            filename=image_path,
        )

    return url_for(
        "static",
        filename="images/crops/default.jpg",
    )


# ============================================================
# 11. HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    if model is None:

        return jsonify({
            "status": "unavailable",
            "model": "not_loaded",
        }), 503

    return jsonify({
        "status": "ok",
        "model": "loaded",
        "application": "Crop Recommendation System",
    }), 200


# ============================================================
# 12. LANDING PAGE
# ============================================================

@app.route("/", methods=["GET"])
def landing():

    return render_template(
        "landing.html"
    )


# ============================================================
# 13. USER REGISTRATION
# ============================================================

@app.route("/signup", methods=["GET", "POST"])
def signup():

    if current_user.is_authenticated:

        return redirect(
            url_for("dashboard")
        )

    if request.method == "POST":

        fullname = request.form.get(
            "fullname", ""
        ).strip()

        email = request.form.get(
            "email", ""
        ).strip().lower()

        password = request.form.get(
            "password", ""
        )

        if not fullname or not email or not password:

            flash(
                "Please fill in all fields.",
                "danger",
            )

            return redirect(
                url_for("signup")
            )

        if len(fullname) > 120 or len(email) > 120:

            flash(
                "Name or email exceeds the allowed length.",
                "danger",
            )

            return redirect(
                url_for("signup")
            )

        if len(password) < 8:

            flash(
                "Password must contain at least 8 characters.",
                "danger",
            )

            return redirect(
                url_for("signup")
            )

        if len(password.encode("utf-8")) > 72:

            flash(
                "Password is too long. Use 72 bytes or fewer.",
                "danger",
            )

            return redirect(
                url_for("signup")
            )

        try:

            existing_user = User.query.filter_by(
                email=email
            ).first()

            if existing_user:

                flash(
                    "Email already exists. Please login instead.",
                    "danger",
                )

                return redirect(
                    url_for("signup")
                )

            hashed_password = (
                bcrypt.generate_password_hash(
                    password
                ).decode("utf-8")
            )

            new_user = User(
                fullname=fullname,
                email=email,
                password=hashed_password,
            )

            db.session.add(new_user)

            db.session.commit()

            flash(
                "Account created successfully. Please login.",
                "success",
            )

            return redirect(
                url_for("login")
            )

        except IntegrityError:

            db.session.rollback()

            flash(
                "Email already exists. Please login instead.",
                "danger",
            )

            return redirect(
                url_for("signup")
            )

        except SQLAlchemyError:

            db.session.rollback()

            app.logger.exception(
                "Database error during registration."
            )

            flash(
                "Registration is temporarily unavailable.",
                "danger",
            )

            return redirect(
                url_for("signup")
            )

    return render_template(
        "signup.html"
    )


# ============================================================
# 14. USER LOGIN
# ============================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if current_user.is_authenticated:

        return redirect(
            url_for("dashboard")
        )

    if request.method == "POST":

        email = request.form.get(
            "email", ""
        ).strip().lower()

        password = request.form.get(
            "password", ""
        )

        if not email or not password:

            flash(
                "Please enter your email and password.",
                "danger",
            )

            return render_template(
                "login.html"
            )

        try:

            user = User.query.filter_by(
                email=email
            ).first()

            if user and bcrypt.check_password_hash(
                user.password,
                password,
            ):

                login_user(user)

                return redirect(
                    url_for("dashboard")
                )

            flash(
                "Invalid email or password.",
                "danger",
            )

        except SQLAlchemyError:

            db.session.rollback()

            app.logger.exception(
                "Database error during login."
            )

            flash(
                "Login is temporarily unavailable.",
                "danger",
            )

    return render_template(
        "login.html"
    )


# ============================================================
# 15. USER DASHBOARD
# ============================================================

@app.route("/dashboard", methods=["GET"])
@login_required
def dashboard():

    return render_template(
        "dashboard.html",
        user=current_user,
    )


# ============================================================
# 16. RANDOM FOREST CROP PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
@login_required
def predict():

    if model is None:

        return jsonify({
            "error": (
                "The machine learning model is unavailable. "
                "Please check RandomForest.pkl."
            )
        }), 503

    data = request.get_json(silent=True)

    if not isinstance(data, dict):

        return jsonify({
            "error": "Please provide the crop inputs as JSON."
        }), 400

    feature_names = [
        "nitrogen",
        "phosphorus",
        "potassium",
        "temperature",
        "humidity",
        "ph",
        "rainfall",
    ]

    try:

        # Maintain the exact feature order used during
        # Random Forest model training.
        values = [
            float(data[name])
            for name in feature_names
        ]

        input_data = np.array(
            [values],
            dtype=np.float64,
        )

        if not np.isfinite(input_data).all():

            return jsonify({
                "error": "Please enter valid finite numbers."
            }), 400

    except (KeyError, ValueError, TypeError):

        return jsonify({
            "error": (
                "Invalid or missing input. "
                "Please provide nitrogen, phosphorus, "
                "potassium, temperature, humidity, "
                "ph, and rainfall."
            )
        }), 400

    try:

        prediction = model.predict(
            input_data
        )

        predicted_crop = (
            str(prediction[0])
            .strip()
            .lower()
        )

    except Exception:

        app.logger.exception(
            "Machine learning prediction error."
        )

        return jsonify({
            "error": "Unable to generate a crop prediction."
        }), 500

    # Save prediction history in Supabase PostgreSQL.
    try:

        history_record = PredictionHistory(
            user_id=current_user.id,
            nitrogen=values[0],
            phosphorus=values[1],
            potassium=values[2],
            temperature=values[3],
            humidity=values[4],
            ph=values[5],
            rainfall=values[6],
            predicted_crop=predicted_crop,
        )

        db.session.add(history_record)

        db.session.commit()

    except SQLAlchemyError:

        db.session.rollback()

        app.logger.exception(
            "Failed to save crop prediction history."
        )

        return jsonify({
            "error": (
                "Prediction was generated, but saving "
                "the result failed. Please try again."
            )
        }), 500

    crop_details = CROP_INFO.get(
        predicted_crop,
        {
            "image": "default.jpg",
            "reason": (
                "This crop matches the provided "
                "soil and climate conditions."
            ),
            "fertilizer": (
                "Use balanced fertilizer "
                "based on soil testing."
            ),
            "tips": (
                "Monitor moisture, soil pH, "
                "and pest activity regularly."
            ),
        },
    )

    image_url = get_crop_image_url(
        crop_details["image"]
    )

    return jsonify({
        "crop": predicted_crop.title(),
        "image": image_url,
        "reason": crop_details["reason"],
        "fertilizer": crop_details["fertilizer"],
        "tips": crop_details["tips"],
    }), 200


# ============================================================
# 17. PREDICTION HISTORY
# ============================================================

@app.route("/history", methods=["GET"])
@login_required
def history():

    try:

        records = (
            PredictionHistory.query
            .filter_by(
                user_id=current_user.id
            )
            .order_by(
                PredictionHistory.created_at.desc()
            )
            .all()
        )

        return render_template(
            "history.html",
            records=records,
        )

    except SQLAlchemyError:

        db.session.rollback()

        app.logger.exception(
            "Failed to load prediction history."
        )

        flash(
            "Unable to load prediction history.",
            "danger",
        )

        return redirect(
            url_for("dashboard")
        )


# ============================================================
# 18. USER LOGOUT
# ============================================================

@app.route("/logout", methods=["GET"])
@login_required
def logout():

    logout_user()

    flash(
        "You have been logged out successfully.",
        "success",
    )

    return redirect(
        url_for("landing")
    )


# ============================================================
# 19. APPLICATION ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def page_not_found(error):

    return jsonify({
        "error": "Page not found."
    }), 404


@app.errorhandler(413)
def request_too_large(error):

    return jsonify({
        "error": "Request size exceeds the allowed limit."
    }), 413


@app.errorhandler(500)
def internal_server_error(error):

    db.session.rollback()

    app.logger.error(
        "Internal server error: %s",
        error,
    )

    return jsonify({
        "error": "An internal server error occurred."
    }), 500


# ============================================================
# 20. DATABASE INITIALIZATION
# ============================================================

# IMPORTANT:
#
# Do NOT call db.create_all() automatically here.
#
# Vercel uses serverless functions. Importing this module
# must not create or modify database tables.
#
# Initialize your Supabase tables once, using init_db.py
# locally, with DATABASE_URL pointing to Supabase.
#
# Existing database models are preserved.


# ============================================================
# 21. APPLICATION ENTRY POINT
# ============================================================

# Vercel automatically detects the Flask application
# exposed through the variable named "app".
#
# It does not require app.run() to start the application.

if __name__ == "__main__":

    debug_mode = (
        not IS_PRODUCTION
        and os.getenv("FLASK_DEBUG", "0") == "1"
    )

    app.run(
        host="127.0.0.1",
        port=int(os.getenv("PORT", "5000")),
        debug=debug_mode,
    )
