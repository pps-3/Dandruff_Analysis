"""
ScalpIQ — Authentication Blueprint
Handles signup, login, logout with PostgreSQL via psycopg2
"""

import os
import uuid
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from functools import wraps

import psycopg2
import psycopg2.extras
from flask import (Blueprint, request, jsonify, session,
                   redirect, url_for, render_template)
from werkzeug.security import generate_password_hash, check_password_hash

auth = Blueprint('auth', __name__)

# ── DB connection ─────────────────────────────────────────────────
def get_db():
    """Return a psycopg2 connection using env vars."""
    return psycopg2.connect(
        host     = os.getenv('DB_HOST',     'localhost'),
        port     = int(os.getenv('DB_PORT', 5432)),
        dbname   = os.getenv('DB_NAME',     'scalpiq'),
        user     = os.getenv('DB_USER',     'postgres'),
        password = os.getenv('DB_PASSWORD', 'postgres'),
        cursor_factory=psycopg2.extras.RealDictCursor
    )

# ── Login required decorator ──────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            if request.is_json:
                return jsonify({'success': False, 'error': 'Authentication required'}), 401
            return redirect(url_for('auth.login_page'))
        return f(*args, **kwargs)
    return decorated

# ── Helper: get current user ──────────────────────────────────────
def get_current_user():
    if 'user_id' not in session:
        return None
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, full_name, email, created_at FROM users WHERE id=%s AND is_active=TRUE",
                    (session['user_id'],))
        return cur.fetchone()
    finally:
        conn.close()

# ── Pages ─────────────────────────────────────────────────────────
@auth.route('/login')
def login_page():
    if 'user_id' in session:
        return redirect(url_for('index'))
    return render_template('login.html')

@auth.route('/signup')
def signup_page():
    if 'user_id' in session:
        return redirect(url_for('index'))
    return render_template('signup.html')

# ── API: Signup ───────────────────────────────────────────────────
@auth.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json()

    full_name = (data.get('full_name') or '').strip()
    email     = (data.get('email')     or '').strip().lower()
    password  = (data.get('password')  or '')

    # Validation
    if not full_name or len(full_name) < 2:
        return jsonify({'success': False, 'error': 'Please enter your full name.'}), 400
    if not email or '@' not in email:
        return jsonify({'success': False, 'error': 'Please enter a valid email address.'}), 400
    if len(password) < 8:
        return jsonify({'success': False, 'error': 'Password must be at least 8 characters.'}), 400
    if not any(c.isupper() for c in password):
        return jsonify({'success': False, 'error': 'Password must contain at least one uppercase letter.'}), 400
    if not any(c.isdigit() for c in password):
        return jsonify({'success': False, 'error': 'Password must contain at least one number.'}), 400

    conn = get_db()
    try:
        cur = conn.cursor()

        # Check existing email
        cur.execute("SELECT id FROM users WHERE email = %s", (email,))
        if cur.fetchone():
            return jsonify({'success': False, 'error': 'An account with this email already exists.'}), 409

        # Create user
        password_hash = generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)
        user_id = str(uuid.uuid4())

        cur.execute("""
            INSERT INTO users (id, full_name, email, password_hash, is_verified)
            VALUES (%s, %s, %s, %s, %s)
        """, (user_id, full_name, email, password_hash, True))  # set True for demo; use email verify in prod

        conn.commit()

        # Auto login after signup
        session['user_id']   = user_id
        session['user_name'] = full_name
        session['user_email'] = email
        session.permanent = True

        # Log last login
        cur.execute("UPDATE users SET last_login=NOW() WHERE id=%s", (user_id,))
        conn.commit()

        return jsonify({'success': True, 'redirect': '/', 'name': full_name})

    except Exception as e:
        conn.rollback()
        return jsonify({'success': False, 'error': f'Database error: {str(e)}'}), 500
    finally:
        conn.close()

# ── API: Login ────────────────────────────────────────────────────
@auth.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()

    email    = (data.get('email')    or '').strip().lower()
    password = (data.get('password') or '')

    if not email or not password:
        return jsonify({'success': False, 'error': 'Email and password are required.'}), 400

    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, full_name, email, password_hash, is_active FROM users WHERE email=%s", (email,))
        user = cur.fetchone()

        if not user:
            return jsonify({'success': False, 'error': 'Invalid email or password.'}), 401
        if not user['is_active']:
            return jsonify({'success': False, 'error': 'Your account has been deactivated.'}), 403
        if not check_password_hash(user['password_hash'], password):
            return jsonify({'success': False, 'error': 'Invalid email or password.'}), 401

        # Set session
        session['user_id']    = str(user['id'])
        session['user_name']  = user['full_name']
        session['user_email'] = user['email']
        session.permanent = True

        # Update last login
        cur.execute("UPDATE users SET last_login=NOW() WHERE id=%s", (user['id'],))
        conn.commit()

        return jsonify({'success': True, 'redirect': '/', 'name': user['full_name']})

    except Exception as e:
        return jsonify({'success': False, 'error': f'Database error: {str(e)}'}), 500
    finally:
        conn.close()

# ── API: Logout ───────────────────────────────────────────────────
@auth.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True, 'redirect': '/login'})

# ── API: Current user ─────────────────────────────────────────────
@auth.route('/api/me')
@login_required
def me():
    user = get_current_user()
    if not user:
        session.clear()
        return jsonify({'success': False, 'error': 'User not found'}), 404
    return jsonify({
        'success':    True,
        'id':         str(user['id']),
        'full_name':  user['full_name'],
        'email':      user['email'],
        'created_at': user['created_at'].isoformat() if user['created_at'] else None
    })

# ── API: Save analysis result ─────────────────────────────────────
@auth.route('/api/save-analysis', methods=['POST'])
@login_required
def save_analysis():
    data = request.get_json()
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO analyses
              (user_id, age, gender, oil_frequency, conditioner_use,
               water_intake, diet_type, stress_level, sleep_hours,
               prediction, confidence, prob_low, prob_medium, prob_high,
               hair_care_index, diet_quality_score, stress_proxy, sleep_score)
            VALUES
              (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            session['user_id'],
            data.get('age'), data.get('gender'),
            data.get('oil_frequency'), data.get('conditioner_use'),
            data.get('water_intake'), data.get('diet_type'),
            data.get('stress_level'), data.get('sleep_hours'),
            data.get('prediction'), data.get('confidence'),
            data.get('prob_low'), data.get('prob_medium'), data.get('prob_high'),
            data.get('hair_care_index'), data.get('diet_quality_score'),
            data.get('stress_proxy'), data.get('sleep_score')
        ))
        conn.commit()
        return jsonify({'success': True})
    except Exception as e:
        conn.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        conn.close()

# ── API: Analysis history ─────────────────────────────────────────
@auth.route('/api/history')
@login_required
def history():
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, prediction, confidence, age, gender, created_at
            FROM analyses
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT 20
        """, (session['user_id'],))
        rows = cur.fetchall()
        return jsonify({'success': True, 'analyses': [dict(r) for r in rows]})
    finally:
        conn.close()