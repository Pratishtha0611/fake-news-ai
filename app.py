import sqlite3
import re
import csv
import io
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, make_response
from werkzeug.security import generate_password_hash, check_password_hash
from transformers import pipeline
from flask_wtf.csrf import CSRFProtect
import PyPDF2
from textblob import TextBlob

app = Flask(__name__)
app.secret_key = 'fake_news_secret_key'
csrf = CSRFProtect(app)
DATABASE = 'app.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

# Initialize Database
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    
    # Safely add is_admin column if it doesn't exist
    cursor.execute("PRAGMA table_info(users)")
    columns = [info[1] for info in cursor.fetchall()]
    if 'is_admin' not in columns:
        cursor.execute('ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0')
        
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            news TEXT NOT NULL,
            result TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            user_id INTEGER,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    
    # Add user_feedback column if it doesn't exist
    cursor.execute("PRAGMA table_info(predictions)")
    columns = [info[1] for info in cursor.fetchall()]
    if 'user_feedback' not in columns:
        cursor.execute('ALTER TABLE predictions ADD COLUMN user_feedback INTEGER DEFAULT 0')
    conn.commit()
    conn.close()

init_db()

# Custom login_required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ML Setup: HuggingFace Pipeline (Zero-Shot / Pre-trained Fake News)
print("Loading HuggingFace Fake News Transformer...")
hf_pipeline = pipeline("text-classification", model="hamzab/roberta-fake-news-classification")
print("Model loaded successfully!")

# Context processor
@app.context_processor
def inject_user():
    user = None
    if 'user_id' in session:
        conn = get_db_connection()
        user_row = conn.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
        conn.close()
        if user_row:
            class CurrentUser:
                is_authenticated = True
                id = user_row['id']
                name = user_row['name']
                email = user_row['email']
                is_admin = bool(user_row['is_admin']) if 'is_admin' in user_row.keys() else False
            user = CurrentUser()
    
    if not user:
        class GuestUser:
            is_authenticated = False
            id = None
            name = None
            email = None
            is_admin = False
        user = GuestUser()
        
    return dict(current_user=user)

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.is_json:
            news = request.json.get('news', '')
            is_ajax = True
        else:
            news = request.form.get('news', '')
            is_ajax = False

        if not news.strip():
            if is_ajax:
                return jsonify({'error': 'News text cannot be empty'}), 400
            else:
                flash("News text cannot be empty", "error")
                return redirect(url_for('dashboard'))

        # URL Extraction Logic
        if news.strip().startswith('http://') or news.strip().startswith('https://'):
            try:
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(news.strip(), headers=headers, timeout=5)
                soup = BeautifulSoup(response.text, 'html.parser')
                paragraphs = soup.find_all('p')
                extracted_text = " ".join([p.get_text() for p in paragraphs])
                if len(extracted_text) > 50:
                    news = extracted_text
                else:
                    return jsonify({'error': 'Could not extract enough text from the URL.'}), 400
            except Exception as e:
                return jsonify({'error': 'Failed to fetch the URL.'}), 400

        # Smart Heuristic Layer for Short Inputs
        text_lower = news.lower()
        real_keywords = ['government', 'police', 'court', 'report', 'official', 'announced', 'said', 'according to', 'minister', 'president', 'hospital', 'news', 'update', 'today', 'yesterday', 'released', 'launch', 'new', 'plan', 'economy', 'market', 'stock', 'company', 'apple', 'google', 'microsoft', 'india', 'usa']
        fake_keywords = ['alien', 'miracle', 'shocking', 'omg', 'secret', 'illuminati', 'mind blowing', "you won't believe", 'cure', 'magic', 'scam', 'hoax', 'unbelievable', 'fake', 'zombie', 'ghost']
        
        found_real_words = [k for k in real_keywords if k in text_lower]
        found_fake_words = [k for k in fake_keywords if k in text_lower]
        
        real_count = len(found_real_words)
        fake_count = len(found_fake_words)
        
        if fake_count > 0 and fake_count >= real_count:
            result = "Fake News"
            confidence_score = round(float(90 + (fake_count * 1.5)), 2)
            if confidence_score > 99.9: confidence_score = 99.9
        elif real_count > 0 and real_count > fake_count:
            result = "Real News"
            confidence_score = round(float(85 + (real_count * 2.0)), 2)
            if confidence_score > 99.9: confidence_score = 99.9
        elif len(news.split()) < 50:
            # Headlines are short and usually real if they don't have fake keywords
            result = "Real News"
            confidence_score = round(float(80 + (len(news.split()) * 0.3)), 2)
        else:
            # Fallback to HuggingFace pipeline for complex/long text
            truncated_news = news[:1500] 
            hf_result = hf_pipeline(truncated_news)[0]
            
            label = str(hf_result['label']).upper()
            confidence = float(hf_result['score'])
            confidence_score = round(confidence * 100, 2)
            
            if 'FAKE' in label or 'FALSE' in label:
                result = "Fake News"
            elif 'REAL' in label or 'TRUE' in label:
                result = "Real News"
            elif '0' in label:
                result = "Fake News"
            elif '1' in label:
                result = "Real News"
            else:
                result = "Fake News" if confidence_score > 50 else "Real News"

        # Save to database
        user_id = session.get('user_id')
        created_at = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO predictions (news, result, confidence, created_at, user_id) VALUES (?, ?, ?, ?, ?)',
            (news, result, confidence_score, created_at, user_id)
        )
        prediction_id = cursor.lastrowid
        conn.commit()

        if is_ajax:
            conn.close()
            return jsonify({
                'prediction_id': prediction_id,
                'prediction': result,
                'confidence': confidence_score,
                'found_real_words': found_real_words,
                'found_fake_words': found_fake_words
            })
        else:
            user_predictions = conn.execute('SELECT * FROM predictions WHERE user_id = ?', (user_id,)).fetchall()
            conn.close()
            total_predictions = len(user_predictions)
            fake_count = sum(1 for p in user_predictions if p['result'] == 'Fake News')
            real_count = sum(1 for p in user_predictions if p['result'] == 'Real News')
            return render_template('dashboard.html', prediction=result, confidence=confidence_score, 
                                   total_predictions=total_predictions, fake_count=fake_count, real_count=real_count)
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        if request.is_json:
            return jsonify({'error': 'An error occurred during prediction.'}), 500
        else:
            flash("An error occurred during prediction.", "error")
            return redirect(url_for('dashboard'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')

        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        
        if user:
            conn.close()
            flash('Email already exists. Please login.', 'error')
            return redirect(url_for('login'))

        hashed_password = generate_password_hash(password, method='scrypt')
        is_admin = 1 if email == 'admin@test.com' else 0
        
        conn.execute('INSERT INTO users (name, email, password, is_admin) VALUES (?, ?, ?, ?)', 
                     (name, email, hashed_password, is_admin))
        conn.commit()
        conn.close()

        flash('Account created successfully! You can now log in.', 'success')
        return redirect(url_for('login'))
        
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            return redirect(url_for('dashboard'))
        else:
            flash('Please check your login details and try again.', 'error')
            
    return render_template('login.html')

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email')
        
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()

        if user:
            # In a real app, send an email. For this demo, redirect directly to reset.
            flash('Email found! You can now reset your password.', 'success')
            return redirect(url_for('reset_password', email=email))
        else:
            flash('No account found with that email address.', 'error')

    return render_template('forgot_password.html')

@app.route('/reset_password/<email>', methods=['GET', 'POST'])
def reset_password(email):
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('reset_password.html', email=email)

        hashed_password = generate_password_hash(password, method='scrypt')
        
        conn = get_db_connection()
        conn.execute('UPDATE users SET password = ? WHERE email = ?', (hashed_password, email))
        conn.commit()
        conn.close()

        flash('Your password has been successfully reset! You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('reset_password.html', email=email)

@app.route('/logout')
@login_required
def logout():
    session.pop('user_id', None)
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/analytics')
@login_required
def analytics():
    conn = get_db_connection()
    user_predictions = conn.execute('SELECT * FROM predictions WHERE user_id = ?', (session['user_id'],)).fetchall()
    conn.close()
    
    total_predictions = len(user_predictions)
    fake_count = sum(1 for p in user_predictions if p['result'] == 'Fake News')
    real_count = sum(1 for p in user_predictions if p['result'] == 'Real News')

    return render_template(
        'analytics.html',
        total_predictions=total_predictions,
        fake_count=fake_count,
        real_count=real_count
    )

@app.route('/history')
@login_required
def history():
    conn = get_db_connection()
    raw_history = conn.execute('SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC', (session['user_id'],)).fetchall()
    conn.close()
    
    history_data = []
    for row in raw_history:
        row_dict = dict(row)
        try:
            row_dict['created_at'] = datetime.strptime(row_dict['created_at'], '%Y-%m-%d %H:%M:%S')
        except:
            row_dict['created_at'] = datetime.utcnow()
        history_data.append(row_dict)

    return render_template('history.html', history=history_data)

@app.route('/admin')
@login_required
def admin():
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
    
    if not user or not ('is_admin' in user.keys() and user['is_admin']): 
        conn.close()
        flash('Access denied. Administrator privileges required.', 'error')
        return redirect(url_for('dashboard'))
        
    users = conn.execute('SELECT * FROM users').fetchall()
    raw_predictions = conn.execute('SELECT * FROM predictions ORDER BY created_at DESC').fetchall()
    conn.close()

    predictions = []
    for row in raw_predictions:
        row_dict = dict(row)
        try:
            row_dict['created_at'] = datetime.strptime(row_dict['created_at'], '%Y-%m-%d %H:%M:%S')
        except:
            row_dict['created_at'] = datetime.utcnow()
        predictions.append(row_dict)

    return render_template('admin.html', users=users, predictions=predictions)

@app.route('/admin/delete/<int:user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    conn = get_db_connection()
    current = conn.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
    
    if not current or not ('is_admin' in current.keys() and current['is_admin']): 
        conn.close()
        flash('Access denied.', 'error')
        return redirect(url_for('dashboard'))

    if session['user_id'] == user_id:
        conn.close()
        flash('You cannot delete your own admin account.', 'error')
        return redirect(url_for('admin'))

    conn.execute('DELETE FROM predictions WHERE user_id = ?', (user_id,))
    conn.execute('DELETE FROM users WHERE id = ?', (user_id,))
    conn.commit()
    conn.close()
    
    flash('User deleted successfully.', 'success')
    return redirect(url_for('admin'))

@app.route('/feedback', methods=['POST'])
@login_required
def feedback():
    if not request.is_json:
        return jsonify({'error': 'Invalid request format'}), 400
    
    data = request.json
    prediction_id = data.get('prediction_id')
    feedback_value = data.get('feedback') # 1 for correct, -1 for incorrect

    if prediction_id is None or feedback_value not in [1, -1]:
        return jsonify({'error': 'Invalid feedback data'}), 400
        
    conn = get_db_connection()
    row = conn.execute('SELECT * FROM predictions WHERE id = ? AND user_id = ?', (prediction_id, session['user_id'])).fetchone()
    if not row:
        conn.close()
        return jsonify({'error': 'Prediction not found or access denied'}), 404
        
    conn.execute('UPDATE predictions SET user_feedback = ? WHERE id = ?', (feedback_value, prediction_id))
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'message': 'Feedback recorded!'})

@app.route('/export')
@login_required
def export_history():
    conn = get_db_connection()
    user_predictions = conn.execute('SELECT news, result, confidence, created_at, user_feedback FROM predictions WHERE user_id = ? ORDER BY created_at DESC', (session['user_id'],)).fetchall()
    conn.close()

    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['Date', 'News Snippet', 'Result', 'Confidence (%)', 'User Feedback'])
    
    for row in user_predictions:
        feedback_str = 'Not Provided'
        if row['user_feedback'] == 1:
            feedback_str = 'Correct'
        elif row['user_feedback'] == -1:
            feedback_str = 'Incorrect'
            
        snippet = row['news'][:100].replace('\n', ' ') + '...' if len(row['news']) > 100 else row['news'].replace('\n', ' ')
        cw.writerow([row['created_at'], snippet, row['result'], row['confidence'], feedback_str])

    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=fake_news_history.csv"
    output.headers["Content-type"] = "text/csv"
    return output

@app.route('/upload_file', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400
        
    extracted_text = ""
    try:
        if file.filename.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + " "
        elif file.filename.endswith('.txt'):
            extracted_text = file.read().decode('utf-8')
        else:
            return jsonify({'error': 'Unsupported file type. Please upload .txt or .pdf.'}), 400
            
        if len(extracted_text.strip()) < 10:
            return jsonify({'error': 'Could not extract enough text from the file.'}), 400
            
        return jsonify({'text': extracted_text.strip()})
    except Exception as e:
        return jsonify({'error': f'Failed to process file: {str(e)}'}), 400

@app.route('/analyze_extra', methods=['POST'])
@login_required
def analyze_extra():
    if not request.is_json:
        return jsonify({'error': 'Invalid request'}), 400
        
    text = request.json.get('text', '')
    if not text.strip():
        return jsonify({'error': 'Empty text'}), 400
        
    # Emotion Analysis using TextBlob
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    if polarity < -0.3:
        tone = "Negative / Angry / Fearful"
        emotion_level = "High"
    elif polarity > 0.3:
        tone = "Positive / Enthusiastic"
        emotion_level = "Medium"
    else:
        tone = "Neutral / Objective"
        emotion_level = "Low"
        
    if subjectivity > 0.6:
        emotion_level = "High (Highly Opinionated)"
        
    # Auto-Summarization (Basic Extraction)
    sentences = blob.sentences
    if len(sentences) <= 3:
        summary = [str(s) for s in sentences]
    else:
        valid_sentences = [s for s in sentences if 20 < len(str(s)) < 200]
        if len(valid_sentences) >= 3:
            summary = [str(s) for s in valid_sentences[:3]]
        else:
            summary = [str(s) for s in sentences[:3]]
            
    return jsonify({
        'tone': tone,
        'emotion_level': emotion_level,
        'summary': summary
    })

@app.route('/delete_my_account', methods=['POST'])
@login_required
def delete_my_account():
    user_id = session['user_id']
    conn = get_db_connection()
    # Delete all predictions associated with this user
    conn.execute('DELETE FROM predictions WHERE user_id = ?', (user_id,))
    # Delete the user account
    conn.execute('DELETE FROM users WHERE id = ?', (user_id,))
    conn.commit()
    conn.close()
    
    # Log the user out
    session.clear()
    flash("Your account and all history have been permanently deleted.", "success")
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)