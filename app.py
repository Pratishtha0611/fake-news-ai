import os
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
from pymongo import MongoClient
from bson.objectid import ObjectId
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'fake_news_secret_key')
csrf = CSRFProtect(app)

# MongoDB Connection
MONGO_URI = os.environ.get('MONGO_URI')
if MONGO_URI and MONGO_URI != "your_mongodb_atlas_connection_string_here":
    client = MongoClient(MONGO_URI)
else:
    # Fallback for local development if URI is not set
    client = MongoClient('mongodb://localhost:27017/')

db = client['fake_news_db']
users_collection = db['users']
predictions_collection = db['predictions']

# Ensure email is unique
users_collection.create_index("email", unique=True)

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
        try:
            user_row = users_collection.find_one({'_id': ObjectId(session['user_id'])})
            if user_row:
                class CurrentUser:
                    is_authenticated = True
                    id = str(user_row['_id'])
                    name = user_row.get('name', '')
                    email = user_row.get('email', '')
                    is_admin = bool(user_row.get('is_admin', False))
                user = CurrentUser()
        except Exception as e:
            print(f"Error fetching user: {e}")
            session.pop('user_id', None)
    
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
        prediction_doc = {
            'news': news,
            'result': result,
            'confidence': confidence_score,
            'created_at': datetime.utcnow(),
            'user_id': ObjectId(user_id) if user_id else None,
            'user_feedback': 0
        }
        
        insert_result = predictions_collection.insert_one(prediction_doc)
        prediction_id = str(insert_result.inserted_id)

        if is_ajax:
            return jsonify({
                'prediction_id': prediction_id,
                'prediction': result,
                'confidence': confidence_score,
                'found_real_words': found_real_words,
                'found_fake_words': found_fake_words
            })
        else:
            user_predictions = list(predictions_collection.find({'user_id': ObjectId(user_id)}))
            total_predictions = len(user_predictions)
            fake_count = sum(1 for p in user_predictions if p.get('result') == 'Fake News')
            real_count = sum(1 for p in user_predictions if p.get('result') == 'Real News')
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

        user = users_collection.find_one({'email': email})
        
        if user:
            flash('Email already exists. Please login.', 'error')
            return redirect(url_for('login'))

        hashed_password = generate_password_hash(password, method='scrypt')
        is_admin = True if email == 'admin@test.com' else False
        
        users_collection.insert_one({
            'name': name,
            'email': email,
            'password': hashed_password,
            'is_admin': is_admin
        })

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

        user = users_collection.find_one({'email': email})

        if user and check_password_hash(user['password'], password):
            session['user_id'] = str(user['_id'])
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
        
        user = users_collection.find_one({'email': email})

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
        
        users_collection.update_one({'email': email}, {'$set': {'password': hashed_password}})

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
    user_predictions = list(predictions_collection.find({'user_id': ObjectId(session['user_id'])}))
    
    total_predictions = len(user_predictions)
    fake_count = sum(1 for p in user_predictions if p.get('result') == 'Fake News')
    real_count = sum(1 for p in user_predictions if p.get('result') == 'Real News')

    return render_template(
        'analytics.html',
        total_predictions=total_predictions,
        fake_count=fake_count,
        real_count=real_count
    )

@app.route('/history')
@login_required
def history():
    raw_history = list(predictions_collection.find({'user_id': ObjectId(session['user_id'])}).sort('created_at', -1))
    
    history_data = []
    for row in raw_history:
        row_dict = dict(row)
        row_dict['id'] = str(row_dict['_id'])
        # Handle created_at formatting
        if 'created_at' in row_dict and isinstance(row_dict['created_at'], datetime):
            pass # already a datetime object
        else:
            try:
                row_dict['created_at'] = datetime.strptime(str(row_dict.get('created_at', '')), '%Y-%m-%d %H:%M:%S')
            except:
                row_dict['created_at'] = datetime.utcnow()
        history_data.append(row_dict)

    return render_template('history.html', history=history_data)

@app.route('/admin')
@login_required
def admin():
    user = users_collection.find_one({'_id': ObjectId(session['user_id'])})
    
    if not user or not user.get('is_admin', False): 
        flash('Access denied. Administrator privileges required.', 'error')
        return redirect(url_for('dashboard'))
        
    users = list(users_collection.find())
    for u in users:
        u['id'] = str(u['_id'])
        
    raw_predictions = list(predictions_collection.find().sort('created_at', -1))

    predictions = []
    for row in raw_predictions:
        row_dict = dict(row)
        row_dict['id'] = str(row_dict['_id'])
        # Add user email for admin view
        pred_user = users_collection.find_one({'_id': row_dict.get('user_id')})
        row_dict['user_email'] = pred_user.get('email', 'Unknown') if pred_user else 'Unknown'
        
        if 'created_at' in row_dict and isinstance(row_dict['created_at'], datetime):
            pass
        else:
            try:
                row_dict['created_at'] = datetime.strptime(str(row_dict.get('created_at', '')), '%Y-%m-%d %H:%M:%S')
            except:
                row_dict['created_at'] = datetime.utcnow()
        predictions.append(row_dict)

    return render_template('admin.html', users=users, predictions=predictions)

@app.route('/admin/delete/<string:user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    current = users_collection.find_one({'_id': ObjectId(session['user_id'])})
    
    if not current or not current.get('is_admin', False): 
        flash('Access denied.', 'error')
        return redirect(url_for('dashboard'))

    if session['user_id'] == user_id:
        flash('You cannot delete your own admin account.', 'error')
        return redirect(url_for('admin'))

    predictions_collection.delete_many({'user_id': ObjectId(user_id)})
    users_collection.delete_one({'_id': ObjectId(user_id)})
    
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

    if not prediction_id or feedback_value not in [1, -1]:
        return jsonify({'error': 'Invalid feedback data'}), 400
        
    try:
        row = predictions_collection.find_one({'_id': ObjectId(prediction_id), 'user_id': ObjectId(session['user_id'])})
        if not row:
            return jsonify({'error': 'Prediction not found or access denied'}), 404
            
        predictions_collection.update_one(
            {'_id': ObjectId(prediction_id)},
            {'$set': {'user_feedback': feedback_value}}
        )
        return jsonify({'success': True, 'message': 'Feedback recorded!'})
    except Exception as e:
        return jsonify({'error': 'Error recording feedback'}), 500

@app.route('/export')
@login_required
def export_history():
    user_predictions = list(predictions_collection.find({'user_id': ObjectId(session['user_id'])}).sort('created_at', -1))

    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['Date', 'News Snippet', 'Result', 'Confidence (%)', 'User Feedback'])
    
    for row in user_predictions:
        feedback_str = 'Not Provided'
        if row.get('user_feedback') == 1:
            feedback_str = 'Correct'
        elif row.get('user_feedback') == -1:
            feedback_str = 'Incorrect'
            
        news = row.get('news', '')
        snippet = news[:100].replace('\n', ' ') + '...' if len(news) > 100 else news.replace('\n', ' ')
        cw.writerow([row.get('created_at'), snippet, row.get('result'), row.get('confidence'), feedback_str])

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
    
    predictions_collection.delete_many({'user_id': ObjectId(user_id)})
    users_collection.delete_one({'_id': ObjectId(user_id)})
    
    session.clear()
    flash("Your account and all history have been permanently deleted.", "success")
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)