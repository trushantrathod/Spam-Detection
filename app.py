from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sqlite3
import datetime
import numpy as np

app = Flask(__name__)

# --- Database Setup ---
def init_db():
    conn = sqlite3.connect('database.db')
    print("Opened database successfully")
    conn.execute('CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, message TEXT, prediction TEXT, timestamp TEXT)')
    print("Table created successfully")
    conn.close()

init_db()
# --------------------

# --- Model Training with Multinomial Naive Bayes ---
df = pd.read_csv("spam_ham_dataset_4000.csv")
df['Spam'] = df['label'].apply(lambda x: 1 if x == 'spam' else 0)

# Drop missing values
df.dropna(subset=['message'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df.message, df.Spam, test_size=0.25, random_state=42)

vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_counts, y_train)

y_pred_test = model.predict(X_test_counts)
accuracy = accuracy_score(y_test, y_pred_test)
print(f"Model trained with MultinomialNB. Test accuracy: {accuracy:.4f}")
# ----------------------------------------

def get_history_and_stats():
    """Helper function to get history and chart data."""
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM predictions ORDER BY id DESC")
    history = cur.fetchall()
    
    cur.execute("SELECT COUNT(*) FROM predictions WHERE prediction='Spam'")
    spam_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM predictions WHERE prediction='Ham'")
    ham_count = cur.fetchone()[0]
    
    conn.close()
    return history, spam_count, ham_count

def get_top_keywords(message_vector, prediction_code):
    """Identifies the most influential words by comparing class probabilities."""
    feature_names = np.array(vectorizer.get_feature_names_out())
    word_indices = message_vector.indices

    if not word_indices.any():
        return []

    # Log probabilities of features given a class, P(x_i|y)
    log_prob_ham = model.feature_log_prob_[0, word_indices]
    log_prob_spam = model.feature_log_prob_[1, word_indices]

    if prediction_code == 0: # Ham
        scores = log_prob_ham - log_prob_spam
    else: # Spam
        scores = log_prob_spam - log_prob_ham

    # Pair words with their scores and sort
    keywords = sorted(zip(feature_names[word_indices], scores), key=lambda item: item[1], reverse=True)
    
    return [word for word, score in keywords[:5]]


@app.route('/')
def home():
    """Renders the main page."""
    history, spam_count, ham_count = get_history_and_stats()
    return render_template('index.html', theme='dark', history=history, spam_count=spam_count, ham_count=ham_count, keywords=None)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the form submission, makes a prediction, saves it, and redirects."""
    if request.method == 'POST':
        message = request.form['message']
        theme = request.form.get('theme', 'dark')
        
        if message:
            con = sqlite3.connect("database.db")
            cur = con.cursor()
            
            cur.execute("SELECT id FROM predictions WHERE message = ?", (message,))
            existing_prediction = cur.fetchone()
            
            last_id = None
            if existing_prediction:
                last_id = existing_prediction[0]
                print(f"Message already exists. Found ID: {last_id}")
            else:
                data = [message]
                vect = vectorizer.transform(data)
                prediction_code = model.predict(vect)[0]
                prediction_text = "Spam" if prediction_code == 1 else "Ham"
                
                try:
                    cur.execute("INSERT INTO predictions (message,prediction,timestamp) VALUES (?,?,?)",
                                (message, prediction_text, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                    con.commit()
                    last_id = cur.lastrowid
                    print(f"Record added successfully with ID: {last_id}")
                except sqlite3.Error as e:
                    print("Database error:", e)
                    con.rollback()
            
            if con:
                con.close()

            if last_id:
                return redirect(url_for('results', prediction_id=last_id, theme=theme))
        
    return redirect(url_for('home'))

@app.route('/results/<int:prediction_id>')
def results(prediction_id):
    """Displays the result of a specific prediction."""
    theme = request.args.get('theme', 'dark')
    history, spam_count, ham_count = get_history_and_stats()

    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM predictions WHERE id=?", (prediction_id,))
    prediction_data = cur.fetchone()
    conn.close()

    if not prediction_data:
        return redirect(url_for('home'))

    message = prediction_data['message']
    prediction_text = prediction_data['prediction']
    prediction_code = 1 if prediction_text == 'Spam' else 0

    # --- Advanced ML Calculations ---
    vect = vectorizer.transform([message])
    probability_scores = model.predict_proba(vect)[0]
    confidence = round(max(probability_scores) * 100, 2)
    top_keywords = get_top_keywords(vect, prediction_code)

    return render_template('index.html', 
                           prediction=prediction_code, 
                           message=message, 
                           theme=theme, 
                           history=history, 
                           spam_count=spam_count, 
                           ham_count=ham_count,
                           confidence=confidence,
                           keywords=top_keywords)


@app.route('/delete/<int:prediction_id>', methods=['POST'])
def delete(prediction_id):
    """Handles asynchronous deletion of a history record."""
    try:
        with sqlite3.connect("database.db") as con:
            cur = con.cursor()
            cur.execute("DELETE FROM predictions WHERE id=?", (prediction_id,))
            con.commit()
            return jsonify({'status': 'success', 'message': 'Record deleted successfully'}), 200
    except Exception as e:
        con.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        if con:
            con.close()

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Deletes all records from the predictions table."""
    try:
        with sqlite3.connect("database.db") as con:
            cur = con.cursor()
            cur.execute("DELETE FROM predictions")
            con.commit()
            return jsonify({'status': 'success', 'message': 'History cleared successfully'}), 200
    except Exception as e:
        con.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        if con:
            con.close()

if __name__ == '__main__':
    app.run(debug=True)
