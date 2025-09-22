from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sqlite3
import datetime

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

# --- Model Training with Balanced Dataset ---
df = pd.read_csv("spam_ham_dataset_4000.csv")
df['Spam'] = df['label'].apply(lambda x: 1 if x == 'spam' else 0)

X_train, X_test, y_train, y_test = train_test_split(df.message, df.Spam, test_size=0.25, random_state=42)

vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_counts, y_train)

y_pred_test = model.predict(X_test_counts)
accuracy = accuracy_score(y_test, y_pred_test)
print(f"Model trained with new dataset. Test accuracy: {accuracy:.4f}")
# ----------------------------------------

def get_history_and_stats():
    """Helper function to get history and chart data."""
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM predictions ORDER BY id DESC") # Get full history for modal
    history = cur.fetchall()
    
    cur.execute("SELECT COUNT(*) FROM predictions WHERE prediction='Spam'")
    spam_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM predictions WHERE prediction='Ham'")
    ham_count = cur.fetchone()[0]
    
    conn.close()
    return history, spam_count, ham_count

@app.route('/')
def home():
    """Renders the main page."""
    history, spam_count, ham_count = get_history_and_stats()
    return render_template('index.html', theme='light', history=history, spam_count=spam_count, ham_count=ham_count)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the form submission, makes a prediction, saves it, and redirects."""
    if request.method == 'POST':
        message = request.form['message']
        theme = request.form.get('theme', 'light')
        
        if message:
            data = [message]
            vect = vectorizer.transform(data).toarray()
            prediction_code = model.predict(vect)[0]
            prediction_text = "Spam" if prediction_code == 1 else "Ham"
            
            last_id = None
            # **FIX:** Improved database connection and error handling
            try:
                con = sqlite3.connect("database.db")
                cur = con.cursor()
                cur.execute("INSERT INTO predictions (message,prediction,timestamp) VALUES (?,?,?)",
                            (message, prediction_text, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                con.commit()
                last_id = cur.lastrowid
                print(f"Record added successfully with ID: {last_id}")
            except sqlite3.Error as e:
                print("Database error:", e)
                if con:
                    con.rollback()
            finally:
                if con:
                    con.close()

            if last_id:
                return redirect(url_for('results', prediction_id=last_id, theme=theme))
        
    return redirect(url_for('home'))

@app.route('/results/<int:prediction_id>')
def results(prediction_id):
    """Displays the result of a specific prediction."""
    theme = request.args.get('theme', 'light')
    history, spam_count, ham_count = get_history_and_stats()

    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM predictions WHERE id=?", (prediction_id,))
    prediction_data = cur.fetchone()
    conn.close()

    prediction_code = 1 if prediction_data and prediction_data['prediction'] == 'Spam' else 0
    message = prediction_data['message'] if prediction_data else ""

    return render_template('index.html', prediction=prediction_code, message=message, theme=theme, history=history, spam_count=spam_count, ham_count=ham_count)


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

if __name__ == '__main__':
    app.run(debug=True)