# 📧 Spam Detection 
A web application that uses **Machine Learning** to classify email messages as **Spam** or **Ham** (not spam).  
The project features a **modern glassmorphism UI**, dynamic history logging, and real-time data visualization.

---

## ✨ Features
- **Intelligent Spam Detection** – Powered by a **Multinomial Naive Bayes** classifier trained on a balanced dataset of over **4,000 messages** for accurate predictions.  
- **Professional UI** – Sleek and responsive frontend with a **liquid glass (glassmorphism)** effect.  
- **Light & Dark Themes** – A **theme switcher** allows users to toggle between **light** and **dark** modes, with preferences saved across sessions.  
- **Prediction History** – Every analyzed message is stored in a **local SQLite database** for easy tracking.  
- **History Modal** – A clean **pop-up modal** displays recent predictions without cluttering the interface.  
- **Asynchronous Deletion** – Users can delete history entries **instantly** without reloading the page.  
- **Dynamic Data Visualization** – A **real-time pie chart** shows the overall distribution of spam vs. ham predictions.

---

## 🛠️ Technologies Used
- **Backend:** Python, Flask  
- **Machine Learning:** Scikit-learn (MultinomialNB, CountVectorizer), Pandas  
- **Database:** SQLite  
- **Frontend:** HTML, CSS, JavaScript  
- **Data Visualization:** Chart.js  

---

## 📂 Project Structure
```
spam-detection-app/
├── static/
│   └── style.css            # Styling for the application
├── templates/
│   └── index.html           # Main HTML page
├── app.py                   # Flask server and ML logic
├── spam_ham_dataset_4000.csv # Balanced dataset for training
├── database.db              # SQLite database (auto-created on first run)
└── README.md                # Project documentation
```

---

## 🚀 How to Run Locally

### ✅ Prerequisites
- Python **3.x** installed on your system.  
- `pip` package manager.

### 1️⃣ Set Up the Project
Clone the repository or download the files into a new folder.  
Ensure that `app.py`, the **static** and **templates** folders, and `spam_ham_dataset_4000.csv` are all in the **root directory**.

### 2️⃣ Install Dependencies
Open a terminal in the project folder and run:
```bash
pip install flask pandas scikit-learn
```

### 3️⃣ Run the Flask Server
Start the server with:
```bash
python app.py
```
You’ll see output indicating the server is running, typically at:
```
http://127.0.0.1:5000
```

### 4️⃣ Open in Browser
Navigate to:
```
http://127.0.0.1:5000
```
Your **Spam Detection** app is now live! 🎉  
The `database.db` file will be automatically created during the first run.
---

## 💡 Future Enhancements
- Integration with email clients (e.g., Gmail API) for real-time spam detection.  
- Advanced NLP techniques like **Transformer-based models** (BERT, RoBERTa).  
- User authentication for personalized history tracking.

---
