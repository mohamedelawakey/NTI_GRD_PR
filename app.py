from flask import Flask, request, jsonify, render_template
import pickle, re, string
import pandas as pd

doctors_df = pd.read_csv("doctors_data.csv")

with open("arabic_symptom_model.pkl", "rb") as f:
    model = pickle.load(f)

AR_STOPWORDS = set([
    'في', 'من', 'على', 'عن', 'إلى', 'أن', 'إن', 'كان', 'ما', 'لم', 'لن', 'قد', 'هذا', 'هذه', 'هناك',
    'هو', 'هي', 'هم', 'نحن', 'كما', 'أو', 'ثم', 'لكن', 'بل', 'أيضاً', 'حتى', 'مع', 'كل', 'أكثر', 'أقل',
    'لقد', 'ولا', 'له', 'منه', 'فيه', 'بين', 'بعد', 'عند', 'إلي', 'بدون', 'أحد', 'أي', 'شيء', 'لي',
    'به', 'فهو', 'فهي', 'كانت', 'ليس', 'لان', 'إذا', 'مثل', 'ضمن', 'أصبح', 'ذلك', 'والتي', 'والذي',
    'وهو', 'وهي', 'الذي', 'التي', 'الذين', 'اللذين', 'اللتي', 'بهذا', 'بها', 'بهم', 'بأن', 'فإن',
    'كما', 'إذ', 'حيث', 'أين', 'منذ', 'إما', 'أثناء', 'بينما', 'لأن', 'كأن', 'إلا', 'قد', 'حتي',
    'أو', 'و', 'ب', 'ل', 'ف', 'س', 'ك', 'ي', 'ت', 'ن', 'ا', 'إنه', 'أنها', 'وإن', 'أن', 'إن', 'مازال',
    'مايزال', 'مافتئ', 'ماانفك', 'ما برح', 'لايزال', 'ليس', 'ما', 'لاتزال', 'لاتزال', 'لن', 'لم', 'لا'
])

def clean_arabic_text(text):
    text = str(text).lower()
    text = re.sub(r'^\s*\d+\s*\.*\s*', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(rf"[{re.escape(string.punctuation)}]", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [w for w in words if w not in AR_STOPWORDS and len(w) > 1]
    return ' '.join(words)

app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "برجاء إدخال نص صالح"}), 400

    clean_text = clean_arabic_text(text)
    prediction = model.predict([clean_text])[0]
    return jsonify({"specialty": prediction})

@app.route("/specialty/<specialty>")
def show_doctors(specialty):
    matched_doctors = doctors_df[doctors_df['specialization'] == specialty]

    if matched_doctors.empty:
        return render_template("doctors.html", specialty=specialty, doctors=[])

    top_doctors = matched_doctors.sort_values(by='rating', ascending=False)
    doctors_list = top_doctors[['doctor_name', 'rating', 'location', 'phone_number']].to_dict(orient='records')

    return render_template("doctors.html", specialty=specialty, doctors=doctors_list)

if __name__ == "__main__":
    app.run(debug=True)