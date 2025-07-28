from flask import Flask, request, jsonify
import pickle
import pandas as pd
import re
import string

with open('Models/arabic_symptom_model.pkl', 'rb') as f:
    model = pickle.load(f)

doctors_df = pd.read_csv('Data Sets/doctors_data.csv')

AR_STOPWORDS = set([
    'في', 'من', 'على', 'عن', 'إلى', 'كما', 'هذا', 'هذه', 'هناك',
    'هو', 'هي', 'هم', 'نحن', 'أيضاً', 'حتى', 'مع', 'كل', 'أكثر', 'أقل',
    'لقد', 'ولا', 'منه', 'فيه', 'بين', 'بعد', 'عند', 'بدون', 'أحد', 'أي'
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

app = Flask(__name__)

@app.route("/recommend_doctors", methods=["POST"])
def recommend_doctors():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "برجاء إدخال الأعراض"}), 400

    cleaned = clean_arabic_text(text)
    predicted_specialty = model.predict([cleaned])[0]

    # Filter doctors by predicted specialty
    filtered_doctors = doctors_df[doctors_df['specialization'] == predicted_specialty]
    top_doctors = filtered_doctors.sort_values(by='rating', ascending=False)

    # Convert to list of dicts
    result = top_doctors[['doctor_name', 'specialization', 'rating', 'location']].to_dict(orient='records')

    return jsonify({
        "predicted_specialty": predicted_specialty,
        "recommended_doctors": result
    })

if __name__ == "__main__":
    app.run(debug=True)
