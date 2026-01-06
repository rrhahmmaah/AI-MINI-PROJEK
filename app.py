import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Inisialisasi Flask
app = Flask(__name__)

# Model lebih stabil
MODEL_NAME = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()

@app.route("/", methods=["GET", "POST"])
def index():
    soal = ""

    if request.method == "POST":
        materi = request.form.get("materi", "").strip()
        jumlah = int(request.form.get("jumlah", 3))
        tingkat = request.form.get("tingkat", "sedang")

        # Mapping tingkat kesulitan ke instruksi
        if tingkat.lower() == "mudah":
            level_text = "pertanyaan dasar dan mudah dipahami"
        elif tingkat.lower() == "sulit":
            level_text = "pertanyaan analitis dan mendalam"
        else:
            level_text = "pertanyaan tingkat sedang"

        prompt = (
            f"Buatkan {jumlah} soal essay dalam bahasa Indonesia berdasarkan materi berikut.\n"
            f"Materi: {materi}\n"
            f"Soal harus berupa {level_text}. "
            f"Tulis pertanyaan bernomor 1 sampai {jumlah}."
        )

        # Tokenisasi prompt
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512  # lebih panjang untuk teks yang lebih kompleks
        )

        # Generate soal
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=250,
                num_beams=5,
                repetition_penalty=2.0,
                no_repeat_ngram_size=3,
                early_stopping=True
            )

        soal = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return render_template("index.html", soal=soal)

# Jalankan app
if __name__ == "__main__":
    app.run(debug=True)
