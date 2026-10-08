import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()


def buat_soal(materi, jumlah, tingkat):

    if not materi.strip():
        return "Silakan masukkan materi terlebih dahulu."

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

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            num_beams=5,
            repetition_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


with gr.Blocks(title="Generator Soal Essay") as demo:

    gr.Markdown(
        """
        # 📝 Generator Soal Essay
        Buat soal essay secara otomatis berdasarkan materi dan tingkat kesulitan.
        """
    )

    materi = gr.Textbox(
        label="Materi",
        placeholder="Masukkan materi..."
    )

    jumlah = gr.Number(
        label="Jumlah Soal",
        value=3,
        minimum=1,
        precision=0
    )

    tingkat = gr.Dropdown(
        choices=["Mudah", "Sedang", "Sulit"],
        value="Sedang",
        label="Tingkat Kesulitan"
    )

    tombol = gr.Button("Buat Soal")

    hasil = gr.Textbox(
        label="Hasil Soal",
        lines=10
    )

    tombol.click(
        fn=buat_soal,
        inputs=[materi, jumlah, tingkat],
        outputs=hasil
    )


demo.launch()
