from flask import Flask, request, jsonify
from video_utils import extract_audio, clip_key_segments
from summarizer import summarize_text
from quiz_generator import generate_quiz
from translator import translate_text
import whisper

app = Flask(__name__)
model = whisper.load_model("base")

@app.route("/upload", methods=["POST"])
def process_video():
    file = request.files["file"]
    filepath = f"./temp/{file.filename}"
    os.makedirs("temp", exist_ok=True)
    file.save(filepath)

    transcript = model.transcribe(filepath)["text"]
    summary = summarize_text(transcript)
    quiz = generate_quiz(summary)
    translated = translate_text(summary, "hi")
    clips = clip_key_segments(filepath)

    return jsonify({
        "transcript": transcript,
        "summary": summary,
        "quiz": quiz,
        "translated_summary": translated,
        "clips": clips
    })

if __name__ == "__main__":
    app.run(debug=True)
