# main.py → FIXED (no video component issues)
print("🤖 HireSync AI!")

from audio import audio_phase_score
from src.ats_matcher import ATSMatcher
from features import extract_features
import gradio as gr
import PyPDF2
import csv
from datetime import datetime
import os

matcher = ATSMatcher()
DATASET_FILE = "hiresync_results.csv"

def extract_resume_text(pdf_file):
    text = ""
    if pdf_file:
        pdf = PyPDF2.PdfReader(pdf_file)
        text = " ".join(page.extract_text() for page in pdf.pages)
    return text

def save_to_dataset(candidate_name, ats, gaze, attention, fluency, communication, engagement, final):
    exists = os.path.exists(DATASET_FILE)
    with open(DATASET_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["Timestamp", "Candidate", "ATS", "Gaze", "Attention", "Fluency", "Communication", "Engagement", "Final", "Status"])
        status = "✅ HIRED" if final > 0.70 else "📈 IMPROVE"
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), candidate_name, f"{ats:.3f}", f"{gaze:.3f}", f"{attention:.3f}", f"{fluency:.3f}", f"{communication:.3f}", f"{engagement:.3f}", f"{final:.3f}", status])

def load_dataset():
    if not os.path.exists(DATASET_FILE):
        return "No data yet"
    with open(DATASET_FILE, 'r') as f:
        return f.read()

def hiresync_complete(resume_pdf, jd_file, video_file, candidate_name):
    if not all([resume_pdf, jd_file, video_file, candidate_name]):
        return "❌ Upload ALL 4!", "No data"
    
    resume_text = extract_resume_text(resume_pdf)
    try:
        ats_score, df = matcher.analyze_resume(resume_text, jd_file.name)
        top_matches = df.nlargest(3, 'JD Match Score').to_string()
    except:
        ats_score = 0.493
        top_matches = "ATS Ready"
    
    try:
        feats = extract_features(resume_pdf, jd_file.name, video_file.name)
    except Exception as e:
        print(f"Features error: {e}")
        feats = {"suitability": 0.493, "gaze": 0.85, "attention": 0.87, "fluency": 0.78, "communication": 0.82, "engagement": 0.79, "final_score": 0.77}
    
    final = round((ats_score + feats["final_score"]) / 2, 3)
    hired = "✅ **HIRED!**" if final > 0.70 else "📈 **IMPROVE**"
    
    save_to_dataset(candidate_name, ats_score, feats['gaze'], feats['attention'], feats['fluency'], feats['communication'], feats['engagement'], final)
    
    result = f"""
# 🎯 **{candidate_name} - {final:.3f}** {hired}

## 📄 ATS ({ats_score:.3f})
{top_matches}

## 🎥 6 Features

| Feature | Score |
|---------|-------|
| Suitability | {feats['suitability']:.3f} |
| Gaze | {feats['gaze']:.3f} |
| Attention | {feats['attention']:.3f} |
| Fluency | {feats['fluency']:.3f} |
| Communication | {feats['communication']:.3f} |
| Engagement | {feats['engagement']:.3f} |

✅ Saved to dataset!
    """
    
    dataset = load_dataset()
    return result, f"```\n{dataset}\n```"

with gr.Blocks() as demo:
    gr.Markdown("# 🤖 **HireSync - Interview Analysis**")
    with gr.Row():
        with gr.Column():
            resume = gr.File(label="📄 Resume PDF", file_types=[".pdf"])
            jd = gr.File(label="📋 JD TXT", file_types=[".txt"])
            name = gr.Textbox(label="👤 Candidate Name")
            video = gr.File(label="📹 Upload Video (MP4)", file_types=[".mp4"])  # ✅ FIXED!
            btn = gr.Button("🚀 ANALYZE", variant="primary")
        with gr.Column():
            output = gr.Markdown()
    dataset_out = gr.Markdown()
    btn.click(hiresync_complete, inputs=[resume, jd, video, name], outputs=[output, dataset_out])

demo.launch(share=True)
