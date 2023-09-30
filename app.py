import base64
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from PyPDF2 import PdfReader
import os
import tempfile
from flask import Flask, render_template, request

app = Flask(__name__)

# Model and Tokenizer
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)

# File loading and preprocessing
def file_processing(file_content):
    if len(file_content) == 0:
        return "The uploaded PDF file is empty."

    pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf_file.write(file_content)
    pdf_file.close()

    pdf = PdfReader(pdf_file.name)
    pages = []
    for page_num in range(len(pdf.pages)):
        pages.append(pdf.pages[page_num].extract_text())

    os.remove(pdf_file.name)  # Remove the temporary PDF file

    final_texts = "\n".join(pages)  # Join pages with a newline character
    return final_texts

# LLM PIPELINE
def llm_pipeline(file_content):
    input_text = file_processing(file_content)

    if "The uploaded PDF file is empty." in input_text:
        return "The uploaded PDF file is empty."

    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True, padding=True).to(device)

    # Generate the summary
    summary_ids = base_model.generate(input_ids, max_length=150, min_length=50, num_beams=4, length_penalty=2.0, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Post-process the summary
    summary = summary.replace("\n", " ")  # Remove line breaks
    summary = " ".join(summary.split())  # Remove extra whitespace

    return summary

@app.route('/', methods=['GET', 'POST'])
def index():
    pdf_display = None  # Initialize the PDF display as None
    summary = None  # Initialize the summary as None

    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file:
            try:
                file_content = uploaded_file.read()
                summary = llm_pipeline(file_content)

                # Display the PDF in the first column
                pdf_base64 = base64.b64encode(file_content).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="100%" height="600" type="application/pdf"></iframe>'
            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                return render_template('index.html', error=error_message, pdf_display=None, summary=None)

    return render_template('index.html', pdf_display=pdf_display, summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
