import gradio as gr
import fitz
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
)
import numpy as np
import faiss
from ollama import chat as ollama_chat

# Global variables
model = "llama3.2"
context_tokenizer = None
context_encoder = None
question_tokenizer = None
question_encoder = None
index = None
paragraphs = None
uploaded_pdf_path = None


def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    all_text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text = page.get_text()
        all_text += text
    return all_text


def split_text_into_paragraphs(all_text):
    paragraphs = all_text.split("\n")
    paragraphs = [para.strip() for para in paragraphs if len(para.strip()) > 0]
    return paragraphs


def initialize_rag_components(pdf_path):
    global context_tokenizer, context_encoder, question_tokenizer, question_encoder, index, paragraphs

    # Extract and split text
    all_text = extract_text_from_pdf(pdf_path)
    paragraphs = split_text_into_paragraphs(all_text)

    # Initialize tokenizer, encoder, and FAISS index
    model_name = "facebook/dpr-ctx_encoder-single-nq-base"
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_name)
    context_encoder = DPRContextEncoder.from_pretrained(model_name)
    d = 768
    index = faiss.IndexFlatL2(d)

    # Encode paragraphs
    paragraph_embeddings = []
    for para in paragraphs:
        para_inputs = context_tokenizer(para, return_tensors="pt")
        para_embedding = context_encoder(**para_inputs).pooler_output.detach().numpy()
        paragraph_embeddings.append(para_embedding)
    paragraph_embeddings = np.vstack(paragraph_embeddings)
    index.add(paragraph_embeddings)

    # Initialize question tokenizer and encoder
    question_model_name = "facebook/dpr-question_encoder-single-nq-base"
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
        question_model_name
    )
    question_encoder = DPRQuestionEncoder.from_pretrained(question_model_name)


def search_relevant_contexts(
    question, question_tokenizer, question_encoder, index, k=5
):
    question_inputs = question_tokenizer(question, return_tensors="pt")
    question_embedding = (
        question_encoder(**question_inputs).pooler_output.detach().numpy()
    )
    D, I = index.search(question_embedding, k)
    return D, I


def generate_answer_with_ollama(question, relevant_contexts=None):
    if relevant_contexts:
        context_text = " ".join(relevant_contexts)
        prompt = f"Context: {context_text}\n\nQuestion: {question}\nAnswer:"
    else:
        prompt = question

    response = ollama_chat(
        model=model, messages=[{"role": "user", "content": prompt}], stream=False
    )

    complete_message = ""
    for line in response:
        if isinstance(line, tuple) and line[0] == "message":
            complete_message += line[1].content

    return complete_message


def process_query(message, history, pdf_path=None):
    global context_tokenizer, context_encoder, question_tokenizer, question_encoder, index, paragraphs

    # If PDF is uploaded and RAG components are not initialized
    if pdf_path and (context_tokenizer is None or index is None):
        initialize_rag_components(pdf_path)

    # If PDF is uploaded and RAG components are initialized
    if pdf_path and context_tokenizer and index is not None:
        D, I = search_relevant_contexts(
            message, question_tokenizer, question_encoder, index, k=5
        )
        relevant_contexts = [paragraphs[i] for i in I[0]]
        response = generate_answer_with_ollama(message, relevant_contexts)
    else:
        response = generate_answer_with_ollama(message)

    return response


def chat_interface(message, history, pdf_path):

    if history is None:
        history = []

    response = process_query(message, history, pdf_path)

    history.append((message, response))

    return history, ""


def create_chat_interface():
    with gr.Blocks() as demo:
        # PDF File Upload
        pdf_upload = gr.File(
            file_types=[".pdf"], label="Upload PDF (Optional)", type="filepath"
        )

        # Chatbot Component
        chatbot = gr.Chatbot(height=500, bubble_full_width=False, layout="bubble")

        # Message Input
        msg = gr.Textbox(
            label="Ask a question", placeholder="Type your message here..."
        )

        # Submit Button
        submit_btn = gr.Button("Send")

        # Interaction Logic
        pdf_upload.upload(
            fn=lambda filepath: filepath, inputs=pdf_upload, outputs=pdf_upload
        )

        msg.submit(
            fn=chat_interface, inputs=[msg, chatbot, pdf_upload], outputs=[chatbot, msg]
        )

        submit_btn.click(
            fn=chat_interface, inputs=[msg, chatbot, pdf_upload], outputs=[chatbot, msg]
        )

    return demo


# Launch the interface
demo = create_chat_interface()
demo.launch()
