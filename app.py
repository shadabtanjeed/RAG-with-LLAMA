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
import gradio as gr

model = "llama3.2"
messages = []
# Roles
USER = "user"
ASSISTANT = "assistant"

# Global variables for tokenizers, encoders, and index
context_tokenizer = None
context_encoder = None
question_tokenizer = None
question_encoder = None
index = None
paragraphs = None


def add_history(content, role):
    messages.append({"role": role, "content": content})


def chat(message):
    add_history(message, USER)

    prompt = """
    Context:
    {message}

    Use the context above to answer the given questions. If you do not know any answer, please respond with "I do not know". Do not make up any information.
    """
    prompt = prompt.format(message=message)

    response = ollama_chat(
        model=model, messages=[{"role": "user", "content": prompt}], stream=False
    )
    complete_message = ""
    for line in response:
        print("Response line:", line)  # Debugging print statement
        if isinstance(line, tuple) and line[0] == "message":
            message_content = line[1].content
            complete_message += message_content
    add_history(complete_message, ASSISTANT)
    return complete_message


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


def search_relevant_contexts(
    question, question_tokenizer, question_encoder, index, k=5
):
    question_inputs = question_tokenizer(question, return_tensors="pt")
    question_embedding = (
        question_encoder(**question_inputs).pooler_output.detach().numpy()
    )
    D, I = index.search(question_embedding, k)
    return D, I


def generate_answer_with_ollama(question, relevant_contexts):
    context_text = " ".join(relevant_contexts)
    prompt = f"Context: {context_text}\n\nQuestion: {question}\nAnswer:"
    response = chat(prompt)
    return response


def initialize_components(pdf_path):
    global context_tokenizer, context_encoder, question_tokenizer, question_encoder, index, paragraphs

    # Extract and split text
    all_text = extract_text_from_pdf(pdf_path)
    paragraphs = split_text_into_paragraphs(all_text)

    # Initialize tokenizer, encoder, and FAISS index
    model_name = "facebook/dpr-ctx_encoder-single-nq-base"
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_name)
    context_encoder = DPRContextEncoder.from_pretrained(model_name)
    d = 768  # Dimension of the embeddings
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


def answer_question_from_pdf(
    question,
    context_tokenizer,
    context_encoder,
    question_tokenizer,
    question_encoder,
    index,
    paragraphs,
):

    D, I = search_relevant_contexts(
        question, question_tokenizer, question_encoder, index, k=20
    )
    relevant_contexts = [paragraphs[i] for i in I[0]]

    answer = generate_answer_with_ollama(question, relevant_contexts)
    return answer


def interactive_question_answering(question, pdf_path):
    global context_tokenizer, context_encoder, question_tokenizer, question_encoder, index, paragraphs

    if (
        context_tokenizer is None
        or context_encoder is None
        or question_tokenizer is None
        or question_encoder is None
        or index is None
        or paragraphs is None
    ):
        initialize_components(pdf_path)

    answer = answer_question_from_pdf(
        question,
        context_tokenizer,
        context_encoder,
        question_tokenizer,
        question_encoder,
        index,
        paragraphs,
    )
    return answer


pdf_path = "random_story.pdf"


def chat_interface(message, history):
    # If history is None, initialize it
    if history is None:
        history = []

    # Generate response using your existing logic
    response = interactive_question_answering(message, pdf_path)

    # Append the message and response to history
    history.append((message, response))

    # Return both the updated history and an empty string for the textbox
    return history, ""


# Create Gradio Blocks interface
def create_chat_interface():
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(
            height=500,  # Adjust height as needed
            bubble_full_width=False,  # Makes chat bubbles more modern
            layout="bubble",  # Gives a more Claude/ChatGPT-like layout
        )
        msg = gr.Textbox(
            label="Ask a question about the PDF",
            placeholder="Type your question here...",
        )
        submit_btn = gr.Button("Send")

        # Define the interaction
        msg.submit(chat_interface, [msg, chatbot], [chatbot, msg])
        submit_btn.click(chat_interface, [msg, chatbot], [chatbot, msg])

    return demo


# Launch the interface
demo = create_chat_interface()
demo.launch()
