# PDF Question Answering with OLLAMA

Python application that allows you to ask questions about your pdf locally with the help of LLAMA 3.2 using OLLAMA.

## About

This project uses Gradio to create an interactive chat interface that allows users to ask questions about the content of a PDF. The system leverages the following technologies:

- **Gradio**: For creating the interactive chat interface.
- **PyMuPDF**: For extracting text from PDF files.
- **Transformers**: For tokenizing and encoding text.
- **FAISS**: For efficient similarity search.
- **OLLAMA (Lllama 3.2)**: For generating answers based on the provided context.

## Screenshot

![OLLAMA_Chat](https://github.com/user-attachments/assets/d839569b-a7d2-4357-81bc-53a0c95b5183)

## Demo
https://github.com/user-attachments/assets/c01855ed-3d0e-45dc-962c-0f4fe1470413

## Prequisites
* Python 3.11+
* Local installation of OLLAMA3.2.

## Get started

* Clone the project:
  ```bash
  git clone https://github.com/shadabtanjeed/RAG-with-LLAMA.git
  cd "RAG WITH OLLAMA"
  ```

* Create a virtual environment:
  ```bash
  python -m venv venv
  source venv/bin/activate
  ```

* Install the required packages:
  ```bash
  pip install -r requirements.txt
  ```

* Add the directory to your desired pdf into the variable: `pdf_path`

* Start your local OLLAMA server.

* Run the python file
  ```bash
  python -m app.py
  ```

## Future Iterations
* Integration of LangChain for better context management and retrieval.
* Add support for more file types.
