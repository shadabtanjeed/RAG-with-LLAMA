# PDF Question Answering with OLLAMA

Python code that allows you to ask questions about your pdf locally with the help of OLLAMA.

## Prequisites
* Local installation of OLLAMA3.2.

## Get started

* Clone the project:
  ```bash
  git clone https://github.com/shadabtanjeed/RAG-with-OLLAMA.git
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
* Add a simple user interface.
* Integration of LangChain for better context management and retrieval.
* Add support for more file types.