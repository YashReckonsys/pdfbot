import os
import re
import uuid

import gradio as gr
from django.http import HttpResponse
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

load_dotenv()

api_key = os.environ["OPENAI_API_KEY"]
embeddings = OpenAIEmbeddings(openai_api_key=api_key)


# file = open("pdfbot/botapp/RiskManagement.pdf", "rb")
def add_text(history, text: str):
    if not text:
        raise gr.Error("No query entered")
    history = history + [(text, "")]
    return history


key = str(uuid.uuid4())


class PDFChatApp:
    def __init__(self):
        self.OPENAI_API_KEY = api_key
        self.chat_history = {}
        self.chains = {}
        self.count = 0
        self.pdfsearch = {}
        # self.key=str(uuid.uuid4())
        # self.files = {}
        self.chunk_size = 1024
        self.chunk_overlap = 128
        self.chat_model = None
        self.chain = None
        self.file_id = key
        self.files = {}

        self.file = open("riskapp/botapp/RiskManagement.pdf", "rb")
        self.files[self.file_id] = self.file

    def process_file(self, file):
        loader = PyPDFLoader(file.name)
        documents = loader.load()
        pattern = r"/([^/]+)$"
        match = re.search(pattern, file.name)
        file_name = match.group(1)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        documents = text_splitter.split_documents(documents)
        return documents, file_name

    def build_chain(self, file):
        documents, file_name = self.process_file(file)
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)

        pdfsearch = Chroma.from_documents(documents, embeddings, collection_name=file_name)
        self.chain = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(temperature=0.0, openai_api_key=api_key),
            retriever=pdfsearch.as_retriever(search_kwargs={"k": 5}),
            chain_type="stuff",
            max_tokens_limit=4000,
            return_source_documents=True,
        )
        return self.chain

    # def add_file(self, file):
    #     # self.file_id = str(uuid.uuid4())
    #     self.files[self.file_id] = file
    #     return self.file_id

    def get_file_id(self, file):
        return self.file_id

    def get_response(self, history, query):
        if not self.file:
            raise gr.Error(message="Upload a PDF")
        self.chain = self.chains[self.file_id]
        result = self.chain(
            {"question": query, "chat_history": self.chat_history[self.file_id]}, return_only_outputs=True
        )
        # print("FILE ID ", self.file_id)
        self.chat_history[self.file_id] += [(query, result["answer"])]
        if len(self.chat_history[self.file_id]) > 3:
            self.chat_history[self.file_id] = self.chat_history[self.file_id][1:]
        for char in result["answer"]:
            history[-1][-1] += char
            yield history, ""

    def start_bot(self):
        if not self.file:
            return None, []
        # self.file_id = self.add_file(self.file)
        self.chains[self.file_id] = self.build_chain(self.file)
        self.chat_history[self.file_id] = []
        return []


with gr.Blocks() as demo:
    app = PDFChatApp()
    with gr.Column():
        with gr.Row():
            chatbot = gr.Chatbot(value=[], elem_id="chatbot").style(height=650)
    with gr.Row():
        with gr.Column(scale=0.50):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter query and click on submit",
            ).style(container=False)
        with gr.Column(scale=0.25):
            submit_btn = gr.Button("Submit")
        with gr.Column(scale=0.25):
            btn = gr.Button("Start Bot")

    btn.click(
        fn=app.start_bot,
        inputs=[],
        outputs=[chatbot],
    )

    submit_btn.click(
        fn=add_text,
        inputs=[chatbot, txt],
        outputs=[
            chatbot,
        ],
        queue=True,
    ).success(
        fn=app.get_response,
        inputs=[chatbot, txt],
        outputs=[chatbot, txt],
    )

demo.queue(api_open=False, concurrency_count=10)


def gradio_chatbot_view(request):
    # Create a new instance of the PDFChatApp for each session
    return HttpResponse(demo.launch(inbrowser=True, max_threads=2), content_type="text/html")
