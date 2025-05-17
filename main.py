from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from bs4 import BeautifulSoup
import requests, fitz
import spacy, networkx as nx
import os
import matplotlib.pyplot as plt

app = FastAPI()


app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")
nlp = spacy.load("en_core_web_sm")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/extract")
async def extract_content(request: Request, pdf: UploadFile = None, url: str = Form(None), text: str = Form(None)):
    content = None

    if pdf and pdf.filename:
        file_data = await pdf.read()
        if file_data:
            content = extract_pdf(file_data)
    elif url:
        content = extract_from_url(url)
    elif text:
        content = text

    if not content:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "graph": None,
            "error": "Please provide valid input."
        })

    G = build_svo_adj_graph(content)
    output_file = os.path.join("static", "graph.png")
    visualize_graph(G, output_file)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "graph": output_file
    })

def extract_pdf(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

def extract_from_url(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.text, "html.parser")
    return soup.get_text()

def build_svo_adj_graph(text):
    doc = nlp(text)
    G = nx.DiGraph()

    for sent in doc.sents:
        for token in sent:
            # Active: Subject-Verb-Object
            if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                obj = [child for child in token.head.children if child.dep_ == "dobj"]
                if obj:
                    G.add_edge(token.text, obj[0].text, label=token.head.text)

            # Passive voice: Object-Verb-Subject (e.g., "The book was written by the author")
            if token.dep_ == "nsubjpass" and token.head.pos_ == "VERB":
                agent = [child for child in token.head.children if child.dep_ == "agent"]
                if agent:
                    agent_obj = [child for child in agent[0].children if child.dep_ == "pobj"]
                    if agent_obj:
                        G.add_edge(token.text, agent_obj[0].text, label=token.head.text)

            # Adjective modifier via copula
            if token.dep_ == "attr":
                subj = [child for child in token.head.children if child.dep_ == "nsubj"]
                if subj:
                    G.add_edge(subj[0].text, token.text, label=token.head.text)

            # Adjective directly modifying a noun
            if token.dep_ == "amod" and token.head.pos_ == "NOUN":
                G.add_edge(token.head.text, token.text, label="is")

    return G

def visualize_graph(G, output_file):
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.5)
    edge_labels = nx.get_edge_attributes(G, 'label')

    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', font_size=9)
    plt.savefig(output_file)
    plt.close()
