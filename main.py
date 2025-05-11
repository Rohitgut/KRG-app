from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from bs4 import BeautifulSoup
import requests, fitz
import spacy, networkx as nx
import matplotlib.pyplot as plt
import os

# Initialize FastAPI
app = FastAPI()

# Mount static directory for serving images
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates directory
templates = Jinja2Templates(directory="templates")

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/extract")
async def extract_content(
    request: Request,
    pdf: UploadFile = None,
    url: str = Form(None),
    text: str = Form(None)
):
    content = None

    # Priority order: PDF > URL > Text
    if pdf and pdf.filename:
        file_data = await pdf.read()
        if file_data:
            content = extract_pdf(file_data)
    elif url:
        content = extract_from_url(url)
    elif text:
        content = text.strip()

    if not content:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "graph": None,
            "error": "Please provide valid input."
        })

    # Build graph and save image
    G = build_graph(content)
    output_file = os.path.join("static", "graph.png")
    visualize_graph(G, output_file)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "graph": "/static/graph.png"  # Important: must be relative to app root
    })


# ========== Helper Functions ==========

def extract_pdf(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

def extract_from_url(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.text, "html.parser")
    return soup.get_text()

def build_graph(text):
    doc = nlp(text)
    G = nx.Graph()
    for sent in doc.sents:
        ents = [ent.text for ent in sent.ents]
        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                G.add_edge(ents[i], ents[j])
    return G

def visualize_graph(G, output_file):
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, k=0.5)
    nx.draw(G, pos, with_labels=True, node_size=5000, node_color="skyblue", font_size=10)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
