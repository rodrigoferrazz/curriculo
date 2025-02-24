import os
import json
import re
from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer
import PyPDF2
import numpy as np

app = Flask(__name__)

# Configurações
PDF_DIR = "resumes"                # Pasta onde estão os PDFs
CACHE_FILE = "embeddings_cache.json"
MODEL_NAME = "paraphrase-MiniLM-L6-v2"  # Modelo para gerar embeddings

# Carrega o modelo localmente
model = SentenceTransformer(MODEL_NAME)

# Dicionários para armazenar os textos pré-processados e os embeddings
resumes = {}     # filename -> texto pré-processado
embeddings = {}  # filename -> embedding (lista de floats)

def preprocess_text(text):
    """Converte o texto para minúsculas, remove pontuação e espaços extras."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_pdf(file_path):
    """Extrai o texto de um PDF utilizando PyPDF2."""
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += " " + page_text
    return text.strip()

def get_embedding(text):
    """Gera o embedding do texto utilizando o modelo do Sentence Transformers."""
    if not text:
        # Retorna um vetor nulo se não houver texto (dimensão 384 para o MiniLM)
        return [0.0] * 384
    return model.encode(text).tolist()

def load_cache():
    """Carrega o cache de embeddings, se existir."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    """Salva o cache de embeddings em um arquivo JSON."""
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f)

def load_resumes():
    cache = load_cache()
    for filename in os.listdir(PDF_DIR):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(PDF_DIR, filename)
            raw_text = extract_text_from_pdf(path)
            text = preprocess_text(raw_text)
            resumes[filename] = text
            print(f"Conteúdo de {filename}: {text}")  # Debug: imprime o texto extraído
            if filename in cache:
                embeddings[filename] = cache[filename]
            else:
                emb = get_embedding(text)
                embeddings[filename] = emb
                cache[filename] = emb
    save_cache(cache)
    print("Currículos carregados:", list(resumes.keys()))



def cosine_similarity(a, b):
    """Calcula a similaridade cosseno entre dois vetores."""
    a = np.array(a)
    b = np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form.get("query")
        query_processed = preprocess_text(query)
        query_embedding = get_embedding(query_processed)
        
        results = []
        for filename, emb in embeddings.items():
            cos_sim = cosine_similarity(query_embedding, emb)
            contains = query_processed in resumes[filename]
            print(f"Arquivo: {filename} - 'odair' está presente? {contains}")
            if contains:
                if len(query_processed.split()) <= 2:
                    score = 1.0
                else:
                    score = cos_sim + 0.1
            else:
                score = cos_sim
            results.append((filename, score))
            print(f"Arquivo: {filename} - Score: {score}")

        
        results.sort(key=lambda x: x[1], reverse=True)
        threshold = 0.1
        matched = [filename for filename, score in results if score >= threshold]
        
        print("Resultados encontrados:", matched)
        return render_template("results.html", query=query, results=matched, all_results=results)
    return render_template("index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run()