"""
TP Projet — Chatbot RAG sur CV
Filière : 4IASDR
Encadrante : Pr. Idrissi Zouggari Nadia
"""

import streamlit as st
from dotenv import load_dotenv
import os
import re
import pickle
import tempfile
from pathlib import Path
# ─── Chargement des bibliothèques ───────────────────────────────────────────
import fitz  # PyMuPDF pour PDF
import docx  # python-docx pour DOCX
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
load_dotenv()

# ─── Configuration ───────────────────────────────────────────────────────────


EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 4
CHUNK_SIZE = 200
OVERLAP = 40

# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE RAG — Fonctions modulaires
# ═══════════════════════════════════════════════════════════════════════════════

def load_cv(path: str) -> str:
    """
    Étape 1 — Chargement du CV.
    Lit un fichier PDF ou DOCX et retourne le texte brut extrait.
    """
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        doc = fitz.open(path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
    elif ext in (".docx", ".doc"):
        document = docx.Document(path)
        text = "\n".join(para.text for para in document.paragraphs)
    else:
        raise ValueError(f"Format non supporté : {ext}. Utilisez PDF ou DOCX.")
    return text


def clean_text(text: str) -> str:
    """
    Étape 2 — Prétraitement du texte.
    Nettoie les espaces multiples, sauts de ligne superflus et caractères parasites.
    """
    text = re.sub(r'\r\n|\r', '\n', text)           # normaliser les fins de ligne
    text = re.sub(r'\n{3,}', '\n\n', text)           # max 2 sauts de ligne consécutifs
    text = re.sub(r'[ \t]{2,}', ' ', text)           # espaces multiples → un seul
    text = re.sub(r'[^\S\n]+', ' ', text)            # tabulations → espace
    text = re.sub(r'[•·●▪▸◦‒–—−]+', '-', text)     # puces et tirets spéciaux
    text = text.strip()
    return text


def chunk_document(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """
    Étape 3 — Chunking.
    Découpe le texte en segments chevauchants pour améliorer la récupération.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def build_embeddings(chunks: list[str], model: SentenceTransformer) -> np.ndarray:
    """
    Étape 4 — Embedding.
    Transforme les chunks en vecteurs numériques avec Sentence-Transformers.
    """
    embeddings = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.astype("float32")


def create_vector_store(chunks: list[str], embeddings: np.ndarray) -> dict:
    """
    Étape 5 — Indexation.
    Stocke les vecteurs dans un index FAISS (recherche approximative rapide).
    Retourne un dictionnaire contenant l'index et les chunks associés.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    faiss.normalize_L2(embeddings)
    index_ip = faiss.IndexFlatIP(dim)   # produit interne (cosine après normalisation)
    index_ip.add(embeddings)
    return {"index": index_ip, "chunks": chunks}


def retrieve_relevant_chunks(question: str, store: dict, embed_model: SentenceTransformer, k: int = TOP_K) -> list[str]:
    """
    Étape 6 — Recherche.
    Récupère les k chunks les plus pertinents par rapport à la question.
    """
    q_vec = embed_model.encode([question], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_vec)
    scores, indices = store["index"].search(q_vec, k)
    results = [store["chunks"][i] for i in indices[0] if i < len(store["chunks"])]
    return results, scores[0]


def build_prompt(question: str, context: str) -> str:
    """
    Étape 7 — Construction du prompt.
    Formate le prompt pour forcer le modèle à répondre uniquement depuis le CV.
    """
    prompt = f"""Tu es un assistant spécialisé dans l'analyse de CV. 
Tu dois répondre UNIQUEMENT en te basant sur le contenu du CV fourni ci-dessous.
Si la question ne concerne pas le CV ou si l'information n'y est pas présente, 
réponds exactement : "Je ne peux répondre qu'aux questions relatives au CV."

--- CONTENU DU CV ---
{context}
--- FIN DU CV ---

Question : {question}

Réponse (en français, basée uniquement sur le CV) :"""
    return prompt




def generate_answer(prompt: str) -> str:
    try:


        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return "Erreur de configuration : définissez GROQ_API_KEY dans votre fichier .env local."
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Erreur lors de la génération : {str(e)}"

def answer_cv_only(question: str, store: dict, embed_model: SentenceTransformer) -> str:
    """
    Étape 9 — Sécurité métier.
    Vérifie si la question est liée au CV avant de générer une réponse.
    Utilise un score de similarité minimal pour filtrer les questions hors sujet.
    """
    SIMILARITY_THRESHOLD = 0.25  # seuil cosine (ajustable)

    chunks, scores = retrieve_relevant_chunks(question, store, embed_model, k=1)

    if not chunks or scores[0] < SIMILARITY_THRESHOLD:
        return "❌ Je ne peux répondre qu'aux questions relatives au CV chargé."

    # Récupérer les top-k chunks pour le contexte
    top_chunks, _ = retrieve_relevant_chunks(question, store, embed_model, k=TOP_K)
    context = "\n\n".join(top_chunks)
    prompt = build_prompt(question, context)
    return generate_answer(prompt)


def reload_vector_store(file_obj, embed_model: SentenceTransformer) -> tuple[dict, str, list[str]]:
    """
    Étape 10 — Chargement dynamique.
    Sauvegarde le fichier uploadé, reconstruit entièrement la base vectorielle.
    Retourne (store, nom_fichier, chunks).
    """
    suffix = Path(file_obj.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_obj.read())
        tmp_path = tmp.name

    raw_text = load_cv(tmp_path)
    clean = clean_text(raw_text)
    chunks = chunk_document(clean, CHUNK_SIZE, OVERLAP)
    embeddings = build_embeddings(chunks, embed_model)
    store = create_vector_store(chunks, embeddings)

    os.unlink(tmp_path)
    return store, file_obj.name, chunks


# ═══════════════════════════════════════════════════════════════════════════════
# INTERFACE STREAMLIT
# ═══════════════════════════════════════════════════════════════════════════════

def run_streamlit_app():
    """
    Étape 11 — Interface Streamlit.
    Lance l'application de dialogue utilisateur avec le chatbot RAG.
    """
    # ── Config page ──────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="Chatbot RAG — CV",
        page_icon="🤖",
        layout="centered",
    )

    # ── CSS personnalisé ─────────────────────────────────────────────────────
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@300;400;600&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .main-title {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 2rem;
            font-weight: 600;
            color: #0f172a;
            letter-spacing: -1px;
            margin-bottom: 0;
        }
        .subtitle {
            font-size: 0.9rem;
            color: #64748b;
            margin-top: 0;
            margin-bottom: 1.5rem;
        }
        .cv-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: #f0fdf4;
            border: 1.5px solid #86efac;
            border-radius: 8px;
            padding: 6px 14px;
            font-size: 0.85rem;
            color: #166534;
            font-weight: 500;
            margin-bottom: 1rem;
        }
        .no-cv-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: #fff7ed;
            border: 1.5px solid #fdba74;
            border-radius: 8px;
            padding: 6px 14px;
            font-size: 0.85rem;
            color: #9a3412;
            font-weight: 500;
            margin-bottom: 1rem;
        }
        .stat-box {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 10px 16px;
            font-size: 0.82rem;
            color: #475569;
        }
        [data-testid="stChatMessage"] {
            border-radius: 12px;
            margin-bottom: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

    # ── Titre ────────────────────────────────────────────────────────────────
    st.markdown('<p class="main-title">🤖 Chatbot RAG — CV</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">4IASDR · Pr. Idrissi Zouggari Nadia · Posez vos questions sur votre CV</p>', unsafe_allow_html=True)
    st.divider()

    # ── Chargement du modèle d'embedding (mis en cache) ──────────────────────
    @st.cache_resource
    def get_embed_model():
        return SentenceTransformer(EMBED_MODEL_NAME)

    embed_model = get_embed_model()

    # ── État de session ───────────────────────────────────────────────────────
    if "store" not in st.session_state:
        st.session_state.store = None
    if "cv_name" not in st.session_state:
        st.session_state.cv_name = None
    if "history" not in st.session_state:
        st.session_state.history = []
    if "chunks_count" not in st.session_state:
        st.session_state.chunks_count = 0

    # ── Sidebar — Upload ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 📄 Chargement du CV")
        uploaded = st.file_uploader(
            "Glissez votre CV ici",
            type=["pdf", "docx"],
            help="Formats acceptés : PDF, DOCX"
        )

        if uploaded:
            # Nouveau CV → reconstruire la base vectorielle
            if uploaded.name != st.session_state.cv_name:
                with st.spinner("⚙️ Construction de la base vectorielle…"):
                    store, name, chunks = reload_vector_store(uploaded, embed_model)
                    st.session_state.store = store
                    st.session_state.cv_name = name
                    st.session_state.history = []   # reset historique
                    st.session_state.chunks_count = len(chunks)
                st.success("✅ Base vectorielle chargée !")

        st.markdown("---")

        # Badge CV chargé
        if st.session_state.cv_name:
            st.markdown(
                f'<div class="cv-badge">📋 {st.session_state.cv_name}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="stat-box">🔢 <b>{st.session_state.chunks_count}</b> chunks indexés<br>'
                f'🧠 Modèle : <code>{EMBED_MODEL_NAME}</code><br>'
                f'🔍 Top-K : <b>{TOP_K}</b></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="no-cv-badge">⚠️ Aucun CV chargé</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        if st.button("🗑️ Vider l'historique"):
            st.session_state.history = []
            st.rerun()

    # ── Zone de chat ──────────────────────────────────────────────────────────
    if not st.session_state.store:
        st.info("👆 Commencez par charger un CV dans la barre latérale.")
    else:
        # Afficher l'historique
        for msg in st.session_state.history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Saisie utilisateur
        question = st.chat_input("Posez une question sur le CV…")
        if question:
            # Ajouter la question à l'historique
            st.session_state.history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            # Générer la réponse
            with st.chat_message("assistant"):
                with st.spinner("Analyse en cours…"):
                    answer = answer_cv_only(
                        question,
                        st.session_state.store,
                        embed_model
                    )
                st.markdown(answer)

            st.session_state.history.append({"role": "assistant", "content": answer})


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_streamlit_app()
