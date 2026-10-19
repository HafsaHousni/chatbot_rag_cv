# Chatbot RAG sur CV

Application Streamlit permettant de charger un CV PDF/DOCX, de découper son contenu en chunks, de créer des embeddings avec Sentence-Transformers, de rechercher les passages pertinents avec FAISS et de générer une réponse avec Groq.

## Structure

```text
chatbot_rag_cv/
├── app.py
├── math_server.py
├── MCP.ipynb
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Prérequis

- Python 3.10 ou version compatible avec les dépendances installées
- Une clé API Groq valide

## Installation sous Windows (CMD)

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Ouvre ensuite `.env` et remplace `replace_with_your_groq_api_key` par ta clé Groq réelle. Le fichier `.env` est ignoré par Git et ne doit jamais être envoyé sur GitHub.

Pour charger les variables du fichier `.env` automatiquement, installe et utilise `python-dotenv` (inclus dans `requirements.txt`). L'application charge ce fichier au démarrage.

## Lancer le chatbot

```cmd
streamlit run app.py
```

Charge un CV PDF ou DOCX depuis la barre latérale, puis pose des questions sur le CV.

## Serveur MCP mathématique (facultatif)

Le fichier `math_server.py` expose les outils `add` et `multiply` via MCP en transport stdio. Pour lancer l'inspecteur MCP, utilise :

```cmd
npx @modelcontextprotocol/inspector python math_server.py
```

## Sécurité

- Ne committe jamais `.env`, les clés API, les CV téléversés ou autres données privées.
- `.env.example` contient uniquement des valeurs fictives.
- Si une clé a été commitée ou partagée, révoque-la et crée une nouvelle clé.
