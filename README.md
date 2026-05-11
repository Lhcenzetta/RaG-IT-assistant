# RaG-IT-assistant

An intelligent internal IT assistant capable of reliably answering IT technicians' questions based on IT support documentation (procedures, incidents, FAQs) using Retrieval-Augmented Generation (RAG).

## 🚀 Key Features

- **Advanced RAG Pipeline**: Uses LangChain to chunk and retrieve relevant contexts from support PDFs.
- **State-of-the-art LLM**: Integrates Google's `gemini-2.5-flash` for high-quality, fast answer generation.
- **Vector Search**: Leverages `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) and ChromaDB for semantic document retrieval.
- **RESTful API**: Built with **FastAPI**, featuring JWT-based authentication for secure access.
- **Relational Database**: Uses **PostgreSQL** (via SQLAlchemy) to store users and query history, including latency and query clustering.
- **Experiment Tracking**: Integrated with **MLflow** to track query performance, prompt structures, and generation/retrieval latency.
- **Containerized**: Fully Dockerized backend and database configuration using `docker-compose`.

## 📂 Project Structure

```text
RaG-IT-assistant/
├── backend/
│   └── app/
│       ├── api/          # FastAPI routers (Authentication, Query endpoints)
│       ├── db/           # SQLAlchemy models, schemas, and session
│       └── main.py       # FastAPI application entry point
├── RaG/
│   └── Save_chroma_data.py # Script for PDF processing, chunking, and ChromaDB vectorization
├── docker-compose.yml    # Docker Compose configuration for PostgreSQL and FastAPI app
├── dockerfile            # Dockerfile for deploying the FastAPI backend
├── requirements.txt      # Python dependencies
├── test.py             
└── README.md
```

## ⚙️ Prerequisites

- Python 3.9+
- Docker and Docker Compose (optional, for easy deployment)
- API Keys: 
  - Google Gemini API Key (`GOOGLE_API_KEY`)
- MLflow environment setup

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd RaG-IT-assistant
   ```

2. **Environment Variables Configuration:**
   Create a `.env` file in the root directory and configure the following:
   ```env
   # Database credentials
   user=your_db_user
   password=your_db_password
   database=your_db_name

   # LLM & Storage Paths
   GOOGLE_API_KEY=your_gemini_api_key
   path_save=./db_chroma           # Path to save ChromaDB vectors
   pdf_path=./Document/support.pdf # Path to the IT Support PDF
   
   # FastAPI Auth
   SECRET_KEY=your_jwt_secret_key
   
   # Clustering Model
   model_path=./path_to_joblib_model_for_query_clustering
   ```

3. **Install Dependencies (Local):**
   ```bash
   python -m venv venv
   source venv/bin/activate # On Windows: venv\\Scripts\\activate
   pip install -r requirements.txt
   ```

## 🐳 Running with Docker

The easiest way to run the PostgreSQL database and FastAPI backend together:

```bash
docker-compose up --build
```
The API will be available at `http://localhost:8000`.

## 📚 Usage

### 1. Vectorizing the PDF
Place your IT support PDF file in the specified `pdf_path` directory. Then run the ingestion script to create embeddings:
```bash
python RaG/Save_chroma_data.py
```

### 2. Running FastAPI Locally (without Docker)
Start the FastAPI server using Uvicorn:
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. API Endpoints
- **POST `/Signup`**: Register a new user.
- **POST `/login`**: Authenticate and receive a JWT Bearer token.
- **POST `/query`**: Ask a question (Requires JWT token). Tracks metrics (latency, cluster number) and logs to the PostgreSQL DB.
- **DELETE `/delete_user/{user_id}`**: Delete a user (Requires JWT token).

### 4. Tracking with MLflow
To review experiments (prompt tracking, chunks retrieved, latency metrics), you can launch the MLflow UI:
```bash
mlflow ui
```

## 🔭 Future Improvements
- [ ] Add support for multiple document formats (Word, Excel, HTML).
- [ ] Build a frontend UI (Streamlit or React).
- [ ] Implement query caching for faster retrieval.
- [ ] Real-time slack/teams bot integration.
