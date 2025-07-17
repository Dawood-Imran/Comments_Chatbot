from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import pandas as pd
from enhanced_chatbot import setup_enhanced_chatbot, analyze_feedback
import os
from typing import List, Dict, Any, Optional
import logging
import json
from datetime import datetime
import re
from pathlib import Path

# Global variable to store the RAG chain
enhanced_rag_chain = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FeedbackRequest(BaseModel):
    question: str

class FeedbackResponse(BaseModel):
    success: bool
    content: str
    error: Optional[str] = None

class HistoryResponse(BaseModel):
    success: bool
    history: List[Dict[str, Any]]
    error: Optional[str] = None

# Configurable paths
DATA_FILE_PATH = os.getenv("DATA_FILE_PATH", "./game_cleaned_data.xlsx")
HISTORY_FILE = os.getenv("HISTORY_FILE_PATH", "chat_history.json")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3002").split(",")

# Ensure history file directory exists
os.makedirs(os.path.dirname(HISTORY_FILE) or '.', exist_ok=True)

def sanitize_input(text: str) -> str:
    """Basic input sanitization to remove potentially harmful characters."""
    # Remove HTML tags and excessive whitespace
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def save_to_history(question: str, content: str, success: bool) -> bool:
    try:
        # Create history entry
        entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "content": content,
            "success": success
        }
        
        # Load existing history
        history = []
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
        
        # Add new entry
        history.append(entry)
        
        # Save updated history
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved history entry for question: {question[:50]}...")
        return True
    except Exception as e:
        logger.error(f"Error saving history: {str(e)}", exc_info=True)
        return False

def get_history() -> List[Dict[str, Any]]:
    try:
        if not os.path.exists(HISTORY_FILE):
            return []
        
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
            
        # Sort by timestamp, newest first
        history.sort(key=lambda x: x["timestamp"], reverse=True)
        return history
    except Exception as e:
        logger.error(f"Error reading history: {str(e)}", exc_info=True)
        return []

def is_general_question(question: str) -> tuple[bool, str | None]:
    """Check if the question is general and return a predefined response if applicable."""
    question = sanitize_input(question.lower().strip())
    if question in ["hi", "hello", "hey"]:
        return True, "Hello! How can I assist you today?"
    elif "what can you do" in question or "capabilities" in question:
        return True, "I can analyze game feedback or answer general questions. Ask about the game experience or anything else!"
    elif "who are you" in question:
        return True, "I'm Analysis, your AI assistant for analyzing game feedback and answering questions!"
    return False, None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global enhanced_rag_chain
    try:
        # Validate data file existence
        if not os.path.exists(DATA_FILE_PATH):
            logger.error(f"Data file not found: {DATA_FILE_PATH}")
            raise FileNotFoundError(f"Data file not found: {DATA_FILE_PATH}")
        
        # Load DataFrame
        logger.info(f"Loading data from {DATA_FILE_PATH}")
        df = pd.read_excel(DATA_FILE_PATH)
        if df.empty:
            logger.error("DataFrame is empty")
            raise ValueError("DataFrame is empty")
        
        enhanced_rag_chain = setup_enhanced_chatbot(df)
        logger.info("Chatbot setup completed successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}", exc_info=True)
        raise
    yield
    logger.info("Shutting down application")

app = FastAPI(
    title="Feedback Analysis API",
    description="A simple FastAPI backend for analyzing game feedback and handling general questions",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware with restricted origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for query
class Query(BaseModel):
    question: str

@app.get("/")
async def root():
    return {"message": "Welcome to the Feedback Analysis API!"}

@app.post("/analyze", response_model=FeedbackResponse)
async def analyze_feedback_endpoint(request: FeedbackRequest):
    question = sanitize_input(request.question)
    logger.info(f"Received request with question: {question[:50]}...")
    
    # Check if RAG chain is loaded
    if enhanced_rag_chain is None:
        logger.error("RAG chain not initialized")
        return FeedbackResponse(
            success=False,
            content="# System Error\n\nThe analysis system is not properly initialized.",
            error="RAG chain not loaded"
        )
    
    try:
        # Validate input
        if not question:
            logger.warning("Empty question provided")
            return FeedbackResponse(
                success=False,
                content="# Invalid Input\n\nPlease provide a valid question.",
                error="Empty question provided"
            )
        
        # Check for general questions
        is_general, response = is_general_question(question)
        print(response)
        if is_general:
            logger.info("General question detected, returning predefined response")
            save_to_history(question, response, True)
            return FeedbackResponse(
                success=True,
                content=response,
                error=None
            )
        
        # Call the analyze_feedback function
        logger.info("Starting analysis...")
        result = analyze_feedback(question, enhanced_rag_chain)
        logger.info("Analysis completed successfully")
        
        # Ensure result has the expected structure
        if not isinstance(result, dict) or "success" not in result:
            logger.error(f"Invalid result format: {result}")
            return FeedbackResponse(
                success=False,
                content="# Analysis Error\n\nReceived invalid response format from analysis engine.",
                error="Invalid result format"
            )
        
        # Create the response
        content = result.get("content", "No content available")
        success = result["success"]
        error = result.get("error")
        
        # Save to history
        save_to_history(
            question=question,
            content=content,
            success=success
        )
        
        # Return the response
        response = FeedbackResponse(
            success=success,
            content=content,
            error=error
        )
        
        logger.info("Returning successful response")
        return response
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return FeedbackResponse(
            success=False,
            content="# Analysis Error\n\nUnable to process your request at this time. Please try again later.",
            error=f"Internal server error: {str(e)}"
        )

@app.get("/history", response_model=HistoryResponse)
async def get_chat_history():
    logger.info("Fetching chat history")
    try:
        history_data = get_history()
        return HistoryResponse(
            success=True,
            history=history_data,
            error=None
        )
    except Exception as e:
        logger.error(f"Error retrieving history: {str(e)}", exc_info=True)
        return HistoryResponse(
            success=False,
            history=[],
            error=f"Failed to retrieve history: {str(e)}"
        )

@app.get("/health")
async def health_check():
    if enhanced_rag_chain is None:
        logger.warning("Health check: Chatbot not initialized")
        return {"status": "unhealthy", "detail": "Chatbot not initialized"}
    return {"status": "healthy"}