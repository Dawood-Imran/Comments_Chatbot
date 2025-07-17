from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os
import logging
import re
from typing import Dict, Any, List
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)
os.environ["GOOGLE_API_KEY"] = "AIzaSyA_Gz68wyrflhVMr3EEfvyQefi4B3FYBMs"


def create_enhanced_documents(df):
    logger.info("Creating enhanced documents...")
    required_columns = ['Verbatim', 'Canvs Emotions', 'Open-Ended Question']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        raise ValueError(f"DataFrame missing required columns: {missing_columns}")
    
    documents = []
    for index, row in df.iterrows():
        verbatim = row['Verbatim']
        emotion = row.get('Canvs Emotions', 'Unknown')
        question = row.get('Open-Ended Question', 'No Question Provided')
        
        # Handle emotion type
        if isinstance(emotion, list):
            emotion_str = ",".join(str(e) for e in emotion)
        elif isinstance(emotion, str):
            emotion_str = emotion
        else:
            emotion_str = str(emotion)
            logger.warning(f"Unexpected emotion type at index {index}: {type(emotion)}")
        
        enhanced_content = f"[Emotion: {emotion_str}] {verbatim}"
        doc = Document(
            page_content=enhanced_content,
            metadata={
                'emotion': emotion_str,
                'original_verbatim': str(verbatim),
                'original_question': str(question),
                'row_index': int(index)
            }
        )
        documents.append(doc)
    
    logger.info(f"Created {len(documents)} documents")
    return documents

def format_context_with_emotion(docs):
    logger.info(f"Retrieved {len(docs)} documents")
    
    formatted_context = ""
    for i, doc in enumerate(docs, 1):
        emotion = doc.metadata.get('emotion', 'Unknown')
        original_text = doc.metadata.get('original_verbatim', doc.page_content)
        question = doc.metadata.get('original_question', 'No question available')
        row_index = doc.metadata.get('row_index', 'Unknown')
        
        logger.debug(f"Document {i}: Row Index={row_index}, Emotion={emotion}, "
                    f"Question={question[:50]}..., Text={original_text[:100]}...")
        
        formatted_context += (
            f"Review {i}:\n"
            f"[Emotion: {emotion}]\n"
            f"[Original Question: {question}]\n"
            f"{original_text}\n\n"
        )
    
    return formatted_context

def sanitize_input(text: str) -> str:
    """Sanitize input for prompt template."""
    text = re.sub(r'[^\w\s.,!?]', '', text)
    return text.strip()

unified_template = """
You are an expert customer feedback analyst specializing in analyzing feedback from people who attended a game. Your task is to analyze the provided customer reviews about their game experience and answer the question thoroughly and comprehensively.

**ANALYSIS INSTRUCTIONS:**
1. Determine the primary focus of the question related to the game experience (e.g., satisfaction with the game, issues encountered, specific aspects like food, seating, atmosphere).
2. Identify key points (positive, negative, or neutral) related to the game experience based on the provided context.
3. Pay close attention to the emotion labels associated with each review in the context. They provide vital information about customer sentiment regarding their game experience.
4. For each key point, include:
    - A clear title using a markdown heading (e.g., `## Feedback on Food and Beverages` or `## Issues with Seating`).
    - A detailed description of the point, specifically relating it to the game experience.
    - Exact customer quotes that support this point. Use bullet points (`* ` or `- `) for the quotes.
    - Mention the emotions associated with the quotes or the overall point.
    - Indicate the frequency of this point (e.g., "mentioned in several reviews", "frequent comment", "one instance").
    - Assess the impact level ('High', 'Medium', or 'Low') on the customer's game experience.
5. Provide a brief executive summary using a markdown heading (e.g., `## Executive Summary of Game Feedback`).
6. Include a section with representative customer quotes using a markdown heading (e.g., `## Representative Game Experience Comments`) and explain why they are representative.
7. Include a breakdown of the emotions present in the relevant reviews using a markdown heading (e.g., `## Emotion Breakdown for Game Feedback`).

**IMPORTANT GUIDELINES:**
- Use ONLY the information provided in the context. If the context does not contain relevant information about the game experience, state that clearly.
- Include as many exact customer quotes as possible to support your points.
- Group similar feedback together under appropriate headings related to game experience aspects.
- Explain the significance of each finding and its implications for the overall game experience.
- Use markdown formatting (# for headings, * or - for bullets) as instructed.

Context:
{context}

Question:
{question}

Answer:
"""

unified_prompt = ChatPromptTemplate.from_template(unified_template)

def analyze_feedback(question: str, rag_chain) -> Dict[str, Any]:
    """
    Analyze customer feedback using the provided question
    
    Args:
        question: The analysis question
        rag_chain: The LangChain chain for analysis
    
    Returns:
        Dict containing success status, content and any error message
    """
    try:
        logger.info(f"Starting analysis for question: {question[:50]}...")
        
        # Handle empty questions
        if not question or not question.strip():
            logger.warning("Empty question provided")
            return {
                "success": False,
                "content": "# Invalid Input\n\nPlease provide a valid question.",
                "error": "Empty question"
            }
        
        # Sanitize input
        question = sanitize_input(question)
        
        # Get response from RAG chain
        logger.debug("Invoking RAG chain...")
        response = rag_chain.invoke(question)
        
        # Extract content from response
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = str(response)
            logger.warning(f"Response converted to string: {content[:100]}...")
        
        logger.info("Analysis completed successfully")
        return {
            "success": True,
            "content": content,
            "error": None
        }
        
    except Exception as e:
        error_message = f"""# Error in Analysis
Unable to process your request. Please try again.
Error details: {str(e)}"""
        logger.error(f"Error in analysis: {str(e)}", exc_info=True)
        return {
            "success": False,
            "content": error_message,
            "error": str(e)
        }

def setup_enhanced_chatbot(df):
    logger.info("Setting up the chatbot...")
    
    # Validate Google API key
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEY environment variable not set")
        raise ValueError("GOOGLE_API_KEY not set")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.1,
        max_tokens=2000,
    )
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    logger.info("Loaded embeddings model")
    
    # Configurable persist directory
    persist_directory = os.getenv("CHROMA_PERSIST_DIR", "./enhanced_chroma_db")
    collection_name = "enhanced_collection"
    
    try:
        # Ensure persist directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        if os.path.exists(persist_directory):
            logger.info("Loading existing vector store...")
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=persist_directory
            )
            logger.info("Vector store loaded successfully")
        else:
            logger.info("Creating new vector store...")
            documents = create_enhanced_documents(df)
            vectorstore = Chroma.from_documents(
                collection_name=collection_name,
                documents=documents,
                embedding=embeddings,
                persist_directory=persist_directory
            )
            logger.info("Vector store created successfully")
    except Exception as e:
        logger.error(f"Error setting up vector store: {str(e)}", exc_info=True)
        raise
    
    # Configurable MMR parameters
    k = int(os.getenv("RETRIEVER_K", 30))
    fetch_k = int(os.getenv("RETRIEVER_FETCH_K", 50))
    lambda_mult = float(os.getenv("RETRIEVER_LAMBDA_MULT", 0.5))
    
    logger.info(f"Setting up retriever with MMR: k={k}, fetch_k={fetch_k}, lambda_mult={lambda_mult}")
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult}
    )
    
    def retrieve_with_logging(question):
        logger.info(f"Retrieving documents for query: {question[:50]}...")
        docs = retriever.get_relevant_documents(question)
        return format_context_with_emotion(docs)
    
    # Create the RAG chain
    unified_rag_chain = (
        {"context": retrieve_with_logging,
         "question": RunnablePassthrough()}
        | unified_prompt
        | llm
    )
    
    logger.info("Chatbot setup complete")
    return unified_rag_chain