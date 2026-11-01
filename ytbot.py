# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.



# Gradio: Create the web interface for the YouTube bot
import gradio as gr

# Regular expressions: Extract the YouTube video ID
import re

# YouTube Transcript API: Retrieve video transcripts
from youtube_transcript_api import YouTubeTranscriptApi

# LangChain: Split transcripts into smaller chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Ollama: Connect to the Llama model
from langchain_ollama import ChatOllama

# Hugging Face embeddings: Convert transcript chunks into vectors
from langchain_community.embeddings import HuggingFaceEmbeddings

# FAISS: Store and search vector embeddings
from langchain_community.vectorstores import FAISS

# LangChain: Create prompt templates
from langchain.prompts import PromptTemplate

# LangChain: Create LLM chains
from langchain.chains import LLMChain

from urllib.parse import urlparse, parse_qs





# def get_video_id(url):
#     # Regex pattern to match YouTube video URLs
#     pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
#     match = re.search(pattern, url)
#     return match.group(1) if match else None





def get_video_id(url):
    if not isinstance(url, str) or not url.strip():
        return None

    url = url.strip()

    # Remove surrounding text if the user pasted a
    # YouTube app share message containing a URL.
    match = re.search(
        r"https?://[^\s<>]+",
        url
    )
    if match:
        url = match.group(0).rstrip(".,)")

    # Accept direct video IDs
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url

    # Add https:// if missing
    if url.startswith((
        "youtu.be/",
        "www.youtu.be/",
        "youtube.com/",
        "www.youtube.com/",
        "m.youtube.com/",
        "music.youtube.com/"
    )):
        url = "https://" + url

    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    path = parsed.path.strip("/")

    video_id = None

    # YouTube mobile share links
    if hostname in ("youtu.be", "www.youtu.be"):
        video_id = path.split("/")[0]

    # Other YouTube URLs
    elif hostname in (
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com"
    ):
        if path == "watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]

        elif path.startswith(("shorts/", "live/", "embed/", "v/")):
            parts = path.split("/")
            if len(parts) >= 2:
                video_id = parts[1]

    # Validate the extracted video ID
    if video_id and re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        return video_id

    return None






def get_transcript(url):
    # Extract YouTube video ID
    video_id = get_video_id(url)

    if not video_id:
        raise ValueError("Invalid YouTube URL or video ID.")

    # Initialize YouTube Transcript API
    ytt_api = YouTubeTranscriptApi()

    try:
        # Retrieve available transcripts
        transcripts = ytt_api.list(video_id)

        manual_transcript = None
        generated_transcript = None

        # Search for English transcripts
        for t in transcripts:
            if t.language_code == "en":
                if t.is_generated:
                    generated_transcript = t
                else:
                    manual_transcript = t
                    break

        # Prioritize manually created English transcripts
        selected_transcript = manual_transcript or generated_transcript

        # If English is unavailable, use another available transcript
        if selected_transcript is None:
            for t in transcripts:
                selected_transcript = t
                break

        if selected_transcript is None:
            print("No transcript available for this video.")
            return []

        # Fetch transcript
        fetched_transcript = selected_transcript.fetch()

        # Convert transcript into dictionaries
        transcript_data = fetched_transcript.to_raw_data()

        print("Transcript fetched successfully!")
        print("Number of transcript segments:", len(transcript_data))

        return transcript_data

    except Exception as e:
        print("Error fetching transcript:", repr(e))
        return []





def process(transcript):
    # Check if transcript is available
    if not transcript:
        return ""

    txt = ""

    for i in transcript:
        # Handle dictionary transcript segments
        if isinstance(i, dict):
            text = i.get("text", "")
            start = i.get("start", 0)

        # Handle transcript objects
        else:
            text = getattr(i, "text", "")
            start = getattr(i, "start", 0)

        # Preserve text and timestamps
        txt += f"Text: {text} Start: {start}\n"

    return txt





def chunk_transcript(processed_transcript, chunk_size=200, chunk_overlap=20):
    # Initialize the RecursiveCharacterTextSplitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Split the transcript into chunks
    chunks = text_splitter.split_text(processed_transcript)
    return chunks



def define_parameters():
    # Return parameters for the Llama model
    return {
        "temperature": 0,
        "num_predict": 900
    }



def setup_credentials():
    # Define the Llama model ID
    model_id = "llama3.2"

    # Initialize the Llama model using Ollama
    llm = ChatOllama(
        model=model_id,
        temperature=0,
        num_predict=900
    )

    # Return the model ID and LLM object
    return model_id, llm


def initialize_ollama_llm(model_id, parameters):
    # If model_id is a tuple, extract the model name
    if isinstance(model_id, tuple):
        model_id = model_id[0]

        print("Model ID:", model_id)
        print("Model ID type:", type(model_id))




    # Validate the model name
    if not isinstance(model_id, str):
        raise TypeError(
            f"Expected model_id to be a string, "
            f"but received {type(model_id).__name__}: {model_id!r}"
        )

    # Initialize Ollama LLM
    return ChatOllama(
        model=model_id,
        temperature=parameters.get("temperature", 0),
        num_predict=parameters.get("num_predict", 900)
    )







def setup_embedding_model():
    # Load the embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Return the embedding model
    return embeddings





def create_faiss_index(chunks, embedding_model):
    """
    Create a FAISS index from text chunks using the specified embedding model.

    :param chunks: List of text chunks
    :param embedding_model: The embedding model to use
    <span data-type="emoji" data-name="return"></span> FAISS index
    """
    # Use the FAISS library to create an index from the provided text chunks
    return FAISS.from_texts(chunks, embedding_model)


def perform_similarity_search(faiss_index, query, k=3):
    """
    Search for specific queries within the embedded transcript using the FAISS index.

    :param faiss_index: The FAISS index containing embedded text chunks
    :param query: The text input for the similarity search
    :param k: The number of similar results to return (default is 3)
    <span data-type="emoji" data-name="return"></span> List of similar results
    """
    # Perform the similarity search using the FAISS index
    results = faiss_index.similarity_search(query, k=k)
    return results




def create_summary_prompt():
    """
    Create a PromptTemplate for summarizing a YouTube video transcript.

    <span data-type="emoji" data-name="return"></span> PromptTemplate object
    """
    # Define the template for the summary prompt
    template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an AI assistant tasked with summarizing YouTube video transcripts. Provide concise, informative summaries that capture the main points of the video content.

    Instructions:
    1. Summarize the transcript in a single concise paragraph.
    2. Ignore any timestamps in your summary.
    3. Focus on the spoken content (Text) of the video.

    Note: In the transcript, "Text" refers to the spoken words in the video, and "start" indicates the timestamp when that part begins in the video.<|eot_id|><|start_header_id|>user<|end_header_id|>
    Please summarize the following YouTube video transcript:

    {transcript}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

    # Create the PromptTemplate object with the defined template
    prompt = PromptTemplate(
        input_variables=["transcript"],
        template=template
    )

    return prompt


def create_summary_chain(llm, prompt, verbose=True):
    """
    Create an LLMChain for generating summaries.

    :param llm: Language model instance
    :param prompt: PromptTemplate instance
    :param verbose: Boolean to enable verbose output (default: True)
    <span data-type="emoji" data-name="return"></span> LLMChain instance
    """
    return LLMChain(llm=llm, prompt=prompt, verbose=verbose)




def retrieve(query, faiss_index, k=7):
    """
    Retrieve relevant context from the FAISS index based on the user's query.

    Parameters:
        query (str): The user's query string.
        faiss_index (FAISS): The FAISS index containing the embedded documents.
        k (int, optional): The number of most relevant documents to retrieve (default is 3).

    Returns:
        list: A list of the k most relevant documents (or document chunks).
    """
    relevant_context = faiss_index.similarity_search(query, k=k)
    return relevant_context




def create_qa_prompt_template():
    """
    Create a PromptTemplate for question answering based on video content.
    Returns:
        PromptTemplate: A PromptTemplate object configured for Q&A tasks.
    """

    # Define the template string
    qa_template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an expert assistant providing detailed and accurate answers based on the following video content. Your responses should be:
    1. Precise and free from repetition
    2. Consistent with the information provided in the video
    3. Well-organized and easy to understand
    4. Focused on addressing the user's question directly
    If you encounter conflicting information in the video content, use your best judgment to provide the most likely correct answer based on context.
    Note: In the transcript, "Text" refers to the spoken words in the video, and "start" indicates the timestamp when that part begins in the video.<|eot_id|>

    <|start_header_id|>user<|end_header_id|>
    Relevant Video Context: {context}
    Based on the above context, please answer the following question:
    {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    # Create the PromptTemplate object
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=qa_template
    )
    return prompt_template


def create_qa_chain(llm, prompt_template, verbose=True):
    """
    Create an LLMChain for question answering.

    Args:
        llm: Language model instance
            The language model to use in the chain (e.g., WatsonxGranite).
        prompt_template: PromptTemplate
            The prompt template to use for structuring inputs to the language model.
        verbose: bool, optional (default=True)
            Whether to enable verbose output for the chain.

    Returns:
        LLMChain: An instantiated LLMChain ready for question answering.
    """

    return LLMChain(llm=llm, prompt=prompt_template, verbose=verbose)


def generate_answer(question, faiss_index, qa_chain, k=7):
    """
    Retrieve relevant context and generate an answer based on user input.

    Args:
        question: str
            The user's question.
        faiss_index: FAISS
            The FAISS index containing the embedded documents.
        qa_chain: LLMChain
            The question-answering chain (LLMChain) to use for generating answers.
        k: int, optional (default=3)
            The number of relevant documents to retrieve.

    Returns:
        str: The generated answer to the user's question.
    """

    # Retrieve relevant context
    relevant_context = retrieve(question, faiss_index, k=k)

    # Generate answer using the QA chain
    answer = qa_chain.predict(context=relevant_context, question=question)

    return answer


# Initialize an empty string to store the processed transcript after fetching and preprocessing
processed_transcript = ""


def summarize_video(video_url):
    """
    Title: Summarize Video

    Description:
    This function generates a summary of the video using the preprocessed transcript.
    If the transcript hasn't been fetched yet, it fetches it first.

    Args:
        video_url (str): The URL of the YouTube video from which the transcript is to be fetched.

    Returns:
        str: The generated summary of the video or a message indicating that no transcript is available.
    """
    global fetched_transcript, processed_transcript

    if video_url:
        # Fetch and preprocess transcript
        fetched_transcript = get_transcript(video_url)
        processed_transcript = process(fetched_transcript)
    else:
        return "Please provide a valid YouTube URL."

    if processed_transcript:

        # Step 1: Set up the Llama model ID
        model_id = setup_credentials()

        # Step 2: Initialize the Llama LLM for summarization
        llm = initialize_ollama_llm(
            model_id,
            define_parameters()
        )


        # Step 3: Create the summary prompt and chain
        summary_prompt = create_summary_prompt()
        summary_chain = create_summary_chain(llm, summary_prompt)

        # Step 4: Generate the video summary
        summary = summary_chain.run({"transcript": processed_transcript})
        return summary
    else:
        return "No transcript available. Please fetch the transcript first."


def answer_question(video_url, user_question):
    """
    Title: Answer User's Question

    Description:
    This function retrieves relevant context from the FAISS index based on the user’s query
    and generates an answer using the preprocessed transcript.
    If the transcript hasn't been fetched yet, it fetches it first.

    Args:
        video_url (str): The URL of the YouTube video from which the transcript is to be fetched.
        user_question (str): The question posed by the user regarding the video.

    Returns:
        str: The answer to the user's question or a message indicating that the transcript
             has not been fetched.
    """
    global fetched_transcript, processed_transcript

    # Check if the transcript needs to be fetched
    if not processed_transcript:
        if video_url:
            # Fetch and preprocess transcript
            fetched_transcript = get_transcript(video_url)
            processed_transcript = process(fetched_transcript)
        else:
            return "Please provide a valid YouTube URL."

    if processed_transcript and user_question:
        # Step 1: Chunk the transcript (only for Q&A)
        chunks = chunk_transcript(processed_transcript)

        # Step 2: Set up IBM Watson credentials

        # Step 3: Set up the Llama model ID
        model_id = setup_credentials()

        # Step 4: Initialize Llama LLM for Q&A
        llm = initialize_ollama_llm(
            model_id,
            define_parameters()
        )

        # Step 4: Create FAISS index for transcript chunks (only needed for Q&A)

        # Initialize the embedding model for FAISS vector search
        embedding_model = setup_embedding_model()

        faiss_index = create_faiss_index(chunks, embedding_model)

        # Step 5: Set up the Q&A prompt and chain
        qa_prompt = create_qa_prompt_template()
        qa_chain = create_qa_chain(llm, qa_prompt)

        # Step 6: Generate the answer using FAISS index
        answer = generate_answer(user_question, faiss_index, qa_chain)
        return answer
    else:
        return "Please provide a valid question and ensure the transcript has been fetched."


with gr.Blocks() as interface:
    gr.Markdown(
        "<h2 style='text-align: center;'>YouTube Video Summarizer and Q&A</h2>"
    )

    # Input field for YouTube URL
    video_url = gr.Textbox(label="YouTube Video URL", placeholder="Enter the YouTube Video URL")

    # Outputs for summary and answer
    summary_output = gr.Textbox(label="Video Summary", lines=5)
    question_input = gr.Textbox(label="Ask a Question About the Video", placeholder="Ask your question")
    answer_output = gr.Textbox(label="Answer to Your Question", lines=5)

    # Buttons for selecting functionalities after fetching transcript
    summarize_btn = gr.Button("Summarize Video")
    question_btn = gr.Button("Ask a Question")

    # Display status message for transcript fetch
    transcript_status = gr.Textbox(label="Transcript Status", interactive=False)

    # Set up button actions
    summarize_btn.click(summarize_video, inputs=video_url, outputs=summary_output)
    question_btn.click(answer_question, inputs=[video_url, question_input], outputs=answer_output)

# Launch the app with specified server name and port
interface.launch(
    server_name="127.0.0.1",
    server_port=7860,
    share=True
)


