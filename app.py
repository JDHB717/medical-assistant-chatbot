import os
import streamlit as st
import PyPDF2

from langchain.document_loaders import DirectoryLoader, PyPDFLoader, CSVLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain

from utils import intent_classifier, semantic_search, ensure_fit_tokens, get_page_contents
from prompts import human_template, system_message
from render import user_msg_container_html_template, bot_msg_container_html_template
import openai
import boto3
from botocore.exceptions import NoCredentialsError

# Set OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]
AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
BUCKET_NAME = st.secrets["BUCKET_NAME"]
REGION = st.secrets["REGION"]
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

s3 = boto3.client('s3',
                  aws_access_key_id=AWS_ACCESS_KEY_ID,
                  aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

st.set_page_config(layout="wide")
st.header("Informed CareGuide")

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Load the Buffett and Branson databases
medicalDB = Chroma(persist_directory=os.path.join('db', 'medical'), embedding_function=embeddings)
medicalDB_retriever = medicalDB.as_retriever(search_kwargs={"k": 3})

persist_directory = 'db'
# Initialize session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Construct messages from chat history
def construct_messages(history):
    messages = [{"role": "system", "content": system_message}]
    
    for entry in history:
        role = "user" if entry["is_user"] else "assistant"
        messages.append({"role": role, "content": entry["message"]})
    
    # Ensure total tokens do not exceed model's limit
    messages = ensure_fit_tokens(messages)
    
    return messages

def medicalDB_handler(query):
    print("Using Branson handler...")
    # Get relevant documents from Branson's database
    relevant_docs = medicalDB_retriever.get_relevant_documents(query)

    # Use the provided function to prepare the context
    context = get_page_contents(relevant_docs)

    # Prepare the prompt for GPT-3.5-turbo with the context
    query_with_context = human_template.format(query=query, context=context)

    return {"role": "user", "content": query_with_context}


def other_handler(query):
    print("Using other handler...")
    # Return the query in the appropriate message format
    return {"role": "user", "content": query}


# Function to route query to correct handler based on category
def route_by_category(query, category):
    if category == "0":
        return medicalDB_handler(query)
    elif category == "1":
        return other_handler(query)
    else:
        raise ValueError("Invalid category")

# Function to generate response
def generate_response():
    # Append user's query to history
    st.session_state.history.append({
        "message": st.session_state.prompt,
        "is_user": True
    })
    
    print("st.session_state.custom_prompt:")
    print(st.session_state.custom_prompt)
    
    print("st.session_state.prompt:")
    print(st.session_state.custom_prompt + ' ' + st.session_state.prompt)
    
    # Classify the intent
    category = intent_classifier(st.session_state.custom_prompt + ' ' + st.session_state.prompt)
    print("category:");
    print(category);
    # Route the query based on category
    new_message = route_by_category(st.session_state.custom_prompt + ' ' + st.session_state.prompt, category)
    print("new_message:");
    print(new_message);
    # Construct messages from chat history
    messages = construct_messages(st.session_state.history)
    print("messages:");
    print(messages);
    # Add the new_message to the list of messages before sending it to the API
    messages.append(new_message)
    
    # Ensure total tokens do not exceed model's limit
    messages = ensure_fit_tokens(messages)
    
    # Call the Chat Completions API with the messages
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    # Extract the assistant's message from the response
    assistant_message = response['choices'][0]['message']['content']
    
    # Append assistant's message to history
    st.session_state.history.append({
        "message": assistant_message,
        "is_user": False
    })
# Upload DOCX to s3
def upload_docx_to_s3(file, filename):
    print("Uploading doc")
    try:
        # Save file to the Streamlit Server
        with open(os.path.join("./docs", filename), "wb") as f:
            f.write(file.getbuffer())

        # Upload the file to S3 bucket
        s3.upload_file(os.path.join("./docs", filename), BUCKET_NAME, filename, ExtraArgs={'ACL': 'public-read'})
        st.success(f"{filename} uploaded to S3 bucket successfully!")
        doc_dir = f"https://{BUCKET_NAME}.s3.{REGION}.amazonaws.com/{filename}"

        print(doc_dir)

        # Load DOC documents from directory, generate embeddings, and persist to Chroma
        embeddings = OpenAIEmbeddings()
        doc_docs = Docx2txtLoader(os.path.join('./docs', filename)).load()
        doc_docs_list = list(doc_docs)  # Convert to list to make it iterable

        print(doc_docs)

        docDB = Chroma.from_documents(doc_docs_list, embeddings, persist_directory=os.path.join(persist_directory, 'medical'))
        docDB.persist()
    except FileNotFoundError:
        st.error(f"File not found: {filename}")
    except NoCredentialsError:
        st.error("Credentials not available!")
        
# Upload csv to s3
def upload_csv_to_s3(file, filename):
    print("Uploading csv")
    try:
        # Save file to the Streamlit Server
        with open(os.path.join("./docs", filename), "wb") as f:
            f.write(file.getbuffer())

        # Upload the file to S3 bucket
        s3.upload_file(os.path.join("./docs", filename), BUCKET_NAME, filename, ExtraArgs={'ACL': 'public-read'})
        st.success(f"{filename} uploaded to S3 bucket successfully!")
        csv_dir = f"https://{BUCKET_NAME}.s3.{REGION}.amazonaws.com/{filename}"

        print(csv_dir)

        # Load CSV documents from directory, generate embeddings, and persist to Chroma
        embeddings = OpenAIEmbeddings()
        csv_docs = CSVLoader(filename=os.path.join('./docs', filename), delimiter=',').load()
        csv_docs_list = list(csv_docs)  # Convert to list to make it iterable

        print(csv_docs)

        csvDB = Chroma.from_documents(csv_docs_list, embeddings, persist_directory=os.path.join(persist_directory, 'medical'))
        csvDB.persist()
    except FileNotFoundError:
        st.error(f"File not found: {filename}")
    except NoCredentialsError:
        st.error("Credentials not available!")
        
# Upload txt to s3
def upload_txt_to_s3(file, filename):
    print("Uploading txt")
    try:
        # Save file to the Streamlit Server
        with open(os.path.join("./docs", filename), "wb") as f:
            f.write(file.getbuffer())

        # Upload the file to S3 bucket
        s3.upload_file(os.path.join("./docs", filename), BUCKET_NAME, filename, ExtraArgs={'ACL': 'public-read'})
        st.success(f"{filename} uploaded to S3 bucket successfully!")
        txt_dir = f"https://{BUCKET_NAME}.s3.{REGION}.amazonaws.com/{filename}"

        print(txt_dir)

        # Load TXT documents from directory, generate embeddings, and persist to Chroma
        embeddings = OpenAIEmbeddings()
        txt_docs = TextLoader(os.path.join('./docs', filename)).load()
        txt_docs_list = list(txt_docs)  # Convert to list to make it iterable

        print(txt_docs)

        txtDB = Chroma.from_documents(txt_docs_list, embeddings, persist_directory=os.path.join(persist_directory, 'medical'))
        txtDB.persist()
    except FileNotFoundError:
        st.error(f"File not found: {filename}")
    except NoCredentialsError:
        st.error("Credentials not available!")
        
# Upload pdf to s3
def upload_pdf_to_s3(file, filename):
    print("Uploading pdf")
    try:
        # Save file to the Streamlit Server
        with open(os.path.join("./docs", filename), "wb") as f:
            f.write(file.getbuffer())

        # Upload the file to S3 bucket
        s3.upload_file(os.path.join("./docs", filename), BUCKET_NAME, filename, ExtraArgs={'ACL': 'public-read'})
        st.success(f"{filename} uploaded to S3 bucket successfully!")
        pdf_dir = f"https://{BUCKET_NAME}.s3.{REGION}.amazonaws.com/{filename}"

        print(os.path.join("./docs"))

        # Load PDF documents from directory, generate embeddings, and persist to Chroma
        embeddings = OpenAIEmbeddings()
        text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=8)
        
        pdf_loader = PyPDFLoader(pdf_dir)
        pdf_pages = pdf_loader.load_and_split()
        
        pdfDB = Chroma.from_documents(pdf_pages, embeddings, persist_directory=os.path.join(persist_directory, 'medical'))
        pdfDB.persist()
    except FileNotFoundError:
        st.error(f"File not found: {filename}")
    except NoCredentialsError:
        st.error("Credentials not available!")
        
# Upload DB to s3 bucket
def upload_to_s3(file, filename):
    extension = os.path.splitext(filename)[1] 
    if extension.lower() == '.pdf':
        upload_pdf_to_s3(file, filename)
    elif extension.lower() == '.csv':
        upload_csv_to_s3(file, filename)
    elif extension.lower() == '.txt':
        upload_txt_to_s3(file, filename)
    elif extension.lower() == '.docx':
        upload_docx_to_s3(file, filename)
    else:
        st.warning(f"{filename} is not a supported file type.")
        return None
        

#Display Chat Result
def displayChatHistory():
    # Display chat history
    messages = st.session_state.history
    print(messages)
    
    messages_str = ""
    for message in messages:
        if message["is_user"] == True:
            messages_str += f"\nUser:\t{message['message']}\n\n"
        elif message["is_user"] == False:
            messages_str += f"Informed CareGuide:\t{message['message']}\n"

    # Return the messages string
    return messages_str
    
def tab1_content():

    col1, col3, col2 = st.columns([3.5, 2, 4])

    # File uploader
    with col1:
        uploaded_file = st.file_uploader("", type=['csv','pdf','txt', 'docx'], key='fileUploader', accept_multiple_files=False)
        # upload file to S3 bucket on button click
        if uploaded_file is not None:
            # call upload_to_s3 immediately after the file is uploaded
            upload_to_s3(uploaded_file, uploaded_file.name)
                
    #space content
    with col3:
        st.empty()

    # Take user input
    with col2:
        st.text_area("", height=100, placeholder="", key="custom_prompt")
    
    col4, col5, col6 = st.columns([8, 1, 2])
    
    # Take user input
    with col4:
        st.text_area("", height=450, value = displayChatHistory(), key="result");
        st.text_input("",key="prompt",placeholder="e.g. 'How can I treat my hurt?'", on_change=generate_response)
    
    with col5:
        st.empty()
    
    with col6:
        st.markdown("""
            <style>
            div[role="button"] {
                border-radius: 50%;
            }
            </style>
        """, unsafe_allow_html=True)
        
        st.button("Voice Mode")
    
if "input_form" not in st.session_state:
    st.session_state["input_form"] = {}
    
# Define the contents of the second tab
def tab2_content():

    # Five checkboxes
    if st.checkbox("Option 1"):
        st.write("Option 1 selected")
    if st.checkbox("Option 2"):
        st.write("Option 2 selected")
    if st.checkbox("Option 3"):
        st.write("Option 3 selected")
    if st.checkbox("Option 4"):
        st.write("Option 4 selected")
    if st.checkbox("Option 5"):
        st.write("Option 5 selected")

# Set up the tabs
tabs = ["Tab 1", "Tab 2"]
current_tab = st.sidebar.selectbox("", tabs)


# Display the appropriate tab
if current_tab == "Tab 1":
    tab1_content()
else:
    tab2_content()
    

