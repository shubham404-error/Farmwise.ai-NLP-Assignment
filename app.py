import streamlit as st
#from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import sqlite3
from datetime import datetime
import streamlit as st
import random
import time
def create_table():
    conn = sqlite3.connect('hotel_booking.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bookings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_name TEXT,
            email TEXT,
            age INTEGER,
            gender TEXT,
            membership_level TEXT,
            residence TEXT,
            booking_date TEXT,
            hotel_name TEXT,
            check_in_date TEXT,
            check_out_date TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Function to insert data into the SQLite table
def insert_data(data):
    conn = sqlite3.connect('hotel_booking.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO bookings (
            customer_name, email, age, gender, membership_level,
            residence, booking_date, hotel_name, check_in_date, check_out_date
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', data)
    conn.commit()
    conn.close()

with st.sidebar:
       # Create SQLite table if not exists
    create_table()

    # Set app title and header
    st.title("Hotel Booking App")
    st.header("Fill in the details to book a hotel")

    # Get user input
    customer_name = st.text_input("Customer Name")
    email = st.text_input("Email")
    age = st.number_input("Age", min_value=0, max_value=120)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    # Add dropdown for membership
    membership_options = ["Bronze", "Silver", "Gold", "Diamond", "Platinum", "Ascendant"]
    membership_level = st.selectbox("Membership Level", membership_options)

    # Add dropdown for hotels based on the selected membership
    hotel_options = {
        "Bronze": ["The Langham Chicago", "Hyatt Regency Chicago", "The Gwen Hotel Chicago"],
        "Silver": ["The Langham Chicago", "Hyatt Regency Chicago", "The Gwen Hotel Chicago"],
        "Gold": ["The Drake Hotel, Chicago", "The Langham Chicago", "Hyatt Regency Chicago", "The Gwen Hotel Chicago"],
        "Diamond": ["The Drake Hotel, Chicago", "Waldorf Astoria Chicago", "The Gwen Hotel Chicago"],
        "Platinum": ["The Drake Hotel, Chicago", "The Langham Chicago", "Hyatt Regency Chicago", "The Gwen Hotel Chicago"],
        "Ascendant": ["The Drake Hotel, Chicago", "Waldorf Astoria Chicago", "The Gwen Hotel Chicago"]
    }

    selected_hotels = st.selectbox("Hotel Name", hotel_options.get(membership_level, []))

    residence = st.text_input("Residence")
    booking_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create columns for check-in and check-out dates
    col1, col2 = st.columns(2)
    with col1:
        check_in_date = st.date_input("Check-in Date")

    with col2:
        check_out_date = st.date_input("Check-out Date")

    # Save data on button click
    if st.button("Book Hotel"):
        data = (
            customer_name, email, age, gender, membership_level,
            residence, booking_date, selected_hotels,
            check_in_date.strftime("%Y-%m-%d"), check_out_date.strftime("%Y-%m-%d")
        )
        insert_data(data)
        st.success("Booking Successful!")
    



load_dotenv()
# print(os.getenv("GOOGLE_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


'''def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks'''


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    # print(response)
    # st.write("Reply: ", response["output_text"])
    return response


'''pdf_docs=["The_Langham_Chicago.pdf","The_Gwen_Hotel_Chicago.pdf","Waldorf_Astoria_Chicago.pdf","Hyatt_Regency_Chicago.pdf","ANNEXURE.pdf","Membership_Level_Details.pdf","TIMESHARE_AGREEMENT.pdf"]
raw_text = get_pdf_text(pdf_docs)
text_chunks = get_text_chunks(raw_text)
get_vector_store(text_chunks)'''


def main():
    st.header("Ask a question💁")
    
    # user_question = st.text_input("Ask a Question from the PDF Files")

    # if user_question:
    #     user_input(user_question)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        response= user_input(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = response
            assistant_response = response
            # Simulate stream of response with milliseconds delay
            
            message_placeholder.markdown(full_response['output_text'])
            message_placeholder.markdown(full_response['output_text'])
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response['output_text']})


if __name__ == "__main__":
    main()
# Add a footer to the app
footer="""<style>
a:link , a:visited{
color: white;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: #f5f5f5;
color: #444;
text-align: center;
}
</style>
<div class="footer">
<p>Made with love❤️ by Shubham Tejani &copy; 2024</p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
