import streamlit as st
from helper.test import process_documents, process_llm_response
from streamlit_navigation_bar import st_navbar
from streamlit_extras.stylable_container import stylable_container

st.set_page_config(page_title="DocMania", page_icon="📑")

def home():
    st.markdown('<h1 style="color: red;">DocMania</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: aqua;">First upload your documents in **Document Management** then ask your questions</p>', unsafe_allow_html=True)
    with st.sidebar:
        st.markdown('<h3 style="color: orange;">Document Management</h3>', unsafe_allow_html=True)
        st.image("./static/folders.png",width=100,use_column_width=100)

        uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

        delete_files = st.multiselect("Select files to delete (if any):", options=[f.name for f in uploaded_files])

        with stylable_container(
                "red",
                css_styles="""
            button {
                background-color: #FF0000;
                color: black;

            }""",
        ):
            if st.button("Delete Selected Files", key="delete_files"):
                if delete_files:
                    uploaded_files = [f for f in uploaded_files if f.name not in delete_files]
                    st.write(f"Files deleted: {', '.join(delete_files)}")
                    # print(len(uploaded_files))

        with stylable_container(
                "green",
                css_styles="""
            button {
                background-color: #00FF00;
                color: black;
            }""",
        ):
            if st.button("Load Documents") and uploaded_files:
                with st.spinner("Processing documents..."):
                    qa_chain = process_documents(uploaded_files)
                    st.session_state.qa_chain = qa_chain
                    st.success("Documents loaded successfully!")

    if "qa_chain" in st.session_state:
        st.subheader("Enter your question:")
        question = st.text_input("")

        if st.button("Ask"):
            if question:
                with st.spinner("Generating response..."):
                    response = st.session_state.qa_chain({"query": question})
                    st.write(process_llm_response(response))
            else:
                st.error("Please enter a question.")


def about():
    st.title("DocMania - About")

    points = [
        "**DocMania** is a cutting-edge project focused on enhancing document information retrieval.",
        "It integrates advanced technologies like the LangChain framework for robust processing.",
        "ChromaDB serves as its efficient storage solution for handling various document types.",
        "The project aims to simplify the process of extracting specific information from documents.",
        "Users can upload documents through an intuitive interface built with Streamlit.",
        "DocMania's backend utilizes NLP techniques to analyze and understand document content.",
        "It employs machine learning algorithms to improve the accuracy of information extraction.",
        "The system categorizes documents and indexes relevant data for quick retrieval.",
        "Users can search for keywords or phrases to locate specific information within documents.",
        "DocMania supports various file formats, ensuring flexibility in document handling.",
        "Security measures are integrated to protect sensitive information stored in ChromaDB.",
        "The Streamlit interface provides real-time feedback and visualizations for enhanced user experience.",
        "The project is designed to scale, accommodating large volumes of documents efficiently.",
        "DocMania's architecture prioritizes speed and accuracy in information retrieval tasks.",
        "It offers customizable search filters to refine results based on user preferences.",
        "Automated updates and version control ensure that users access the most current information.",
        "DocMania can be deployed both locally and on cloud platforms for versatility.",
        "It includes features for collaborative document management and sharing.",
        "The project emphasizes seamless integration with existing workflows and systems.",
        "DocMania aims to revolutionize document handling by optimizing retrieval processes."
    ]

    st.write("\n".join(["- " + point for point in points]))

def contact():
    st.title("DocMania - Contact")
    st.write("For inquiries, please contact us at **[support@docmania@gmail.com](mailto:pavanmandavilli485@gmail.com)**.")

    st.subheader("Send us a message")
    st.write("You can also send us an email directly by clicking the link above.")

selected_page = st_navbar(["Home", "About", "Contact"])

if selected_page == "Home":
    home()
elif selected_page == "About":
    about()
elif selected_page == "Contact":
    contact()
