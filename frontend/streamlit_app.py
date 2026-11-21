import streamlit as st
import requests
import base64
import io

from PIL import Image


# ==========================================
# CONFIGURATION
# ==========================================

BACKEND_URL = "http://127.0.0.1:8000"

IMAGE_API_URL = f"{BACKEND_URL}/api/image/predict"
RAG_INGEST_URL = f"{BACKEND_URL}/api/rag/ingest"
RAG_ASK_URL = f"{BACKEND_URL}/api/rag/ask"


st.set_page_config(
    page_title="Intelligent Healthcare Assistant",
    page_icon="🩺",
    layout="wide",
)


# ==========================================
# HEADER
# ==========================================

st.title("🩺 Intelligent Healthcare Assistant")

st.caption(
    "AI-powered healthcare assistance with "
    "medical image analysis and document-based knowledge."
)

st.divider()


# ==========================================
# TABS
# ==========================================

xray_tab, rag_tab = st.tabs(
    [
        "🩻 Chest X-ray Analysis",
        "📚 Medical Knowledge Assistant",
    ]
)


# ==========================================
# TAB 1: CHEST X-RAY ANALYSIS
# ==========================================

with xray_tab:

    st.header("Chest X-ray Analysis")

    st.write(
        "Upload a chest X-ray image to analyze it "
        "using the trained deep learning model."
    )

    uploaded_file = st.file_uploader(
        "Upload Chest X-ray",
        type=["jpg", "jpeg", "png"],
        key="xray_upload",
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Chest X-ray",
            use_container_width=True,
        )

        if st.button(
            "Analyze X-ray",
            type="primary",
            key="analyze_xray",
        ):

            with st.spinner("Analyzing X-ray..."):

                try:

                    image_bytes = uploaded_file.getvalue()

                    files = {
                        "file": (
                            uploaded_file.name,
                            image_bytes,
                            uploaded_file.type,
                        )
                    }

                    response = requests.post(
                        IMAGE_API_URL,
                        files=files,
                        timeout=120,
                    )

                    response.raise_for_status()

                    result = response.json()

                    prediction = result["predicted_class"]

                    st.divider()

                    st.subheader("Analysis Results")

                    if prediction == "PNEUMONIA":

                        st.warning(
                            f"Prediction: {prediction}"
                        )

                    else:

                        st.success(
                            f"Prediction: {prediction}"
                        )

                    # ==================================
                    # CLASS PROBABILITIES
                    # ==================================

                    st.subheader("Class Probabilities")

                    probabilities = result["probabilities"]

                    for class_name, probability in probabilities.items():

                        st.write(
                            f"{class_name}: "
                            f"{probability * 100:.2f}%"
                        )

                        st.progress(
                            float(probability)
                        )

                    # ==================================
                    # GRAD-CAM
                    # ==================================

                    st.divider()

                    st.subheader(
                        "Grad-CAM Explainability"
                    )

                    gradcam_data = result.get(
                        "gradcam_png_base64"
                    )

                    if gradcam_data:

                        gradcam_bytes = base64.b64decode(
                            gradcam_data
                        )

                        gradcam_image = Image.open(
                            io.BytesIO(gradcam_bytes)
                        )

                        st.image(
                            gradcam_image,
                            caption=(
                                "Grad-CAM visualization. "
                                "Highlighted regions indicate "
                                "areas influencing the model."
                            ),
                            use_container_width=True,
                        )

                    else:

                        st.warning(
                            result.get(
                                "gradcam_note",
                                "Grad-CAM could not be generated.",
                            )
                        )

                    # ==================================
                    # DISCLAIMER
                    # ==================================

                    st.info(
                        result.get(
                            "disclaimer",
                            "Educational model output only. "
                            "Not a medical diagnosis.",
                        )
                    )

                except requests.exceptions.ConnectionError:

                    st.error(
                        "Cannot connect to backend. "
                        "Make sure FastAPI is running."
                    )

                except requests.exceptions.Timeout:

                    st.error(
                        "The prediction request timed out."
                    )

                except requests.exceptions.HTTPError as exc:

                    st.error(
                        f"API error: {exc}"
                    )

                except Exception as exc:

                    st.error(
                        f"Unexpected error: {exc}"
                    )


# ==========================================
# TAB 2: MEDICAL KNOWLEDGE ASSISTANT
# ==========================================

with rag_tab:

    st.header("Medical Knowledge Assistant")

    st.write(
        "Ask questions based on medical PDF documents "
        "indexed in the healthcare knowledge base."
    )

    # ==========================================
    # DOCUMENT INGESTION
    # ==========================================

    st.subheader("Knowledge Base")

    st.info(
        "Place your medical PDF documents inside "
        "`medical_documents/raw/` in your project folder."
    )

    if st.button(
        "📥 Ingest Medical Documents",
        type="primary",
        key="ingest_documents",
    ):

        with st.spinner(
            "Extracting documents and generating embeddings..."
        ):

            try:

                response = requests.post(
                    RAG_INGEST_URL,
                    timeout=300,
                )

                response.raise_for_status()

                ingestion_result = response.json()

                st.success(
                    ingestion_result.get(
                        "message",
                        "Ingestion completed.",
                    )
                )

                st.metric(
                    "Chunks Indexed",
                    ingestion_result.get(
                        "ingested_chunks",
                        0,
                    ),
                )

                documents = ingestion_result.get(
                    "documents",
                    [],
                )

                if documents:

                    st.subheader("Indexed Documents")

                    for document in documents:

                        st.write(
                            f"📄 {document}"
                        )

                else:

                    st.warning(
                        "No documents were indexed. "
                        "Check your medical_documents/raw folder."
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Cannot connect to backend. "
                    "Make sure FastAPI is running."
                )

            except requests.exceptions.Timeout:

                st.error(
                    "Document ingestion timed out."
                )

            except requests.exceptions.HTTPError as exc:

                st.error(
                    f"Ingestion API error: {exc}"
                )

            except Exception as exc:

                st.error(
                    f"Unexpected error: {exc}"
                )

    st.divider()

    # ==========================================
    # CHAT HISTORY
    # ==========================================

    st.subheader("Medical Document Chat")

    if "rag_messages" not in st.session_state:

        st.session_state.rag_messages = []

    # Display previous messages

    for message in st.session_state.rag_messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message.get("sources"):

                with st.expander(
                    "📚 View Sources"
                ):

                    for source in message["sources"]:

                        st.markdown(
                            f"**Document:** {source['source']}"
                        )

                        st.markdown(
                            f"**Page:** {source.get('page', 'N/A')}"
                        )

                        st.markdown(
                            f"**Similarity:** "
                            f"{source.get('similarity', 0):.4f}"
                        )

                        st.markdown(
                            "**Retrieved Passage:**"
                        )

                        st.write(
                            source.get("excerpt", "")
                        )

                        st.divider()

    # ==========================================
    # CHAT INPUT
    # ==========================================

    question = st.chat_input(
        "Ask a question about your medical documents..."
    )

    if question:

        # Display user question

        st.session_state.rag_messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):

            st.markdown(question)

        # Generate answer

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching medical documents..."
            ):

                try:

                    response = requests.post(
                        RAG_ASK_URL,
                        json={
                            "question": question,
                            "top_k": 4,
                        },
                        timeout=120,
                    )

                    response.raise_for_status()

                    result = response.json()

                    answer = result.get(
                        "answer",
                        "No answer generated.",
                    )

                    sources = result.get(
                        "sources",
                        [],
                    )

                    disclaimer = result.get(
                        "disclaimer",
                        "Educational information only. "
                        "Not medical advice.",
                    )

                    st.markdown(answer)

                    # ==================================
                    # SOURCE CITATIONS
                    # ==================================

                    if sources:

                        with st.expander(
                            "📚 View Retrieved Sources",
                            expanded=True,
                        ):

                            for source in sources:

                                st.markdown(
                                    f"**Document:** "
                                    f"{source['source']}"
                                )

                                st.markdown(
                                    f"**Page:** "
                                    f"{source.get('page', 'N/A')}"
                                )

                                st.markdown(
                                    f"**Similarity Score:** "
                                    f"{source.get('similarity', 0):.4f}"
                                )

                                st.markdown(
                                    "**Relevant Passage:**"
                                )

                                st.write(
                                    source.get(
                                        "excerpt",
                                        "",
                                    )
                                )

                                st.divider()

                    # ==================================
                    # DISCLAIMER
                    # ==================================

                    st.info(disclaimer)

                    # Store answer in history

                    st.session_state.rag_messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )

                except requests.exceptions.ConnectionError:

                    st.error(
                        "Cannot connect to backend. "
                        "Make sure FastAPI is running."
                    )

                except requests.exceptions.Timeout:

                    st.error(
                        "The RAG request timed out."
                    )

                except requests.exceptions.HTTPError as exc:

                    st.error(
                        f"RAG API error: {exc}"
                    )

                except Exception as exc:

                    st.error(
                        f"Unexpected error: {exc}"
                    )

    # ==========================================
    # CLEAR CHAT
    # ==========================================

    if st.session_state.rag_messages:

        if st.button(
            "🗑️ Clear Chat History",
            key="clear_rag_chat",
        ):

            st.session_state.rag_messages = []

            st.rerun()


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "⚕️ Intelligent Healthcare Assistant | "
    "Educational and research purposes only. "
    "This application is not a substitute for professional medical advice."
)