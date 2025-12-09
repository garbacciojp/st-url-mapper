import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import warnings
import multiprocessing as mp

# Disable safetensors usage
os.environ["TRANSFORMERS_NO_SAFETENSORS"] = "1"

# Set environment variables to mitigate OpenMP and fork safety issues on macOS
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_INIT_AT_FORK"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Monkey-patch the resource_tracker on macOS to ignore semaphore registrations
if sys.platform == "darwin":
    import multiprocessing.resource_tracker as resource_tracker
    original_register = resource_tracker.register
    def register(name, rtype):
        if rtype == "semaphore":
            return
        return original_register(name, rtype)
    resource_tracker.register = register

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

warnings.filterwarnings("ignore", message="resource_tracker: There appear to be")

def create_text_field(df, cols):
    """Combine selected columns into one text field."""
    return df[cols].fillna("").astype(str).agg(" ".join, axis=1)

# App title and description
st.title("Semantic URL Mapper")

# Inject custom CSS for green styling of the expander header and content
st.markdown(
    """
    <style>
    /* Expander header styling */
    div[data-testid="stExpander"] div[role="button"] {
        background-color: #d4edda !important;
        border: 1px solid #c3e6cb !important;
        border-radius: 5px !important;
        padding: 10px !important;
    }
    /* Expander content styling */
    div[data-testid="stExpander"] div[data-testid="stExpanderContent"] {
        background-color: #d4edda !important;
        border: 1px solid #c3e6cb !important;
        border-radius: 5px !important;
        padding: 10px !important;
    }
    </style>
    """, unsafe_allow_html=True
)

# Checklist instructions
st.write("### How to Use This App:")
st.markdown(
    """
1. **Upload CSV Files:** Upload your original URLs CSV file and your new destination URLs CSV file.
2. **Select Descriptive Columns:** Choose the columns that contain descriptive data from both files.
3. **Map URLs:** Click on the **Run URL Mapping** button to start the matching process.
    """
)

# Expandable "How It Works" section with a lightbulb icon
with st.expander("💡 How It Works"):
    st.markdown(
        """
The app leverages **MiniLM** (using the paraphrase-MiniLM-L6-v2 model) to convert your descriptive data into numerical embeddings that capture semantic meaning.

It then uses **FAISS**, an efficient similarity search library, to index these embeddings and quickly find the new URL that best matches each original URL.
        """
    )

# File uploaders for CSV files
src_file = st.file_uploader("Upload Source CSV", type="csv", key="src")
tgt_file = st.file_uploader("Upload Destination CSV", type="csv", key="tgt")

if src_file and tgt_file:
    # Read the CSV files into dataframes
    src_df = pd.read_csv(src_file)
    tgt_df = pd.read_csv(tgt_file)

    st.success(f"✅ Source file loaded: {len(src_df)} rows, columns: {', '.join(src_df.columns.tolist())}")
    st.success(f"✅ Target file loaded: {len(tgt_df)} rows, columns: {', '.join(tgt_df.columns.tolist())}")

    st.header("Choose Columns for Semantic Matching")
    st.write("Select which columns contain the descriptive text to match on:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Source File Columns")
        src_selected_cols = st.multiselect(
            "Select descriptive columns from Source file", 
            options=list(src_df.columns),
            key="src_cols",
            help="Choose columns that describe what the URL is about"
        )
    
    with col2:
        st.subheader("Target File Columns")
        tgt_selected_cols = st.multiselect(
            "Select descriptive columns from Target file", 
            options=list(tgt_df.columns),
            key="tgt_cols",
            help="Choose columns that describe what the URL is about"
        )

    if not src_selected_cols:
        st.warning("⚠️ Please select at least one column from the Source file.")
    if not tgt_selected_cols:
        st.warning("⚠️ Please select at least one column from the Target file.")

    if src_selected_cols and tgt_selected_cols:
        st.success(f"✅ Will match using: Source[{', '.join(src_selected_cols)}] ↔ Target[{', '.join(tgt_selected_cols)}]")
        
        st.subheader("URL Column Selection")
        st.write("Now select which columns contain the actual URLs:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Use 'Address' column if it exists, otherwise let user choose
            if "Address" in src_df.columns:
                src_url_col = "Address"
                st.info("✅ Auto-detected 'Address' column from source file")
            else:
                src_url_col = st.selectbox(
                    "Source URL column", 
                    list(src_df.columns), 
                    key="src_url",
                    help="The column containing the source URLs"
                )
        
        with col2:
            if "Address" in tgt_df.columns:
                tgt_url_col = "Address"
                st.info("✅ Auto-detected 'Address' column from target file")
            else:
                tgt_url_col = st.selectbox(
                    "Target URL column", 
                    list(tgt_df.columns), 
                    key="tgt_url",
                    help="The column containing the target URLs"
                )

        # Button to start the mapping process
        if st.button("🚀 Run URL Mapping", type="primary"):
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Step 1/5: Combining columns into text fields...")
                progress_bar.progress(10)
                # Create a new text field for embedding using the selected columns
                src_df["semantic_text"] = create_text_field(src_df, src_selected_cols)
                tgt_df["semantic_text"] = create_text_field(tgt_df, tgt_selected_cols)

                status_text.text("Step 2/5: Loading AI model (this may take a moment on first run)...")
                progress_bar.progress(20)
                # Option 1: Load model from Hugging Face with a cache folder
                model = SentenceTransformer("paraphrase-MiniLM-L6-v2", cache_folder="./model_cache")
                
                # Option 2 (if you have downloaded the model locally, uncomment the next line and comment the above):
                # model = SentenceTransformer("./paraphrase-MiniLM-L6-v2")
                
                status_text.text(f"Step 3/5: Encoding {len(src_df)} source URLs...")
                progress_bar.progress(40)
                src_embeddings = model.encode(src_df["semantic_text"].tolist(), show_progress_bar=False)
                
                status_text.text(f"Step 4/5: Encoding {len(tgt_df)} target URLs...")
                progress_bar.progress(60)
                tgt_embeddings = model.encode(tgt_df["semantic_text"].tolist(), show_progress_bar=False)

                status_text.text("Step 5/5: Finding best matches using FAISS...")
                progress_bar.progress(80)
                # Build a FAISS index for the destination embeddings
                vec_dimension = src_embeddings.shape[1]
                index = faiss.IndexFlatL2(vec_dimension)
                index.add(np.array(tgt_embeddings, dtype=np.float32))

                # Perform a nearest-neighbor search for each source embedding
                distances, indices = index.search(np.array(src_embeddings, dtype=np.float32), k=1)

                # Normalize distances and convert to a similarity score
                norm_factor = distances.max()
                similarity = 1 - (distances / norm_factor)

                # Create a results DataFrame
                mapping_df = pd.DataFrame({
                    "Source URL": src_df[src_url_col],
                    "Matched URL": tgt_df[tgt_url_col].iloc[indices.flatten()].values,
                    "Similarity Score": np.round(similarity.flatten(), 4)
                })
                
                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()

                st.success(f"✅ Mapping complete! Matched {len(mapping_df)} URLs.")
                st.dataframe(mapping_df, use_container_width=True)

                # Allow the user to download the results as a CSV
                csv_data = mapping_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Mapping as CSV",
                    data=csv_data,
                    file_name="matched_urls.csv",
                    mime="text/csv",
                    type="primary"
                )
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.exception(e)