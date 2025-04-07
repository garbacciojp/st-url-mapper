import os
import sys

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

import multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    # The start method has already been set
    pass

import warnings
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be")

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

def create_text_field(df, cols):
    """Combine selected columns into one text field."""
    return df[cols].fillna("").astype(str).agg(" ".join, axis=1)

# App title and description
st.title("Semantic URL Mapper")

# Inject custom CSS with more specific selectors to style the expander header and content in green
st.markdown(
    """
    <style>
    /* Target only the expander header within stExpander */
    div[data-testid="stExpander"] div[role="button"] {
        background-color: #d4edda !important;
        border: 1px solid #c3e6cb !important;
        border-radius: 5px !important;
        padding: 10px !important;
    }
    /* Target the expander content */
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
The app leverages **MiniLM**, a lightweight transformer model, to convert your descriptive data into numerical embeddings that capture the semantic meaning of the text.

It then uses **FAISS**, an efficient similarity search library, to index these embeddings and quickly find the destination URL that best matches each original URL based on semantic similarity.
        """
    )

# File uploaders for CSV files
src_file = st.file_uploader("Upload Source CSV", type="csv", key="src")
tgt_file = st.file_uploader("Upload Destination CSV", type="csv", key="tgt")

if src_file and tgt_file:
    # Read the CSV files into dataframes
    src_df = pd.read_csv(src_file)
    tgt_df = pd.read_csv(tgt_file)

    st.header("Choose Columns for Semantic Matching")
    # Identify columns common to both datasets
    common_cols = list(set(src_df.columns) & set(tgt_df.columns))
    selected_cols = st.multiselect("Select one or more columns to analyze", options=common_cols)

    if selected_cols:
        st.subheader("URL Column Selection")
        # If a default 'Address' column exists, use it; otherwise let the user choose
        if "Address" in src_df.columns:
            src_url_col = "Address"
            st.info("Using 'Address' column from source file as URL.")
        else:
            src_url_col = st.selectbox("Select the URL column from the source file", src_df.columns, key="src_url")

        if "Address" in tgt_df.columns:
            tgt_url_col = "Address"
            st.info("Using 'Address' column from destination file as URL.")
        else:
            tgt_url_col = st.selectbox("Select the URL column from the destination file", tgt_df.columns, key="tgt_url")

        # Button to start the mapping process
        if st.button("Run URL Mapping"):
            with st.spinner("Calculating semantic similarities..."):
                # Create a new text field for embedding using the selected columns
                src_df["semantic_text"] = create_text_field(src_df, selected_cols)
                tgt_df["semantic_text"] = create_text_field(tgt_df, selected_cols)

                # Load the pre-trained MiniLM model and encode the text fields
                model = SentenceTransformer("all-MiniLM-L6-v2")
                src_embeddings = model.encode(src_df["semantic_text"].tolist(), show_progress_bar=True)
                tgt_embeddings = model.encode(tgt_df["semantic_text"].tolist(), show_progress_bar=True)

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

            st.success("Mapping complete!")
            st.dataframe(mapping_df)

            # Allow the user to download the results as a CSV
            csv_data = mapping_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Mapping as CSV",
                data=csv_data,
                file_name="mapped_urls.csv",
                mime="text/csv"
            )
