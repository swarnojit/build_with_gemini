import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Increase PIL decompression bomb limit for large drone images
Image.MAX_IMAGE_PIXELS = None  # Remove limit entirely

# Configure page
st.set_page_config(
    page_title="Afforestation Monitoring System",
    page_icon="🌳",
    layout="wide"
)

# Title and description
st.title("🌳 Afforestation Sapling Survival Analysis")
st.markdown("""
This system uses **Gemini 2.5 Flash** to analyze drone imagery and detect sapling survival 
in afforestation patches with intelligent image compression and chunking.

**Note:** Large drone images are automatically handled and compressed for processing.
""")

# Sidebar for API key and configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Get API key from environment variable
    api_key = os.getenv("GEMINI_API_KEY")
    
    if api_key:
        genai.configure(api_key=api_key)
        st.success("✅ API Key loaded from .env file")
    else:
        st.error("❌ GEMINI_API_KEY not found in .env file")
        st.info("Please add GEMINI_API_KEY=your_key_here to your .env file")
    
    st.markdown("---")
    st.subheader("🖼️ Image Processing")
    
    # Compression settings
    max_dimension = st.slider("Max Image Dimension (px)", 512, 4096, 3072, 128, 
                             help="Larger = more detail but slower processing. Your images are very large, so compression is essential.")
    compression_quality = st.slider("JPEG Quality", 50, 100, 85, 5,
                                   help="Higher = better quality but larger file")
    
    # Chunking settings
    enable_chunking = st.checkbox("Enable Image Chunking", value=True,
                                 help="Split large images into smaller chunks for detailed analysis")
    
    if enable_chunking:
        chunk_size = st.slider("Chunk Size", 512, 2048, 1024, 128,
                              help="Size of each chunk in pixels")
        overlap = st.slider("Chunk Overlap", 0, 50, 10, 5,
                           help="Overlap percentage between chunks")
    
    st.markdown("---")
    st.subheader("📁 Direct File Paths")
    
    # Show current working directory
    st.caption(f"Current directory: {os.getcwd()}")
    
    # Direct file path input
    st.markdown("**Enter Full File Paths:**")
    
    op1_path = st.text_input(
        "OP1 Full Path", 
        value=r"C:\Users\maitr\Documents\AI HACKATHON\Debadihi VF-20260117T193559Z-1-001\Debadihi VF\Ortho Data\Post-Pitting\Post-Pitting.tif"
    )
    
    op3_path = st.text_input(
        "OP3 Full Path", 
        value=r"C:\Users\maitr\Documents\AI HACKATHON\Debadihi VF-20260117T193559Z-1-002\Debadihi VF\Ortho Data\Post-SW\map.tif"
    )
    
    st.markdown("---")
    st.subheader("📊 Patch Information")
    patch_name = st.text_input("Patch Name", "Debadihi VF")
    patch_area = st.number_input("Area (hectares)", value=6.25, step=0.01)
    total_saplings = st.number_input("Total Saplings Planted", value=10000, step=100)
    spacing = st.number_input("Spacing (meters)", value=2.5, step=0.1)

# Function to compress image
def compress_image(image, max_dimension=2048, quality=85):
    """Compress and resize image for API upload"""
    # Get original dimensions
    orig_width, orig_height = image.size
    
    # Calculate new dimensions maintaining aspect ratio
    if orig_width > max_dimension or orig_height > max_dimension:
        if orig_width > orig_height:
            new_width = max_dimension
            new_height = int((max_dimension / orig_width) * orig_height)
        else:
            new_height = max_dimension
            new_width = int((max_dimension / orig_height) * orig_width)
        
        image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Compress to JPEG
    buffer = io.BytesIO()
    image.convert('RGB').save(buffer, format='JPEG', quality=quality, optimize=True)
    buffer.seek(0)
    
    compressed_image = Image.open(buffer)
    
    return compressed_image, image.size

# Function to split image into chunks
def create_image_chunks(image, chunk_size=1024, overlap_percent=10):
    """Split image into overlapping chunks for detailed analysis"""
    width, height = image.size
    overlap = int(chunk_size * overlap_percent / 100)
    step = chunk_size - overlap
    
    chunks = []
    chunk_info = []
    
    y = 0
    row = 0
    while y < height:
        x = 0
        col = 0
        while x < width:
            # Calculate chunk boundaries
            x_end = min(x + chunk_size, width)
            y_end = min(y + chunk_size, height)
            
            # Extract chunk
            chunk = image.crop((x, y, x_end, y_end))
            chunks.append(chunk)
            chunk_info.append({
                'row': row,
                'col': col,
                'x': x,
                'y': y,
                'width': x_end - x,
                'height': y_end - y
            })
            
            x += step
            col += 1
            if x >= width:
                break
        
        y += step
        row += 1
        if y >= height:
            break
    
    return chunks, chunk_info

# Function to load image from path
def load_image_from_path(file_path):
    """Load image from the file system with large image support"""
    try:
        if os.path.exists(file_path):
            # Open image with PIL - large image support enabled via MAX_IMAGE_PIXELS = None
            img = Image.open(file_path)
            # Load the image data
            img.load()
            return img, file_path
        else:
            return None, None
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None, None

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["📸 Image Analysis", "📈 Results Dashboard", "📁 File Browser", "ℹ️ About"])

with tab1:
    st.header("Automated Analysis from Project Files")
    
    # Display selected paths
    st.markdown("### 📂 Selected Files")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**OP1**: {op1_path}")
    
    with col2:
        st.info(f"**OP3**: {op3_path}")
    
    st.markdown("---")
    
    # Load and display images
    col1, col2 = st.columns(2)
    
    op1_img = None
    op3_img = None
    
    with col1:
        st.subheader("OP1 Image (Reference)")
        with st.spinner("Loading OP1 image (this may take a moment for large files)..."):
            if os.path.exists(op1_path):
                try:
                    op1_img = Image.open(op1_path)
                    op1_img.load()  # Force load the image
                    st.success(f"✅ Loaded successfully")
                    st.caption(f"Original size: {op1_img.size[0]:,}x{op1_img.size[1]:,} px ({op1_img.size[0] * op1_img.size[1] / 1_000_000:.1f}M pixels)")
                    st.caption(f"File: {os.path.basename(op1_path)}")
                    
                    # Create a small preview for display
                    preview_img = op1_img.copy()
                    preview_img.thumbnail((800, 800), Image.LANCZOS)
                    st.image(preview_img, caption="OP1 Preview (downsampled for display)", use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Error loading: {e}")
                    st.info("Try reducing the max dimension in the sidebar settings.")
            else:
                st.error(f"❌ File not found: {op1_path}")
    
    with col2:
        st.subheader("OP3 Image (Current)")
        with st.spinner("Loading OP3 image (this may take a moment for large files)..."):
            if os.path.exists(op3_path):
                try:
                    op3_img = Image.open(op3_path)
                    op3_img.load()  # Force load the image
                    st.success(f"✅ Loaded successfully")
                    st.caption(f"Original size: {op3_img.size[0]:,}x{op3_img.size[1]:,} px ({op3_img.size[0] * op3_img.size[1] / 1_000_000:.1f}M pixels)")
                    st.caption(f"File: {os.path.basename(op3_path)}")
                    
                    # Create a small preview for display
                    preview_img = op3_img.copy()
                    preview_img.thumbnail((800, 800), Image.LANCZOS)
                    st.image(preview_img, caption="OP3 Preview (downsampled for display)", use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Error loading: {e}")
                    st.info("Try reducing the max dimension in the sidebar settings.")
            else:
                st.error(f"❌ File not found: {op3_path}")
    
    st.markdown("---")
    
    # Analysis button
    if st.button("🔍 Analyze Sapling Survival", type="primary", use_container_width=True):
        if not api_key:
            st.error("⚠️ Please add GEMINI_API_KEY to your .env file")
        elif not op1_img or not op3_img:
            st.error("⚠️ Could not load images. Please check file paths.")
        else:
            start_time = time.time()
            
            with st.spinner("Processing images..."):
                # Initialize results storage
                all_results = []
                
                # Step 1: Compress images
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Compressing OP1 image...")
                compressed_op1, new_size_op1 = compress_image(op1_img, max_dimension, compression_quality)
                st.info(f"OP1 compressed: {op1_img.size} → {new_size_op1}")
                progress_bar.progress(10)
                
                status_text.text("Compressing OP3 image...")
                compressed_op3, new_size_op3 = compress_image(op3_img, max_dimension, compression_quality)
                st.info(f"OP3 compressed: {op3_img.size} → {new_size_op3}")
                progress_bar.progress(20)
                
                try:
                    # Initialize Gemini model (using 2.5 Flash)
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    
                    if enable_chunking:
                        # Step 2: Create chunks
                        status_text.text("Creating image chunks...")
                        op1_chunks, op1_chunk_info = create_image_chunks(compressed_op1, chunk_size, overlap)
                        op3_chunks, op3_chunk_info = create_image_chunks(compressed_op3, chunk_size, overlap)
                        
                        st.info(f"Created {len(op1_chunks)} chunks for OP1 and {len(op3_chunks)} chunks for OP3")
                        progress_bar.progress(30)
                        
                        # Step 3: Analyze chunks
                        total_chunks = min(len(op1_chunks), len(op3_chunks))
                        
                        for i in range(total_chunks):
                            status_text.text(f"Analyzing chunk {i+1}/{total_chunks}...")
                            
                            chunk_prompt = f"""Analyze this pair of corresponding image chunks from an afforestation monitoring project.

**Chunk Location**: Row {op1_chunk_info[i]['row']}, Column {op1_chunk_info[i]['col']}
**Coordinates**: x={op1_chunk_info[i]['x']}, y={op1_chunk_info[i]['y']}
**Chunk Size**: {op1_chunk_info[i]['width']}x{op1_chunk_info[i]['height']} pixels

**Image 1 (OP1)**: Shows pits after digging (45x45x45cm), spaced {spacing}m apart
**Image 2 (OP3)**: Shows the same area after planting and weeding

Count and identify:
1. Number of pits visible in OP1 chunk
2. Number of surviving saplings in OP3 chunk (with cleared 1m diameter)
3. Number of casualties (pits without saplings)

Respond in JSON format:
{{
    "chunk_id": {i},
    "pits_detected": <number>,
    "surviving_saplings": <number>,
    "casualties": <number>,
    "observations": "<brief notes>"
}}"""

                            try:
                                response = model.generate_content([chunk_prompt, op1_chunks[i], op3_chunks[i]])
                                
                                # Parse response
                                result_text = response.text
                                if "```json" in result_text:
                                    result_text = result_text.split("```json")[1].split("```")[0]
                                elif "```" in result_text:
                                    result_text = result_text.split("```")[1].split("```")[0]
                                
                                chunk_result = json.loads(result_text.strip())
                                chunk_result['chunk_info'] = op1_chunk_info[i]
                                all_results.append(chunk_result)
                                
                            except Exception as e:
                                st.warning(f"Error analyzing chunk {i+1}: {e}")
                            
                            progress_bar.progress(30 + int(60 * (i + 1) / total_chunks))
                        
                        # Step 4: Aggregate results
                        status_text.text("Aggregating chunk results...")
                        
                        total_pits = sum(r.get('pits_detected', 0) for r in all_results)
                        total_surviving = sum(r.get('surviving_saplings', 0) for r in all_results)
                        total_casualties = sum(r.get('casualties', 0) for r in all_results)
                        
                        survival_pct = (total_surviving / total_pits * 100) if total_pits > 0 else 0
                        
                        # Create final result
                        final_result = {
                            "total_pits_detected": total_pits,
                            "surviving_saplings": total_surviving,
                            "casualties": total_casualties,
                            "survival_percentage": round(survival_pct, 2),
                            "mortality_percentage": round(100 - survival_pct, 2),
                            "chunks_analyzed": len(all_results),
                            "chunk_results": all_results,
                            "confidence_level": "high" if len(all_results) > 5 else "medium",
                            "analysis_notes": f"Analyzed {len(all_results)} chunks with {chunk_size}x{chunk_size}px resolution",
                            "processing_method": "chunked_analysis"
                        }
                        
                        progress_bar.progress(90)
                        
                    else:
                        # Single image analysis (no chunking)
                        status_text.text("Analyzing full images...")
                        
                        prompt = f"""You are an expert in analyzing drone imagery for afforestation monitoring. 

Analyze these two compressed drone orthomosaic images from {patch_name}:

**Image 1 (OP1)**: Shows pits (45x45x45cm) after digging, spaced {spacing}m apart
**Image 2 (OP3)**: Shows saplings after planting and weeding (1m cleared diameter)

**Patch Details**:
- Area: {patch_area} hectares
- Expected saplings: {total_saplings}
- Spacing: {spacing} meters

Provide realistic estimates based on visible features. Respond in JSON:
{{
    "total_pits_detected": <number>,
    "surviving_saplings": <number>,
    "casualties": <number>,
    "survival_percentage": <percentage>,
    "mortality_percentage": <percentage>,
    "confidence_level": "<high/medium/low>",
    "analysis_notes": "<observations>",
    "processing_method": "full_image_analysis"
}}"""

                        response = model.generate_content([prompt, compressed_op1, compressed_op3])
                        
                        result_text = response.text
                        if "```json" in result_text:
                            result_text = result_text.split("```json")[1].split("```")[0]
                        elif "```" in result_text:
                            result_text = result_text.split("```")[1].split("```")[0]
                        
                        final_result = json.loads(result_text.strip())
                        progress_bar.progress(90)
                    
                    # Store results
                    processing_time = time.time() - start_time
                    final_result['processing_time_seconds'] = round(processing_time, 2)
                    
                    st.session_state.analysis_result = json.dumps(final_result, indent=2)
                    st.session_state.analysis_json = final_result
                    st.session_state.analysis_timestamp = datetime.now()
                    st.session_state.op1_path = op1_path
                    st.session_state.op3_path = op3_path
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    
                    st.success(f"✅ Analysis completed in {processing_time:.2f} seconds!")
                    time.sleep(1)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.exception(e)
    
    # Display results if available
    if 'analysis_result' in st.session_state:
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        
        # Show analyzed files
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"OP1: {os.path.basename(st.session_state.get('op1_path', 'N/A'))}")
        with col2:
            st.caption(f"OP3: {os.path.basename(st.session_state.get('op3_path', 'N/A'))}")
        
        if st.session_state.get('analysis_json'):
            result = st.session_state.analysis_json
            
            # Processing info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Processing Time", f"{result.get('processing_time_seconds', 0)}s")
            with col2:
                st.metric("Method", result.get('processing_method', 'N/A'))
            with col3:
                if result.get('chunks_analyzed'):
                    st.metric("Chunks Analyzed", result.get('chunks_analyzed', 0))
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Pits Detected", result.get('total_pits_detected', 'N/A'))
            with col2:
                st.metric("Surviving Saplings", result.get('surviving_saplings', 'N/A'))
            with col3:
                st.metric("Casualties", result.get('casualties', 'N/A'))
            with col4:
                survival_pct = result.get('survival_percentage', 0)
                st.metric("Survival Rate", f"{survival_pct}%")
            
            # Progress bar
            st.progress(survival_pct / 100)
            
            # Detailed information
            st.markdown("### 📝 Detailed Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Confidence Level**")
                confidence = result.get('confidence_level', 'Unknown')
                if confidence.lower() == 'high':
                    st.success(f"🟢 {confidence}")
                elif confidence.lower() == 'medium':
                    st.warning(f"🟡 {confidence}")
                else:
                    st.error(f"🔴 {confidence}")
            
            with col2:
                st.markdown("**Mortality Rate**")
                mortality = result.get('mortality_percentage', 100 - survival_pct)
                st.metric("Deaths", f"{mortality}%")
            
            # Analysis notes
            st.markdown("**Analysis Notes**")
            st.info(result.get('analysis_notes', 'No notes available'))
            
            # Chunk results (if available)
            if result.get('chunk_results'):
                with st.expander("🔍 View Chunk-by-Chunk Results"):
                    chunk_df = pd.DataFrame(result['chunk_results'])
                    st.dataframe(chunk_df, use_container_width=True)
        
        # Raw response
        with st.expander("🔍 View Raw JSON Response"):
            st.code(st.session_state.analysis_result, language="json")

with tab2:
    st.header("Results Dashboard")
    
    if 'analysis_json' in st.session_state and st.session_state.analysis_json:
        result = st.session_state.analysis_json
        
        # Summary statistics
        st.subheader("📊 Summary Statistics")
        st.caption(f"Analysis performed: {st.session_state.get('analysis_timestamp', 'N/A')}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Create chart data
            survival_data = pd.DataFrame({
                'Status': ['Surviving', 'Casualties'],
                'Count': [
                    result.get('surviving_saplings', 0),
                    result.get('casualties', 0)
                ]
            })
            st.markdown("**Sapling Status Distribution**")
            st.bar_chart(survival_data.set_index('Status'))
        
        with col2:
            # Statistics table
            stats_data = {
                'Metric': [
                    'Total Planted',
                    'Pits Detected',
                    'Surviving Saplings',
                    'Casualties',
                    'Survival Rate',
                    'Mortality Rate',
                    'Processing Time'
                ],
                'Value': [
                    total_saplings,
                    result.get('total_pits_detected', 'N/A'),
                    result.get('surviving_saplings', 'N/A'),
                    result.get('casualties', 'N/A'),
                    f"{result.get('survival_percentage', 0)}%",
                    f"{result.get('mortality_percentage', 100 - result.get('survival_percentage', 0))}%",
                    f"{result.get('processing_time_seconds', 0)}s"
                ]
            }
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
        
        # Performance metrics
        st.markdown("---")
        st.subheader("🎯 Performance Assessment")
        
        survival_rate = result.get('survival_percentage', 0)
        
        if survival_rate >= 80:
            st.success(f"✅ Excellent survival rate! ({survival_rate}%)")
        elif survival_rate >= 60:
            st.warning(f"⚠️ Moderate survival rate. ({survival_rate}%)")
        else:
            st.error(f"❌ Low survival rate. Immediate attention needed. ({survival_rate}%)")
        
        # Chunk visualization (if available)
        if result.get('chunk_results'):
            st.markdown("---")
            st.subheader("🗺️ Spatial Analysis (Chunk Map)")
            
            chunk_df = pd.DataFrame(result['chunk_results'])
            if 'chunk_info' in chunk_df.columns:
                # Create a heatmap visualization
                st.markdown("**Survival Rate by Chunk:**")
                chunk_df['survival_rate'] = (chunk_df['surviving_saplings'] / chunk_df['pits_detected'] * 100).fillna(0)
                st.dataframe(chunk_df[['chunk_id', 'pits_detected', 'surviving_saplings', 'casualties', 'survival_rate']], 
                           use_container_width=True)
        
        # Export results
        st.markdown("---")
        st.subheader("💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export as JSON
            json_str = json.dumps(st.session_state.analysis_json, indent=2)
            st.download_button(
                label="📥 Download JSON Report",
                data=json_str,
                file_name=f"afforestation_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            # Export chunk results
            if result.get('chunk_results'):
                chunk_df = pd.DataFrame(result['chunk_results'])
                csv = chunk_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Chunk CSV",
                    data=csv,
                    file_name=f"chunk_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            # Export summary
            summary_text = f"""Afforestation Analysis Report
Generated: {st.session_state.get('analysis_timestamp', 'N/A')}

Patch: {patch_name}
Area: {patch_area} hectares
Expected Saplings: {total_saplings}

Processing:
- Method: {result.get('processing_method', 'N/A')}
- Time: {result.get('processing_time_seconds', 0)}s
- Chunks: {result.get('chunks_analyzed', 'N/A')}

Results:
- Pits Detected: {result.get('total_pits_detected', 'N/A')}
- Surviving: {result.get('surviving_saplings', 'N/A')}
- Casualties: {result.get('casualties', 'N/A')}
- Survival Rate: {result.get('survival_percentage', 0)}%

Confidence: {result.get('confidence_level', 'N/A')}

Notes:
{result.get('analysis_notes', 'N/A')}
"""
            st.download_button(
                label="📥 Download Summary",
                data=summary_text,
                file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    else:
        st.info("📊 No analysis results available yet. Please run an analysis in the Image Analysis tab.")

with tab3:
    st.header("📁 File Browser")
    
    st.markdown("### File Path Verification")
    
    # Check OP1 path
    st.markdown("**OP1 Image:**")
    if os.path.exists(op1_path):
        file_size = os.path.getsize(op1_path) / (1024 * 1024)  # MB
        st.success(f"✅ File exists")
        st.caption(f"   Path: {op1_path}")
        st.caption(f"   Size: {file_size:.2f} MB")
    else:
        st.error(f"❌ File not found: {op1_path}")
    
    st.markdown("**OP3 Image:**")
    if os.path.exists(op3_path):
        file_size = os.path.getsize(op3_path) / (1024 * 1024)  # MB
        st.success(f"✅ File exists")
        st.caption(f"   Path: {op3_path}")
        st.caption(f"   Size: {file_size:.2f} MB")
    else:
        st.error(f"❌ File not found: {op3_path}")

with tab4:
    st.header("ℹ️ About")
    
    st.markdown("""
    ### Afforestation Monitoring System
    
    This application uses Google's Gemini 2.0 Flash model to analyze drone imagery 
    for afforestation monitoring projects.
    
    **Features:**
    - ✅ Automatic API key loading from .env file
    - ✅ Large image support (removes PIL decompression bomb limit)
    - ✅ Image compression and optimization
    - ✅ Chunked analysis for large images
    - ✅ Survival rate calculation
    - ✅ Detailed reporting and export
    - ✅ Uses Gemini 2.5 Flash model
    
    **Setup Instructions:**
    
    1. Create a `.env` file in your project directory
    2. Add your Gemini API key:
       ```
       GEMINI_API_KEY=your_api_key_here
       ```
    3. Update the file paths in the sidebar
    4. Click "Analyze Sapling Survival"
    
    **Required Python packages:**
    ```
    pip install streamlit google-generativeai pillow python-dotenv pandas numpy
    ```
    """)