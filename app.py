import os
import io
import time
import base64
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai
from dotenv import load_dotenv
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter
from io import BytesIO
from streamlit_drawable_canvas import st_canvas
import warnings

# Suppress Streamlit warning: `label got an empty value.`
warnings.filterwarnings("ignore", ".*label got an empty value.*")

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(page_title="Industrial SOW Generator (Image Annotation)", layout="wide")

# API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LANDING_API_KEY = os.getenv("LANDING_API_KEY")

# Initialize session state defaults
def reset_defaults():
    return {
        'annotations': [],
        'google_api_key_configured': False,
        'genai_model': None,
        'current_image_obj': None,
        'current_image_name': 'N/A',
        'detect_prompt': '',
        'uploaded_file_data': None, # Store file data instead of object directly for canvas
        'image_annotation_list': [],
        'sow_text': None
    }
for key, default in reset_defaults().items():
    if key not in st.session_state:
        st.session_state[key] = default

# Initialize Google Generative AI
def initialize_genai(api_key):
    if not api_key:
        st.error("GOOGLE_API_KEY not found in .env file.")
        return
    try:
        genai.configure(api_key=api_key)
        st.session_state.genai_model = genai.GenerativeModel("gemini-1.5-flash")
        st.session_state.google_api_key_configured = True
    except Exception as e:
        st.error(f"Error initializing Google Generative AI: {e}")

# Load uploaded image
def load_image(file_or_bytes):
    try:
        img = Image.open(file_or_bytes)
        return img.convert("RGB") if img.mode != "RGB" else img
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

# Create combined prompt for multiple images
def create_combined_prompt(image_list, notes):
    prompt = "Generate a detailed Scope of Work (SOW) for an industrial project based on the following images and annotations:\n\n"
    for entry in image_list:
        prompt += f"Image: {entry['name']}\nAnnotations:\n"
        if entry['annotations']:
            for i, ann in enumerate(entry['annotations']):
                # Ensure text exists and handle potential None or missing key
                text = ann.get('text', 'No description')
                prompt += f"- Annotation {i+1}: {text if text else 'No description'}\n"
        else:
            prompt += "- No annotations provided.\n"
        prompt += "\n"
    prompt += f"Additional Notes:\n{notes if notes else 'None'}\n\n"
    prompt += (
        "Instructions for SOW Generation:\n"
        "- Integrate all images and annotations into one cohesive SOW.\n"
        "- Use clear headings: Task Description, Safety, Materials.\n"
        "- Be comprehensive and precise based only on provided inputs.\n"
        "Generate the Scope of Work below:\n"
    )
    return prompt

# Generate SOW text via Google AI
def generate_scope_of_work(prompt):
    model = st.session_state.genai_model
    if not model:
        st.error("Google AI is not initialized.")
        return ""
    try:
        # Prepare content for multi-modal input if needed (though here we only use text prompt)
        # For this specific prompt structure, simple text generation is likely sufficient.
        # If you were passing images directly to Gemini, the format would differ.
        resp = model.generate_content(prompt)

        # Handle potential variations in response structure
        if hasattr(resp, 'text'):
            return resp.text
        elif hasattr(resp, 'parts') and resp.parts:
            return ''.join(part.text for part in resp.parts if hasattr(part, 'text'))
        elif isinstance(resp, str): # Fallback if it's just a string
             return resp
        else:
             # Attempt to convert any remaining response type to string, log warning
             st.warning(f"Unexpected response type from Gemini: {type(resp)}. Attempting conversion.")
             return str(resp)

    except Exception as e:
        st.error(f"Error generating SOW with Google AI: {e}")
        return ""


# Generate PDF for multiple images
def generate_sow_pdf_multi(sow_text, image_list):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    story = [Paragraph("Generated Scope of Work", styles['h1']), Spacer(1, 0.2*inch)]

    # Clean up SOW text for ReportLab Paragraph
    sow_text_cleaned = sow_text.replace('\n', '<br/>').replace('\t', '&nbsp;'*4)
    story.append(Paragraph(sow_text_cleaned, styles['Normal']))
    story.append(Spacer(1, 0.3*inch))

    for entry in image_list:
        img = entry['image'].copy() # Use the PIL image stored in the list
        anns = entry['annotations']
        draw = ImageDraw.Draw(img)
        font = None
        try:
            # Try to load a common font, fall back to default
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            st.warning("Arial font not found, using default font for PDF annotations.")
            font = ImageFont.load_default()

        for i, ann in enumerate(anns):
            r = ann.get('rect')
            if r and all(k in r for k in ['left', 'top', 'width', 'height']):
                try:
                    left, top, width, height = r['left'], r['top'], r['width'], r['height']
                    draw.rectangle([left, top, left + width, top + height], outline="red", width=3)
                    pos = (left, top - 20 if top > 20 else top + height + 5)
                    text = ann.get('text', '')
                    draw.text(pos, f"{i+1}. {text}", fill="red", font=font)
                except Exception as e:
                    st.warning(f"Could not draw annotation {i+1} for image {entry['name']}: {e}")
            else:
                st.warning(f"Skipping invalid annotation {i+1} for image {entry['name']}: Missing rect data.")

        buf = BytesIO()
        try:
            img.save(buf, format="PNG")
            buf.seek(0)
            rl_img = RLImage(buf)

            # Scale image to fit page width
            max_w = doc.width
            iw, ih = rl_img.imageWidth, rl_img.imageHeight # Use imageWidth/Height for RLImage
            ratio = ih / iw if iw else 1
            rl_img.drawWidth = min(iw, max_w)
            rl_img.drawHeight = rl_img.drawWidth * ratio

            story.extend([Paragraph(f"Image: {entry['name']}", styles['h2']), Spacer(1,0.1*inch), rl_img, Spacer(1,0.2*inch)])

            if anns:
                story.append(Paragraph("Annotations:", styles['h3']))
                for i, ann in enumerate(anns):
                     text = ann.get('text', 'No description')
                     story.extend([Paragraph(f"{i+1}. {text}", styles['Normal']), Spacer(1,0.05*inch)])
            story.append(Spacer(1,0.3*inch))

        except Exception as e:
            st.error(f"Error processing image {entry['name']} for PDF: {e}")
            story.extend([Paragraph(f"Image: {entry['name']} (Error processing)", styles['h2']), Spacer(1, 0.2*inch)])


    try:
        doc.build(story)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Error building PDF document: {e}")
        # Return an empty buffer or handle error appropriately
        return BytesIO()


# Initialize AI
initialize_genai(GOOGLE_API_KEY)

# App UI
st.title("🏭 Industrial SOW Generator (Image Annotation)")
st.markdown("Upload images, annotate (auto or manual), add notes, then generate a single SOW for all images.")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    st.write(f"Google AI Initialized: {st.session_state.google_api_key_configured}")
    st.write(f"Landing AI Key Found: {bool(LANDING_API_KEY)}")
    if st.button("Reset Session"):
        st.session_state.update(reset_defaults())
        st.rerun()


col1, col2 = st.columns([3,2])

with col1:
    st.header("1. Image Upload & Annotation")
    uploaded = st.file_uploader("Upload an image", type=["png","jpg","jpeg"], key="file_uploader")

    # Handle new upload: Load image, store data, reset annotations
    if uploaded is not None:
        # Check if it's a new file by comparing names or data (simple check here)
        if uploaded.name != st.session_state.current_image_name:
            img_bytes = uploaded.getvalue() # Read bytes once
            img = load_image(io.BytesIO(img_bytes)) # Load from bytes
            if img:
                st.session_state.current_image_obj = img
                st.session_state.current_image_name = uploaded.name
                st.session_state.annotations = [] # Reset annotations for new image
                st.session_state.uploaded_file_data = img_bytes # Store bytes for canvas
                # Clear previous SOW text when a new image is uploaded and processed
                st.session_state.sow_text = None
                # Explicitly reset canvas related state if needed, though key change might handle it
                # st.session_state.canvas = None # Or similar if needed
            else:
                # If loading fails, reset relevant states
                st.session_state.current_image_obj = None
                st.session_state.current_image_name = 'N/A'
                st.session_state.uploaded_file_data = None

    # Display and annotation section - only if an image is loaded
    if st.session_state.current_image_obj:
        st.image(st.session_state.current_image_obj, caption=st.session_state.current_image_name, use_column_width=True)

        st.text_input("Auto-detection prompt (optional)", key="detect_prompt")
        if st.button("Auto-detect Annotations"):
            if not LANDING_API_KEY:
                st.error("LANDING_API_KEY not set in environment variables.")
            elif not st.session_state.current_image_obj:
                 st.warning("Please upload an image first.")
            else:
                with st.spinner("Detecting objects..."):
                    try:
                        # Use the stored image object for detection
                        buf = BytesIO()
                        st.session_state.current_image_obj.save(buf, format="PNG")
                        buf.seek(0)

                        api_url = "https://api.va.landing.ai/v1/tools/agentic-object-detection"
                        headers = {
                            "Authorization": f"Basic {LANDING_API_KEY}",
                            "Accept": "application/json" # Good practice to include
                        }
                        files = {"image": (st.session_state.current_image_name, buf, "image/png")}
                        data = {"prompts": st.session_state.detect_prompt or "object", "model": "agentic"} # Provide default prompt

                        r = requests.post(api_url, headers=headers, files=files, data=data)
                        r.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

                        preds = r.json().get('predictions', [])
                        h, w = st.session_state.current_image_obj.height, st.session_state.current_image_obj.width
                        anns = []
                        for det in preds:
                            b = det.get('bounding_box', {})
                            if all(k in b for k in ['x', 'y', 'width', 'height']):
                                left = b['x'] * w
                                top = b['y'] * h
                                width = b['width'] * w
                                height = b['height'] * h
                                label = det.get('label', 'Object')
                                # Use time combined with index for potentially more unique ID
                                ann_id = f"{time.time()}_{len(anns)}"
                                anns.append({'rect': {'left': left, 'top': top, 'width': width, 'height': height}, 'text': label, 'id': ann_id})
                            else:
                                st.warning(f"Skipping detection with invalid bounding box: {b}")
                        st.session_state.annotations = anns # Replace existing annotations
                        st.success(f"Found {len(anns)} annotations.")
                        st.rerun() # Rerun to update annotation descriptions section

                    except requests.exceptions.RequestException as e:
                        st.error(f"API Request Error: {e}")
                    except Exception as e:
                        st.error(f"Annotation detection error: {e}")


        st.subheader("Manual Annotation / Review")
        canvas_result = None
        # Use the stored image bytes for the canvas background
        if st.session_state.uploaded_file_data:
             # Prepare initial JSON data for canvas based on current annotations
            initial_drawing = {"version": "4.4.0", "objects": []}
            img_w = st.session_state.current_image_obj.width
            img_h = st.session_state.current_image_obj.height
            for i, ann in enumerate(st.session_state.annotations):
                 r = ann.get('rect')
                 if r:
                    initial_drawing["objects"].append({
                         "type": "rect", "version": "4.4.0", "originX": "left", "originY": "top",
                         "left": r.get('left', 0), "top": r.get('top', 0),
                         "width": r.get('width', 50), "height": r.get('height', 50),
                         "fill": "rgba(255,0,0,0)", "stroke": "red", "strokeWidth": 3,
                         "strokeDashArray": None, "strokeLineCap": "butt", "strokeDashOffset": 0,
                         "strokeLineJoin": "miter", "strokeUniform": False, "strokeMiterLimit": 4,
                         "scaleX": 1, "scaleY": 1, "angle": 0, "flipX": False, "flipY": False,
                         "opacity": 1, "shadow": None, "visible": True, "backgroundColor": "",
                         "fillRule": "nonzero", "paintFirst": "fill", "globalCompositeOperation": "source-over",
                         "skewX": 0, "skewY": 0, "rx": 0, "ry": 0,
                         # Add custom ID to link back to annotation data
                         "id": ann.get('id', f"manual_{i}_{time.time()}")
                    })


            try:
                # Create a unique key based on the image name to reset canvas on image change
                canvas_key = f"canvas_{st.session_state.current_image_name}"
                canvas_result = st_canvas(
                    drawing_mode="rect", # Allow drawing new rectangles
                    stroke_width=3,
                    stroke_color="red",
                    fill_color="rgba(255,0,0,0.1)", # Slight fill for visibility
                    # *** CORRECTED ARGUMENT ***
                    background_image=st.session_state.current_image_obj,
                    initial_drawing=initial_drawing, # Load existing annotations
                    update_streamlit=True, # Important for interaction
                    height=img_h,
                    width=img_w,
                    key=canvas_key # Force re-render on image change
                )
            except Exception as e:
                st.warning(f"Could not load annotation canvas: {e}")
        else:
            st.info("Upload an image to enable manual annotation.")

        # Process canvas results if available
        if canvas_result and canvas_result.json_data and 'objects' in canvas_result.json_data:
            new_annotations = []
            drawn_objects = canvas_result.json_data['objects']

            # Create a map of existing annotations by ID for quick lookup
            existing_anns_map = {ann['id']: ann for ann in st.session_state.annotations}

            for obj in drawn_objects:
                # Use the ID from the canvas object if it exists, otherwise generate new
                ann_id = obj.get('id', f"new_{time.time()}_{len(new_annotations)}")
                rect = {
                    'left': obj.get('left'), 'top': obj.get('top'),
                    'width': obj.get('width'), 'height': obj.get('height')
                }

                # Check if all rect values are valid numbers
                if not all(isinstance(v, (int, float)) for v in rect.values()):
                    st.warning(f"Skipping canvas object with invalid geometry: {rect}")
                    continue

                # Preserve text if annotation already existed, otherwise default
                existing_text = existing_anns_map.get(ann_id, {}).get('text', f'Annotation {len(new_annotations) + 1}')

                new_annotations.append({
                    'rect': rect,
                    'text': existing_text,
                    'id': ann_id # Ensure ID is consistent
                })

            # Only update if there's a change to avoid infinite loops
            if new_annotations != st.session_state.annotations:
                 st.session_state.annotations = new_annotations
                 st.rerun() # Rerun to update description fields below


        # Display text input fields for annotation descriptions
        st.subheader("Annotation Descriptions")
        # Create a temporary list for modifications to avoid issues while iterating
        current_annotations = list(st.session_state.annotations)
        annotations_changed = False
        for i, ann in enumerate(current_annotations):
             ann_id = ann.get('id', f"desc_{i}") # Ensure stable key
             # Use ann['text'] or provide a default value if missing/None
             default_text = ann.get('text') or f"Annotation {i+1}"
             new_text = st.text_input(f"Box {i+1}", value=default_text, key=f"txt_{ann_id}")
             # Update the text in the original list in session state
             if st.session_state.annotations[i]['text'] != new_text:
                 st.session_state.annotations[i]['text'] = new_text
                 annotations_changed = True # Flag that a change occurred

        # ----- Add Image to List Button -----
        if st.button("✅ Add Current Image to SOW List"):
            if st.session_state.current_image_obj and st.session_state.current_image_name != 'N/A':
                st.session_state.image_annotation_list.append({
                    'image': st.session_state.current_image_obj.copy(), # Store a copy of the PIL image
                    'name': st.session_state.current_image_name,
                    'annotations': list(st.session_state.annotations) # Store a copy of annotations
                })
                st.success(f"Image '{st.session_state.current_image_name}' added to list.")

                # Reset current image state to prepare for next upload
                st.session_state.current_image_obj = None
                st.session_state.current_image_name = 'N/A'
                st.session_state.annotations = []
                st.session_state.uploaded_file_data = None
                st.session_state.detect_prompt = "" # Clear prompt too
                # Clear the file uploader widget state using its key
                st.session_state.file_uploader = None
                st.rerun() # Rerun to clear the displayed image and canvas
            else:
                st.warning("No current image loaded to add.")

    # ----- Display List of Added Images -----
    if st.session_state.image_annotation_list:
        st.markdown("---")
        st.markdown("#### Images Added for SOW Generation")
        for idx, entry in enumerate(st.session_state.image_annotation_list):
            expander_title = f"{idx+1}. {entry['name']} ({len(entry.get('annotations', []))} annotations)"
            with st.expander(expander_title):
                st.image(entry['image'], width=150) # Show small thumbnail
                if entry.get('annotations'):
                    for ann_idx, ann in enumerate(entry['annotations']):
                         st.write(f"- Annotation {ann_idx+1}: {ann.get('text', 'No description')}")
                else:
                    st.write("No annotations for this image.")
                # Add a button to remove the image from the list
                if st.button(f"❌ Remove Image {idx+1}", key=f"remove_{idx}_{entry['name']}"):
                    st.session_state.image_annotation_list.pop(idx)
                    st.rerun()


# Right Column: SOW Generation
with col2:
    st.header("2. Generate SOW")
    notes = st.text_area("Additional Notes (General)", height=150, key="notes")

    if st.button("📝 Generate SOW Document", disabled=not st.session_state.image_annotation_list):
        if not st.session_state.image_annotation_list:
            st.warning("Please add at least one annotated image to the list first.")
        elif not st.session_state.google_api_key_configured:
            st.error("Google AI is not configured. Please check your API key.")
        else:
            with st.spinner("Generating Scope of Work..."):
                # Create the prompt based on the list of images/annotations
                prompt = create_combined_prompt(st.session_state.image_annotation_list, st.session_state.notes)
                # Generate the SOW text
                st.session_state.sow_text = generate_scope_of_work(prompt)
                if not st.session_state.sow_text:
                     st.error("SOW generation failed. Please check the logs or try again.")


    if st.session_state.sow_text:
        st.subheader("Generated Scope of Work")
        # Display the generated SOW text
        st.text_area("SOW Content", st.session_state.sow_text, height=400, key="sow_display")

        # Generate the PDF
        pdf_buffer = generate_sow_pdf_multi(st.session_state.sow_text, st.session_state.image_annotation_list)

        if pdf_buffer.getbuffer().nbytes > 0: # Check if PDF generation was successful
            st.download_button(
                label="⬇️ Download SOW PDF",
                data=pdf_buffer,
                file_name=f"scope_of_work_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
            st.success("SOW generated and PDF is ready for download.")
        else:
            st.error("Failed to generate PDF document.")
    elif not st.session_state.image_annotation_list:
         st.info("Add images using the '✅ Add Current Image to SOW List' button in the left panel before generating the SOW.")
