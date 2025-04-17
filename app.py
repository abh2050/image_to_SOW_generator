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
def load_image(file):
    try:
        img = Image.open(file)
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
                prompt += f"- Annotation {i+1}: {ann.get('text', 'No description')}\n"
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
        resp = model.generate_content(prompt)
        if hasattr(resp, 'text'):
            return resp.text
        if resp.parts:
            return ''.join(part.text for part in resp.parts)
        return str(resp)
    except Exception as e:
        st.error(f"Error generating SOW: {e}")
        return ""

# Generate PDF for multiple images
def generate_sow_pdf_multi(sow_text, image_list):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    story = [Paragraph("Generated Scope of Work", styles['h1']), Spacer(1, 0.2*inch)]
    story.append(Paragraph(sow_text.replace('\n', '<br/>'), styles['Normal']))
    story.append(Spacer(1, 0.3*inch))

    for entry in image_list:
        img = entry['image'].copy()
        anns = entry['annotations']
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()
        for i, ann in enumerate(anns):
            r = ann['rect']
            draw.rectangle([r['left'], r['top'], r['left']+r['width'], r['top']+r['height']], outline="red", width=3)
            pos = (r['left'], r['top']-20 if r['top']>20 else r['top']+r['height']+5)
            draw.text(pos, f"{i+1}. {ann.get('text','')}", fill="red", font=font)
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        rl_img = RLImage(buf)
        max_w = doc.width
        iw, ih = rl_img.drawWidth, rl_img.drawHeight
        ratio = ih/iw if iw else 1
        rl_img.drawWidth = min(iw, max_w)
        rl_img.drawHeight = rl_img.drawWidth * ratio
        story.extend([Paragraph(f"Image: {entry['name']}", styles['h2']), Spacer(1,0.1*inch), rl_img, Spacer(1,0.2*inch)])
        if anns:
            story.append(Paragraph("Annotations:", styles['h3']))
            for i, ann in enumerate(anns):
                story.extend([Paragraph(f"{i+1}. {ann.get('text','')}", styles['Normal']), Spacer(1,0.05*inch)])
        story.append(Spacer(1,0.3*inch))

    doc.build(story)
    buffer.seek(0)
    return buffer

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

col1, col2 = st.columns([3,2])

with col1:
    st.header("1. Image Upload & Annotation")
    uploaded = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
    if uploaded:
        img = load_image(uploaded)
        if img:
            st.session_state.current_image_obj = img
            st.session_state.current_image_name = uploaded.name
            st.session_state.annotations = []

    if st.session_state.current_image_obj:
        st.image(st.session_state.current_image_obj, caption=st.session_state.current_image_name)
        st.text_input("Auto-detection prompt (optional)", key="detect_prompt")
        if st.button("Auto-detect Annotations"):
            if not LANDING_API_KEY:
                st.error("LANDING_API_KEY not set.")
            else:
                buf = BytesIO()
                st.session_state.current_image_obj.save(buf, format="PNG"); buf.seek(0)
                r = requests.post(
                    "https://api.va.landing.ai/v1/tools/agentic-object-detection",
                    headers={"Authorization": f"Basic {LANDING_API_KEY}"},
                    files={"image": buf},
                    data={"prompts": st.session_state.detect_prompt, "model": "agentic"}
                )
                if r.ok:
                    preds = r.json().get('predictions', [])
                    h,w = st.session_state.current_image_obj.height, st.session_state.current_image_obj.width
                    anns=[]
                    for det in preds:
                        b=det.get('bounding_box',{})
                        left, top = b.get('x',0)*w, b.get('y',0)*
