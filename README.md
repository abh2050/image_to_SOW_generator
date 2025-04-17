# Industrial SOW Generator (Image Annotation)
![](https://aquilacommercial.com/wp-content/uploads/2018/03/project-manager-definition.jpg)

A Streamlit application that allows users to upload one or more industrial equipment images, automatically detect and manually annotate regions of interest, add contextual notes, and generate a single cohesive Scope of Work (SOW) document (viewable in-app and downloadable as PDF).

---

## Features

- **Image Upload & Management**: Upload multiple PNG/JPG images and maintain a list of saved entries.
- **Auto-Detection**: Leverage Landing AI’s Agentic Object Detection API to propose bounding boxes and labels.
- **Manual Annotation**: Draw transparent, red-outlined rectangles directly on images to highlight areas of interest.
- **Annotation Descriptions**: Provide custom text for each drawn region.
- **Notes Section**: Add free-form contextual or project-specific notes.
- **Unified SOW Generation**: Create one combined prompt from all images, annotations, and notes, then send to Google Generative AI for SOW content.
- **In-App Preview**: Display generated SOW text without leaving the app.
- **PDF Export**: Download a single PDF containing the SOW, reference images with overlaid annotations, and annotation details.

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/abh2050/image_to_SOW_generator.git
   cd industrial-sow-generator
   ```

2. **Create and activate a Python virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate    # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   Create a `.env` file in the project root with the following keys:
   ```ini
   GOOGLE_API_KEY=your_google_generative_ai_key
   LANDING_API_KEY=your_landing_ai_basic_auth_token
   ```

---

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

1. **Upload & Annotate**  
   - In the left pane, upload an image.  
   - Optionally, enter a prompt and click **Auto-detect Annotations** to call Landing AI.  
   - Or draw rectangles manually (transparent fill, red border).  
   - Enter descriptions for each box.  
   - Click **Add Current Image** to save it and repeat for additional images.

2. **Generate SOW**  
   - In the right pane, enter any additional project notes.  
   - Click **Show SOW** to send the combined prompt to Google AI and preview the text.  
   - Review the generated SOW in the embedded text area.

3. **Download PDF**  
   - Once satisfied, click **Download SOW PDF** to obtain a PDF containing:
     - The generated Scope of Work text.
     - Each reference image with overlaid annotations.
     - A list of annotation descriptions.

---

## File Structure

```
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── README.md            # This documentation
└── .env.example         # Sample environment variable file
```

---

## Environment Variables

- `GOOGLE_API_KEY`: API key for Google Generative AI (Gemini Flash).  
- `LANDING_API_KEY`: Basic auth token for Landing AI’s Agentic Object Detection endpoint.

---

## Customization

- **Model Selection**: Change the Google model identifier in `initialize_genai()` (default: `gemini-1.5-flash`).  
- **PDF Layout**: Modify styles or margins in `generate_sow_pdf_multi()`.  
- **Canvas Settings**: Adjust default stroke width or color in the `st_canvas` call.

---

## Troubleshooting

- **ERROR: Google AI not initialized**  
  Ensure `GOOGLE_API_KEY` is correctly set in `.env` and the key is valid.

- **Detection fails**  
  Verify `LANDING_API_KEY` format (`Basic <base64>`), network connectivity, and API usage limits.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

