import os
import re
import io
import uuid
import unicodedata
import base64
import tempfile
from flask import Flask, render_template, request, send_file, jsonify
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

app = Flask(__name__)
app.secret_key = 'suangaypdf-secret-key'

# Temp download directory
DOWNLOAD_DIR = os.path.join(tempfile.gettempdir(), 'suangaypdf_downloads')
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# ===== Tesseract Setup =====
import platform

if platform.system() == 'Windows':
    TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    TESSDATA_DIR = os.path.join(os.path.expanduser('~'), 'tessdata')
else:
    # Linux (Render, Docker, etc.)
    TESSERACT_PATH = '/usr/bin/tesseract'
    TESSDATA_DIR = '/usr/share/tesseract-ocr/5/tessdata'
    if not os.path.isdir(TESSDATA_DIR):
        TESSDATA_DIR = '/usr/share/tesseract-ocr/4.00/tessdata'
    if not os.path.isdir(TESSDATA_DIR):
        TESSDATA_DIR = '/usr/share/tessdata'

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
if os.path.isdir(TESSDATA_DIR):
    os.environ['TESSDATA_PREFIX'] = TESSDATA_DIR

# Determine best OCR language
OCR_LANG = 'eng'
if os.path.exists(os.path.join(TESSDATA_DIR, 'vie.traineddata')):
    OCR_LANG = 'vie+eng'

# Text-based PDF regex
DATE_PATTERN = re.compile(
    r'ng[aàả]y\s*(\d{1,2})\s*th[aáả]ng\s*(\d{1,2})\s*n[aăâ]m\s*(\d{4})',
    re.IGNORECASE | re.UNICODE
)

# ===== Helper Functions =====

def pdf_page_to_image(doc, page_num=0, dpi=300):
    page = doc[page_num]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def ocr_find_date(image):
    """Use OCR to find day, month, year numbers in the top portion of the page.
    
    Strategy: Find 3 numbers on the same horizontal line in the top 40% of the page
    where: first=1-2 digits (day), second=1-2 digits (month), third=4 digits (year).
    """
    ocr_data = pytesseract.image_to_data(
        image, lang=OCR_LANG, output_type=pytesseract.Output.DICT
    )
    
    n_boxes = len(ocr_data['text'])
    page_height = image.size[1]
    top_region = page_height * 0.4  # Only look in top 40%
    
    # Collect all words with positions
    all_words = []
    for i in range(n_boxes):
        text = ocr_data['text'][i].strip()
        if text:
            all_words.append({
                'text': text,
                'left': ocr_data['left'][i],
                'top': ocr_data['top'][i],
                'width': ocr_data['width'][i],
                'height': ocr_data['height'][i],
                'conf': int(ocr_data['conf'][i]) if str(ocr_data['conf'][i]) != '-1' else 0
            })
    
    # Collect numbers in the top region
    numbers_top = [w for w in all_words if w['top'] < top_region and w['text'].isdigit()]
    
    # Find triplets of numbers on the same line (day, month, year)
    best_match = None
    y_tolerance = 30  # pixels tolerance for "same line"
    
    for i, n1 in enumerate(numbers_top):
        for j, n2 in enumerate(numbers_top):
            if j <= i:
                continue
            for k, n3 in enumerate(numbers_top):
                if k <= j:
                    continue
                
                # Check same line
                tops = [n1['top'], n2['top'], n3['top']]
                if max(tops) - min(tops) > y_tolerance:
                    continue
                
                # Ensure left-to-right order
                if not (n1['left'] < n2['left'] < n3['left']):
                    continue
                
                d = int(n1['text'])
                m = int(n2['text'])
                y = int(n3['text'])
                
                if 1 <= d <= 31 and 1 <= m <= 12 and 2000 <= y <= 2099:
                    # Find max height from ALL words on this line for reference
                    avg_top = (n1['top'] + n2['top'] + n3['top']) / 3
                    line_words = [w for w in all_words 
                                  if abs(w['top'] - avg_top) <= y_tolerance]
                    ref_height = max(w['height'] for w in line_words) if line_words else n1['height']
                    
                    best_match = {
                        'day': n1['text'],
                        'month': n2['text'],
                        'year': n3['text'],
                        'day_box': n1,
                        'month_box': n2,
                        'year_box': n3,
                        'ref_height': ref_height,
                    }
                    break
            if best_match:
                break
        if best_match:
            break
    
    return best_match


def try_text_extraction(doc, page_num=0):
    """Try standard text extraction (for text-based PDFs)."""
    page = doc[page_num]
    text = page.get_text("text")
    normalized = unicodedata.normalize('NFC', re.sub(r'\s+', ' ', text))
    
    match = DATE_PATTERN.search(normalized)
    if match:
        return {
            'found': True,
            'day': match.group(1),
            'month': match.group(2),
            'year': match.group(3),
            'method': 'text',
        }
    
    # Try block-by-block
    blocks = page.get_text("dict")["blocks"]
    for block in blocks:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            line_text = "".join(span["text"] for span in line["spans"])
            normalized_line = unicodedata.normalize('NFC', re.sub(r'\s+', ' ', line_text))
            match = DATE_PATTERN.search(normalized_line)
            if match:
                return {
                    'found': True,
                    'day': match.group(1),
                    'month': match.group(2),
                    'year': match.group(3),
                    'method': 'text_blocks',
                }
    
    return {'found': False}


# ===== Routes =====

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/download/<token>')
def download_file(token):
    """Serve a previously generated file for download."""
    # Find the file matching this token
    for fname in os.listdir(DOWNLOAD_DIR):
        if fname.startswith(token + '__'):
            filepath = os.path.join(DOWNLOAD_DIR, fname)
            # Extract original filename from stored name
            original_name = fname[len(token) + 2:]  # skip token + '__'
            return send_file(
                filepath,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=original_name
            )
    return jsonify({"error": "File not found"}), 404


@app.route('/api/preview', methods=['POST'])
def api_preview():
    """Upload PDF, show preview, extract date."""
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "Vui lòng chọn file PDF!"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "Vui lòng chọn file!"})
    
    try:
        pdf_bytes = file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        # Generate preview
        preview_img = pdf_page_to_image(doc, page_num=0, dpi=150)
        buf = io.BytesIO()
        preview_img.save(buf, format='PNG', optimize=True)
        buf.seek(0)
        preview_b64 = base64.b64encode(buf.read()).decode('utf-8')
        
        # Try text extraction first
        text_result = try_text_extraction(doc)
        if text_result['found']:
            doc.close()
            return jsonify({
                "success": True,
                "preview": preview_b64,
                "day": text_result['day'],
                "month": text_result['month'],
                "year": text_result['year'],
                "method": text_result['method'],
            })
        
        # Fallback: OCR
        ocr_img = pdf_page_to_image(doc, page_num=0, dpi=300)
        ocr_result = ocr_find_date(ocr_img)
        doc.close()
        
        if ocr_result:
            return jsonify({
                "success": True,
                "preview": preview_b64,
                "day": ocr_result['day'],
                "month": ocr_result['month'],
                "year": ocr_result['year'],
                "method": "ocr",
            })
        else:
            return jsonify({
                "success": False,
                "preview": preview_b64,
                "error": "Không tìm thấy ngày tháng tự động. Vui lòng nhập thủ công.",
                "manual_mode": True,
            })
    except Exception as e:
        return jsonify({"success": False, "error": f"Lỗi: {str(e)}"})


@app.route('/api/edit-date', methods=['POST'])
def api_edit_date():
    """Replace date numbers in the PDF and return modified file."""
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "Vui lòng chọn file PDF!"})
    
    file = request.files['file']
    new_day = request.form.get('new_day', '').strip()
    new_month = request.form.get('new_month', '').strip()
    new_year = request.form.get('new_year', '').strip()
    
    if not new_day or not new_month or not new_year:
        return jsonify({"success": False, "error": "Vui lòng nhập đầy đủ ngày, tháng, năm!"})
    
    try:
        pdf_bytes = file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[0]
        
        # ---- Method 1: Text-based PDF ----
        text_result = try_text_extraction(doc)
        if text_result['found']:
            old_day = text_result['day']
            old_month = text_result['month']
            old_year = text_result['year']
            
            new_date_text = f"ngày {new_day.zfill(2)} tháng {new_month.zfill(2)} năm {new_year}"
            
            # Try to find and replace the full date string
            patterns = [
                f"ngày {old_day} tháng {old_month} năm {old_year}",
                f"ngày  {old_day}  tháng  {old_month}  năm  {old_year}",
            ]
            
            for pat in patterns:
                rects = page.search_for(pat)
                if rects:
                    font_size = 13
                    for block in page.get_text("dict")["blocks"]:
                        if "lines" not in block:
                            continue
                        for line in block["lines"]:
                            for span in line["spans"]:
                                if any(k in span["text"].lower() for k in ["ngày", "tháng", "năm"]):
                                    font_size = span["size"]
                                    break
                    
                    page.add_redact_annot(rects[0], new_date_text, fontname="helv",
                                         fontsize=font_size, align=fitz.TEXT_ALIGN_CENTER)
                    page.apply_redactions()
                    
                    output_name = f"{os.path.splitext(file.filename)[0]}_sua_ngay.pdf"
                    token = str(uuid.uuid4())
                    save_path = os.path.join(DOWNLOAD_DIR, f"{token}__{output_name}")
                    doc.save(save_path)
                    doc.close()
                    return jsonify({"success": True, "download_url": f"/download/{token}"})
        
        # ---- Method 2: OCR-based (scanned PDF) - Image-level editing ----
        ocr_img = pdf_page_to_image(doc, page_num=0, dpi=300)
        ocr_result = ocr_find_date(ocr_img)
        
        if not ocr_result:
            doc.close()
            return jsonify({"success": False, "error": "Không tìm thấy vị trí ngày tháng trong PDF!"})
        
        # Edit directly on the image using PIL
        from PIL import ImageDraw, ImageFont, ImageFilter
        import numpy as np
        
        # Sample text color from nearby words on the same line
        def sample_text_color(image, words_data, ocr_data):
            """Get average dark pixel color from text near the date."""
            n = len(ocr_data['text'])
            avg_top = (words_data['day_box']['top'] + words_data['year_box']['top']) / 2
            
            text_colors = []
            for i in range(n):
                t = ocr_data['text'][i].strip()
                top = ocr_data['top'][i]
                if t and abs(top - avg_top) < 40 and not t.isdigit():
                    x0 = ocr_data['left'][i]
                    y0 = ocr_data['top'][i]
                    x1 = x0 + ocr_data['width'][i]
                    y1 = y0 + ocr_data['height'][i]
                    arr = np.array(image.crop((x0, y0, x1, y1)))
                    dark = arr[arr.mean(axis=2) < 128]
                    if len(dark) > 0:
                        text_colors.extend(dark.tolist())
            
            if text_colors:
                avg = np.array(text_colors).mean(axis=0).astype(int)
                return tuple(avg)
            return (24, 24, 24)  # fallback dark grey
        
        def sample_bg_color(image, box):
            """Get background color from area just above the text."""
            x0, y0 = box['left'], box['top'] - 25
            x1 = box['left'] + box['width']
            y1 = box['top'] - 5
            if y0 < 0:
                y0 = 0
            arr = np.array(image.crop((x0, y0, x1, y1)))
            light = arr[arr.mean(axis=2) > 200]
            if len(light) > 0:
                return tuple(np.array(light).mean(axis=0).astype(int))
            return (252, 252, 252)
        
        # Re-run OCR data for color sampling
        ocr_data_full = pytesseract.image_to_data(
            ocr_img, lang=OCR_LANG, output_type=pytesseract.Output.DICT
        )
        text_color = sample_text_color(ocr_img, ocr_result, ocr_data_full)
        bg_color = sample_bg_color(ocr_img, ocr_result['day_box'])
        
        draw = ImageDraw.Draw(ocr_img)
        
        def replace_number_on_image(box, new_text, ref_height):
            """Replace number with scan-accurate rendering."""
            x0 = box['left']
            y0 = box['top']
            x1 = box['left'] + box['width']
            y1 = box['top'] + box['height']
            
            # Generous padding to fully cover old number
            pad_x = 6
            pad_y = 4
            fill_rect = [x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y]
            draw.rectangle(fill_rect, fill=bg_color)
            
            # Use Times New Roman (serif) to match original document
            # PIL font size ≠ pixel height. Calibrated: size=139 → rendered~96px
            font_size = int(ref_height * 1.50)
            if font_size < 28:
                font_size = 28
            
            # Try multiple font paths (Windows + Linux)
            font = None
            font_candidates = [
                'times.ttf',  # Windows system
                'C:\\Windows\\Fonts\\times.ttf',
                '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf',
                '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf',
                '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf',
                '/usr/share/fonts/truetype/freefont/FreeSerif.ttf',
            ]
            for fpath in font_candidates:
                try:
                    font = ImageFont.truetype(fpath, font_size)
                    break
                except:
                    continue
            if font is None:
                font = ImageFont.load_default()
            
            # Render text on a separate image, then blur to match scan quality
            bbox = draw.textbbox((0, 0), new_text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            # Create text layer with padding for blur
            blur_pad = 8
            layer_w = text_w + blur_pad * 2
            layer_h = text_h + blur_pad * 2
            text_layer = Image.new('RGBA', (layer_w, layer_h), (*bg_color, 255))
            text_draw = ImageDraw.Draw(text_layer)
            text_draw.text((blur_pad - bbox[0], blur_pad - bbox[1]), new_text, 
                          fill=(*text_color, 255), font=font)
            
            # Apply Gaussian blur to match the fuzzy scan quality
            text_layer = text_layer.filter(ImageFilter.GaussianBlur(radius=1.2))
            
            # Position: center horizontally in original box, align vertically
            paste_x = int(x0 + (x1 - x0 - text_w) / 2 - blur_pad)
            paste_y = int(y0 + (y1 - y0 - text_h) / 2 - blur_pad - 1)
            
            # Paste onto the main image
            ocr_img.paste(text_layer, (paste_x, paste_y), text_layer)
        
        # Use year box height as the most reliable reference (4-digit number)
        ref_h = ocr_result['year_box']['height']
        replace_number_on_image(ocr_result['day_box'], new_day.zfill(2), ref_h)
        replace_number_on_image(ocr_result['month_box'], new_month.zfill(2), ref_h)
        replace_number_on_image(ocr_result['year_box'], new_year, ref_h)
        
        # Save modified image as PDF
        # Get original page dimensions
        orig_page = doc[0]
        page_width = orig_page.rect.width
        page_height = orig_page.rect.height
        
        # Save image to bytes
        img_bytes = io.BytesIO()
        ocr_img.save(img_bytes, format='JPEG', quality=95)
        img_bytes.seek(0)
        
        # Create new PDF with the modified image
        new_doc = fitz.open()
        new_page = new_doc.new_page(width=page_width, height=page_height)
        new_page.insert_image(new_page.rect, stream=img_bytes.read())
        
        # Copy remaining pages from original
        for p in range(1, len(doc)):
            new_doc.insert_pdf(doc, from_page=p, to_page=p)
        
        output_name = f"{os.path.splitext(file.filename)[0]}_sua_ngay.pdf"
        token = str(uuid.uuid4())
        save_path = os.path.join(DOWNLOAD_DIR, f"{token}__{output_name}")
        new_doc.save(save_path)
        new_doc.close()
        doc.close()
        
        return jsonify({"success": True, "download_url": f"/download/{token}"})
    except Exception as e:
        return jsonify({"success": False, "error": f"Lỗi xử lý PDF: {str(e)}"})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5007))
    app.run(debug=False, host='0.0.0.0', port=port)
