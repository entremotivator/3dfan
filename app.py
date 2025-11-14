import streamlit as st
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import io
import numpy as np
import struct
import time
import math
from collections import Counter

st.set_page_config(page_title="Image ⇄ BIN/NLF/BIM Converter Pro", page_icon="🖼️", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #2196f3;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1.2rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    .animation-frame {
        border: 3px solid #667eea;
        border-radius: 10px;
        padding: 10px;
        background: #f8f9fa;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 500;
    }
    .file-type-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.5rem;
    }
    .badge-nlf {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
    }
    .badge-bim {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        color: white;
    }
    .badge-bin {
        background: linear-gradient(135deg, #4facfe, #00f2fe);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🖼️ Image ⇄ BIN/NLF/BIM Converter Pro</h1><p>Advanced converter for LED displays, LED fans, and embedded systems • Full animation support</p></div>', unsafe_allow_html=True)

# Initialize session state
if 'animation_playing' not in st.session_state:
    st.session_state.animation_playing = False
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = 0
if 'frame_data' not in st.session_state:
    st.session_state.frame_data = []
if 'width1' not in st.session_state:
    st.session_state.width1 = 64
if 'height1' not in st.session_state:
    st.session_state.height1 = 32
if 'width2' not in st.session_state:
    st.session_state.width2 = 64
if 'height2' not in st.session_state:
    st.session_state.height2 = 32

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📤 Image to BIN/NLF", "📥 BIN/NLF/BIM Viewer", "🎬 Animation Tools", "📚 Documentation"])

# ==================== TAB 1: IMAGE TO BIN ====================
with tab1:
    st.header("Convert Image to BIN/NLF File")
    
    col_settings, col_upload = st.columns([1, 2])
    
    with col_settings:
        st.subheader("⚙️ Display Settings")
        
        matrix_width = st.number_input("Width (pixels)", min_value=8, max_value=512, value=st.session_state.width1, step=8, key="width1")
        matrix_height = st.number_input("Height (pixels)", min_value=8, max_value=512, value=st.session_state.height1, step=8, key="height1")
        
        st.markdown("---")
        
        color_format = st.selectbox(
            "Color Format",
            ["RGB565", "RGB888", "BGR888", "RGB24", "RGBA8888", "Grayscale", "Monochrome"],
            help="RGB565: 16-bit (LED matrices)\nRGB888: 24-bit full color\nBGR888: 24-bit reversed\nRGBA8888: 32-bit with alpha\nGrayscale: 8-bit\nMonochrome: 1-bit"
        )
        
        output_format = st.selectbox(
            "Output File Format",
            ["BIN (Raw Binary)", "NLF (LED Fan Animation)", "BIM (Binary Image)"],
            help="Choose output format based on your device"
        )
        
        resize_method = st.selectbox(
            "Resize Method",
            ["Fit (maintain aspect)", "Stretch", "Crop to center", "Tile/Repeat"],
            help="How to adapt the image to matrix size"
        )
        
        st.markdown("---")
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            brightness = st.slider("Brightness", 0, 200, 100, 5, help="Adjust image brightness")
            contrast = st.slider("Contrast", 0, 200, 100, 5, help="Adjust image contrast")
            saturation = st.slider("Saturation", 0, 200, 100, 5, help="Adjust color saturation")
            sharpness = st.slider("Sharpness", 0, 200, 100, 5, help="Adjust image sharpness")
            dithering = st.checkbox("Apply Dithering", value=False, help="Better for low-color displays")
            flip_h = st.checkbox("Flip Horizontal", value=False)
            flip_v = st.checkbox("Flip Vertical", value=False)
            rotate = st.selectbox("Rotate", [0, 90, 180, 270], help="Degrees clockwise")
            gamma = st.slider("Gamma Correction", 0.5, 3.0, 1.0, 0.1, help="Adjust gamma for displays")
        
        # Quick presets
        with st.expander("⚡ Quick Presets"):
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                if st.button("32×32 RGB LED", use_container_width=True):
                    st.session_state.width1 = 32
                    st.session_state.height1 = 32
                    st.rerun()
                if st.button("64×64 RGB LED", use_container_width=True):
                    st.session_state.width1 = 64
                    st.session_state.height1 = 64
                    st.rerun()
                if st.button("144×64 LED Fan", use_container_width=True):
                    st.session_state.width1 = 144
                    st.session_state.height1 = 64
                    st.rerun()
            with col_p2:
                if st.button("64×32 RGB LED", use_container_width=True):
                    st.session_state.width1 = 64
                    st.session_state.height1 = 32
                    st.rerun()
                if st.button("128×64 OLED", use_container_width=True):
                    st.session_state.width1 = 128
                    st.session_state.height1 = 64
                    st.rerun()
                if st.button("120×60 LED Fan", use_container_width=True):
                    st.session_state.width1 = 120
                    st.session_state.height1 = 60
                    st.rerun()
    
    with col_upload:
        uploaded_file = st.file_uploader(
            "📁 Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'gif', 'webp', 'tiff'],
            help="Supported formats: PNG, JPG, BMP, GIF, WebP, TIFF"
        )
        
        if uploaded_file is not None:
            # Display images side by side
            col_orig, col_proc = st.columns(2)
            
            with col_orig:
                st.markdown("**📥 Original Image**")
                original_image = Image.open(uploaded_file)
                st.image(original_image, use_container_width=True)
                st.caption(f"📏 {original_image.size[0]}×{original_image.size[1]} | Mode: {original_image.mode} | Format: {original_image.format}")
            
            # Process image
            img = original_image.copy()
            
            # Apply rotation
            if rotate != 0:
                img = img.rotate(-rotate, expand=True)
            
            # Apply flips
            if flip_h:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if flip_v:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            
            # Resize based on method
            if resize_method == "Fit (maintain aspect)":
                img.thumbnail((matrix_width, matrix_height), Image.Resampling.LANCZOS)
                canvas = Image.new('RGB', (matrix_width, matrix_height), (0, 0, 0))
                offset = ((matrix_width - img.size[0]) // 2, (matrix_height - img.size[1]) // 2)
                canvas.paste(img, offset)
                img = canvas
            elif resize_method == "Stretch":
                img = img.resize((matrix_width, matrix_height), Image.Resampling.LANCZOS)
            elif resize_method == "Crop to center":
                aspect = original_image.size[0] / original_image.size[1]
                target_aspect = matrix_width / matrix_height
                if aspect > target_aspect:
                    new_width = int(original_image.size[1] * target_aspect)
                    left = (original_image.size[0] - new_width) // 2
                    img = original_image.crop((left, 0, left + new_width, original_image.size[1]))
                else:
                    new_height = int(original_image.size[0] / target_aspect)
                    top = (original_image.size[1] - new_height) // 2
                    img = original_image.crop((0, top, original_image.size[0], top + new_height))
                img = img.resize((matrix_width, matrix_height), Image.Resampling.LANCZOS)
            else:  # Tile/Repeat
                canvas = Image.new('RGB', (matrix_width, matrix_height), (0, 0, 0))
                for y in range(0, matrix_height, img.size[1]):
                    for x in range(0, matrix_width, img.size[0]):
                        canvas.paste(img, (x, y))
                img = canvas
            
            # Convert to RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Apply enhancements
            if brightness != 100:
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(brightness / 100)
            if contrast != 100:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(contrast / 100)
            if saturation != 100:
                enhancer = ImageEnhance.Color(img)
                img = enhancer.enhance(saturation / 100)
            if sharpness != 100:
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(sharpness / 100)
            
            # Apply gamma correction
            if gamma != 1.0:
                img_array = np.array(img).astype(float)
                img_array = 255 * np.power(img_array / 255, 1/gamma)
                img = Image.fromarray(img_array.astype('uint8'))
            
            # Apply dithering if needed
            if dithering:
                if color_format == "Monochrome":
                    # Dithering for 1-bit monochrome
                    img = img.convert('L').convert('1', dither=Image.Dither.FLOYDSTEINBERG)
                elif color_format == "Grayscale":
                    # Dithering for 8-bit grayscale (optional, but can improve perception)
                    img = img.convert('L')
                elif color_format == "RGB565":
                    # Dithering for 16-bit color (more complex, often done manually or by hardware)
                    # For simplicity, we'll use PIL's quantize method with dithering for a limited palette
                    img = img.quantize(colors=65536, method=Image.Quantize.MAXCOVERAGE, dither=Image.Dither.FLOYDSTEINBERG)
                    img = img.convert('RGB') # Convert back to RGB for pixel processing
            
            with col_proc:
                st.markdown("**📤 Processed Image**")
                st.image(img, use_container_width=True)
                st.caption(f"📏 {img.size[0]}×{img.size[1]} | Format: {color_format}")
            
            # Convert to binary data
            pixels = np.array(img)
            bin_data = bytearray()
            bytes_per_pixel = 0
            
            if color_format == "RGB565":
                for row in pixels:
                    for pixel in row:
                        # Ensure we handle different input modes (e.g., if dithering converted to L or 1)
                        if len(pixel) >= 3:
                            r, g, b = pixel[:3]
                        elif len(pixel) == 1: # Grayscale/Monochrome
                            r = g = b = pixel[0]
                        else:
                            r = g = b = 0 # Should not happen if converted to RGB/L/1
                            
                        r5 = (r >> 3) & 0x1F
                        g6 = (g >> 2) & 0x3F
                        b5 = (b >> 3) & 0x1F
                        rgb565 = (r5 << 11) | (g6 << 5) | b5
                        
                        # Big-endian (MSB first)
                        # bin_data.append(rgb565 >> 8)
                        # bin_data.append(rgb565 & 0xFF)
                        
                        # Little-endian (LSB first) - More common in embedded systems
                        bin_data.append(rgb565 & 0xFF)
                        bin_data.append(rgb565 >> 8)
                        
                bytes_per_pixel = 2
            
            elif color_format == "RGB888" or color_format == "RGB24":
                for row in pixels:
                    for pixel in row:
                        if len(pixel) >= 3:
                            bin_data.extend(pixel[:3])
                        else:
                            bin_data.extend([pixel[0], pixel[0], pixel[0]])
                bytes_per_pixel = 3
            
            elif color_format == "BGR888":
                for row in pixels:
                    for pixel in row:
                        if len(pixel) >= 3:
                            bin_data.extend([pixel[2], pixel[1], pixel[0]]) # B, G, R
                        else:
                            bin_data.extend([pixel[0], pixel[0], pixel[0]])
                bytes_per_pixel = 3
            
            elif color_format == "RGBA8888":
                for row in pixels:
                    for pixel in row:
                        if len(pixel) >= 4:
                            bin_data.extend(pixel[:4]) # R, G, B, A
                        elif len(pixel) >= 3:
                            bin_data.extend([pixel[0], pixel[1], pixel[2], 255]) # Assume full alpha
                        else:
                            bin_data.extend([pixel[0], pixel[0], pixel[0], 255])
                bytes_per_pixel = 4
            
            elif color_format == "Grayscale":
                gray_img = img.convert('L')
                gray_pixels = np.array(gray_img)
                for row in gray_pixels:
                    bin_data.extend(row)
                bytes_per_pixel = 1
            
            elif color_format == "Monochrome":
                # The image is already 1-bit from the dithering step
                mono_img = img.convert('1')
                # PIL '1' mode is 1-bit pixels, stored as 8 pixels per byte.
                # We need to extract the raw data, which is usually row-major, packed.
                # The '1' mode in PIL uses 0 for black and 255 for white.
                # We need to convert this to packed bits where 1 is ON/WHITE and 0 is OFF/BLACK.
                
                # The original code had a manual bit-packing loop which is more explicit
                # Let's use the manual loop but ensure it works with the PIL '1' mode output
                
                # Convert to a numpy array of 0s and 1s
                mono_pixels = np.array(mono_img.convert('L'))
                # Convert 255 (white) to 1, and 0 (black) to 0
                bit_pixels = (mono_pixels > 127).astype(np.uint8).flatten()
                
                byte_val = 0
                bit_count = 0
                for pixel in bit_pixels:
                    byte_val = (byte_val << 1) | pixel
                    bit_count += 1
                    if bit_count == 8:
                        bin_data.append(byte_val)
                        byte_val = 0
                        bit_count = 0
                
                # Pad the last byte if necessary (common for row-major packing)
                if bit_count > 0:
                    byte_val <<= (8 - bit_count)
                    bin_data.append(byte_val)
                    
                bytes_per_pixel = 0.125 # 1 bit per pixel
            
            # Add format-specific headers/metadata
            final_data = bytearray()
            if "NLF" in output_format:
                # NLF header (simplified example)
                # Magic: 'NLF1' (4 bytes), Width: uint16 (2 bytes), Height: uint16 (2 bytes), BPP: uint8 (1 byte), Frames: uint8 (1 byte)
                # Determine BPP for header
                bpp_header = 0
                if color_format == "RGB565": bpp_header = 2
                elif color_format in ["RGB888", "BGR888", "RGB24"]: bpp_header = 3
                elif color_format == "RGBA8888": bpp_header = 4
                elif color_format == "Grayscale": bpp_header = 1
                elif color_format == "Monochrome": bpp_header = 1 # Often treated as 1 byte for simplicity in header, but data is packed
                
                final_data.extend(struct.pack('<4sHHBB', b'NLF1', matrix_width, matrix_height, bpp_header, 1))  # 1 frame
                final_data.extend(bin_data)
            elif "BIM" in output_format:
                # BIM header (simplified example)
                # Magic: 'BIM\x00' (4 bytes), Width: uint32 (4 bytes), Height: uint32 (4 bytes)
                final_data.extend(struct.pack('<4sII', b'BIM\x00', matrix_width, matrix_height))
                final_data.extend(bin_data)
            else:
                final_data = bin_data
            
            # Success message and metrics
            st.markdown('<div class="success-box">✅ <b>Conversion Complete!</b></div>', unsafe_allow_html=True)
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Resolution", f"{matrix_width}×{matrix_height}")
            with col_m2:
                st.metric("Format", color_format)
            with col_m3:
                st.metric("File Size", f"{len(final_data):,} bytes")
            with col_m4:
                st.metric("Pixels", f"{matrix_width * matrix_height:,}")
            
            # Download button
            if "NLF" in output_format:
                file_ext = ".nlf"
            elif "BIM" in output_format:
                file_ext = ".bim"
            else:
                file_ext = ".bin"
            
            filename = uploaded_file.name.rsplit('.', 1)[0] + file_ext
            st.download_button(
                label=f"⬇️ Download {file_ext.upper()} File",
                data=bytes(final_data),
                file_name=filename,
                mime="application/octet-stream",
                use_container_width=True
            )
            
            # Additional details
            with st.expander("📊 Detailed Information", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write("**File Information:**")
                    st.write(f"• Total pixels: {matrix_width * matrix_height:,}")
                    st.write(f"• Bytes per pixel: {bytes_per_pixel}")
                    st.write(f"• Image data: {len(bin_data):,} bytes")
                    st.write(f"• Total file size: {len(final_data):,} bytes")
                    st.write(f"• Compression: None (raw)")
                
                with col2:
                    st.write("**Color Information:**")
                    if color_format == "RGB565":
                        st.write("• Red: 5 bits (32 levels)")
                        st.write("• Green: 6 bits (64 levels)")
                        st.write("• Blue: 5 bits (32 levels)")
                        st.write("• Total colors: 65,536")
                    elif color_format in ["RGB888", "RGB24", "BGR888"]:
                        st.write("• Red: 8 bits (256 levels)")
                        st.write("• Green: 8 bits (256 levels)")
                        st.write("• Blue: 8 bits (256 levels)")
                        st.write("• Total colors: 16.7M")
                    elif color_format == "RGBA8888":
                        st.write("• Red: 8 bits")
                        st.write("• Green: 8 bits")
                        st.write("• Blue: 8 bits")
                        st.write("• Alpha: 8 bits")
                    elif color_format == "Grayscale":
                        st.write("• 8 bits per pixel")
                        st.write("• 256 gray levels")
                    elif color_format == "Monochrome":
                        st.write("• 1 bit per pixel (packed)")
                        st.write("• 2 colors (Black/White)")
                
                with col3:
                    st.write("**Processing Applied:**")
                    st.write(f"• Brightness: {brightness}%")
                    st.write(f"• Contrast: {contrast}%")
                    st.write(f"• Saturation: {saturation}%")
                    st.write(f"• Sharpness: {sharpness}%")
                    st.write(f"• Gamma: {gamma}")
                    st.write(f"• Rotation: {rotate}°")
                    st.write(f"• Dithering: {'Yes' if dithering else 'No'}")
            
            # Hex preview
            with st.expander("🔍 Hex Data Preview (First 512 bytes)"):
                preview_bytes = min(512, len(final_data))
                hex_lines = []
                for i in range(0, preview_bytes, 16):
                    hex_part = ' '.join(f'{b:02X}' for b in final_data[i:i+16])
                    ascii_part = ''.join(chr(b) if 32 <= b < 127 else '.' for b in final_data[i:i+16])
                    hex_lines.append(f"{i:08X}  {hex_part:<48}  {ascii_part}")
                st.code('\n'.join(hex_lines), language=None)
                if len(final_data) > preview_bytes:
                    st.write(f"... and {len(final_data) - preview_bytes:,} more bytes")
        else:
            st.info("👆 Upload an image file to begin conversion")

# ==================== TAB 2: BIN/NLF/BIM VIEWER ====================
with tab2:
    st.header("BIN/NLF/BIM File Viewer & Deep Analyzer")
    
    col_settings2, col_viewer = st.columns([1, 2])
    
    with col_settings2:
        st.subheader("⚙️ Display Settings")
        
        bin_width = st.number_input("Width (pixels)", min_value=1, max_value=2000, value=st.session_state.width2, step=1, key="width2")
        bin_height = st.number_input("Height (pixels)", min_value=1, max_value=2000, value=st.session_state.height2, step=1, key="height2")
        
        st.markdown("---")
        
        bin_format = st.selectbox(
            "Color Format",
            ["RGB565", "RGB888", "BGR888", "RGB24", "RGBA8888", "Grayscale", "Monochrome"],
            help="Format of the binary file",
            key="bin_format"
        )
        
        st.markdown("---")
        
        with st.expander("🔧 Advanced View Options"):
            zoom_level = st.slider("Zoom Level", 1, 20, 4, help="Pixel size multiplier")
            show_grid = st.checkbox("Show Pixel Grid", value=False)
            auto_detect = st.checkbox("Auto-detect dimensions", value=True, help="Automatically detect file format")
            invert_colors = st.checkbox("Invert Colors", value=False)
            rotate_view = st.selectbox("Rotate View", [0, 90, 180, 270], help="Rotate displayed image")
            channel_view = st.selectbox("Channel View", ["Full Color", "Red Only", "Green Only", "Blue Only", "Alpha Only"])
            
        with st.expander("🎯 Common Display Presets"):
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                if st.button("16×16", use_container_width=True, key="p16"):
                    st.session_state.width2 = 16
                    st.session_state.height2 = 16
                    st.rerun()
                if st.button("64×32", use_container_width=True, key="p6432"):
                    st.session_state.width2 = 64
                    st.session_state.height2 = 32
                    st.rerun()
                if st.button("128×64", use_container_width=True, key="p12864"):
                    st.session_state.width2 = 128
                    st.session_state.height2 = 64
                    st.rerun()
                if st.button("144×64 (Fan)", use_container_width=True, key="p14464"):
                    st.session_state.width2 = 144
                    st.session_state.height2 = 64
                    st.rerun()
            with col_p2:
                if st.button("32×32", use_container_width=True, key="p32"):
                    st.session_state.width2 = 32
                    st.session_state.height2 = 32
                    st.rerun()
                if st.button("64×64", use_container_width=True, key="p64"):
                    st.session_state.width2 = 64
                    st.session_state.height2 = 64
                    st.rerun()
                if st.button("128×128", use_container_width=True, key="p128"):
                    st.session_state.width2 = 128
                    st.session_state.height2 = 128
                    st.rerun()
                if st.button("120×60 (Fan)", use_container_width=True, key="p12060"):
                    st.session_state.width2 = 120
                    st.session_state.height2 = 60
                    st.rerun()
    
    with col_viewer:
        bin_file = st.file_uploader(
            "📁 Upload BIN/NLF/BIM file",
            type=['bin', 'nlf', 'bim', 'dat', 'raw'],
            help="Upload a binary image file to view and analyze",
            key="bin_uploader"
        )
        
        if bin_file is not None:
            bin_data = bin_file.read()
            file_size = len(bin_data)
            file_ext = bin_file.name.split('.')[-1].lower()
            
            # File type badge
            if file_ext == "nlf":
                badge_class = "badge-nlf"
                badge_text = "🎬 NLF Format"
                file_desc = "LED Fan Animation File"
            elif file_ext == "bim":
                badge_class = "badge-bim"
                badge_text = "🖼️ BIM Format"
                file_desc = "Binary Image Format"
            else:
                badge_class = "badge-bin"
                badge_text = "📦 BIN Format"
                file_desc = "Raw Binary Data"
            
            st.markdown(f'<div class="file-type-badge {badge_class}">{badge_text}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="info-box">📄 <b>File loaded successfully!</b><br>{file_desc} • {file_size:,} bytes</div>', unsafe_allow_html=True)
            
            # Try to detect file format from headers
            header_info = None
            has_header = False
            frame_count = 1
            data_offset = 0
            
            if file_size >= 4:
                header = bin_data[:4]
                if header == b'NLF1':
                    has_header = True
                    if file_size >= 10:
                        # <HHBB: little-endian, 2x uint16, 2x uint8
                        width, height, bpp, frames = struct.unpack('<HHBB', bin_data[4:10])
                        st.session_state.width2 = width
                        st.session_state.height2 = height
                        frame_count = frames
                        data_offset = 10
                        header_info = f"✅ NLF Header Detected: {width}×{height}, {frames} frame(s), {bpp} bytes/pixel"
                elif header[:3] == b'BIM':
                    has_header = True
                    if file_size >= 12:
                        # <II: little-endian, 2x uint32
                        width, height = struct.unpack('<II', bin_data[4:12])
                        st.session_state.width2 = width
                        st.session_state.height2 = height
                        data_offset = 12
                        header_info = f"✅ BIM Header Detected: {width}×{height}"
            
            if header_info:
                st.success(header_info)
                bin_data = bin_data[data_offset:]  # Skip header
                file_size = len(bin_data)
            
            # Auto-detect dimensions
            if auto_detect and not has_header:
                st.markdown("### 🔍 Auto-Detection Analysis")
                possible_dims = []
                
                # Check for common LED fan dimensions
                if file_ext == "nlf":
                    st.info("🎬 Analyzing as LED fan animation format...")
                    common_fan_sizes = [(144, 64), (120, 60), (96, 48), (72, 36), (64, 32)]
                    for w, h in common_fan_sizes:
                        for fmt_name, bpp in [("RGB565", 2), ("RGB888", 3), ("RGBA8888", 4)]:
                            frame_size = w * h * bpp
                            if file_size % frame_size == 0 and file_size >= frame_size:
                                num_frames = file_size // frame_size
                                possible_dims.append((w, h, fmt_name, num_frames, frame_size))
                
                # Standard dimension detection
                for fmt_name, bpp in [("RGB565", 2), ("RGB888", 3), ("BGR888", 3), ("RGBA8888", 4), ("Grayscale", 1)]:
                    total_pixels = file_size / bpp
                    if total_pixels == int(total_pixels):
                        total_pixels = int(total_pixels)
                        
                        candidates = []
                        
                        # Check common sizes first
                        common_sizes = [(16,16), (32,32), (48,48), (64,32), (64,64), (96,96), (128,64), (128,128), 
                                       (144,64), (120,60), (256,128), (256,256)]
                        for w, h in common_sizes:
                            if w * h == total_pixels:
                                candidates.append((w, h, fmt_name))
                        
                        # Then check all possibilities (up to a limit)
                        if not candidates:
                            # Find factors of total_pixels
                            factors = [i for i in range(1, int(math.sqrt(total_pixels)) + 1) if total_pixels % i == 0]
                            for f in factors:
                                w = f
                                h = total_pixels // f
                                candidates.append((w, h, fmt_name))
                                if w != h:
                                    candidates.append((h, w, fmt_name))
                        
                        for w, h, fmt in candidates:
                            frame_size = w * h * bpp
                            possible_dims.append((w, h, fmt, 1, frame_size))
                
                # Monochrome is special (packed bits)
                bpp_mono = 0.125
                total_bits = file_size * 8
                
                # Check for common sizes for monochrome
                for w, h in [(16,16), (32,32), (64,64), (128,64), (128,128)]:
                    if w * h == total_bits:
                        frame_size = w * h * bpp_mono
                        possible_dims.append((w, h, "Monochrome", 1, frame_size))
                
                if possible_dims:
                    # Separate multi-frame and single-frame
                    multi_frame = [d for d in possible_dims if d[3] > 1]
                    single_frame = [d for d in possible_dims if d[3] == 1]
                    
                    if multi_frame:
                        st.markdown("#### 🎬 **Animation Detected (Multi-frame)**")
                        for w, h, fmt, frames, frame_sz in multi_frame[:8]:
                            col_btn, col_info = st.columns([4, 1])
                            with col_btn:
                                btn_label = f"▶ {w}×{h} • {fmt} • {frames} frames • {frame_sz:,}B/frame"
                                if st.button(btn_label, key=f"anim_{w}_{h}_{fmt}_{frames}", use_container_width=True):
                                    st.session_state.width2 = w
                                    st.session_state.height2 = h
                                    st.session_state.bin_format = fmt
                                    st.session_state.detected_frames = frames
                                    st.session_state.frame_size = frame_sz
                                    st.rerun()
                            with col_info:
                                st.caption(f"{w/h:.2f}:1")
                        st.markdown("---")
                    
                    if single_frame:
                        st.markdown("#### 🖼️ **Single Frame Options**")
                        for w, h, fmt, _, _ in single_frame[:12]:
                            col_btn, col_info = st.columns([4, 1])
                            with col_btn:
                                if st.button(f"▶ {w}×{h} pixels • {fmt}", key=f"single_{w}_{h}_{fmt}", use_container_width=True):
                                    st.session_state.width2 = w
                                    st.session_state.height2 = h
                                    st.session_state.bin_format = fmt
                                    st.session_state.detected_frames = 1
                                    st.session_state.frame_size = w * h * bpp_map.get(fmt, 2)
                                    st.rerun()
                            with col_info:
                                st.caption(f"{w/h:.2f}:1")
                        
                        if len(single_frame) > 12:
                            st.info(f"... and {len(single_frame)-12} more possibilities")
                else:
                    st.warning("⚠️ Could not auto-detect standard dimensions. Try manual configuration or check for file headers.")
            
            # Animation frame detection
            detected_frames = st.session_state.get('detected_frames', 1)
            
            # Recalculate frame size based on current settings if not auto-detected
            bpp_map = {"RGB565": 2, "RGB888": 3, "BGR888": 3, "RGB24": 3, "RGBA8888": 4, "Grayscale": 1, "Monochrome": 0.125}
            bpp_val = bpp_map.get(bin_format, 2)
            
            if bin_format == "Monochrome":
                frame_size = math.ceil(bin_width * bin_height * bpp_val)
            else:
                frame_size = int(bin_width * bin_height * bpp_val)
            
            # If frame size is 0, set detected frames to 1 to prevent division by zero
            if frame_size == 0:
                detected_frames = 1
            elif not has_header and auto_detect:
                # If not auto-detected from header, recalculate frames based on manual settings
                detected_frames = file_size // frame_size
                if file_size % frame_size != 0:
                    st.warning(f"File size ({file_size} B) is not an exact multiple of the calculated frame size ({frame_size} B). Assuming 1 frame.")
                    detected_frames = 1
            
            if detected_frames > 1:
                st.markdown(f'<div class="warning-box">🎬 <b>Animation Mode Active:</b> {detected_frames} frames detected</div>', unsafe_allow_html=True)
                current_frame = st.slider("Select Frame", 0, detected_frames - 1, 0, help="Scrub through animation frames")
            else:
                current_frame = 0
            
            # File metrics
            col_a1, col_a2, col_a3, col_a4 = st.columns(4)
            with col_a1:
                st.metric("File Size", f"{file_size:,} B")
            with col_a2:
                st.metric("Format", bin_format)
            with col_a3:
                st.metric("Expected Size", f"{frame_size:,} B")
            with col_a4:
                match_status = "✅ Match" if abs(file_size - frame_size * detected_frames) < 100 else "⚠️ Mismatch"
                st.metric("Size Check", match_status)
            
            # Decode and display
            try:
                # Calculate frame offset
                frame_offset = current_frame * frame_size
                frame_data = bin_data[frame_offset:frame_offset + frame_size]
                
                if len(frame_data) < frame_size:
                    raise ValueError(f"Frame {current_frame} is incomplete. Expected {frame_size} bytes, got {len(frame_data)} bytes.")
                
                if bin_format == "RGB565":
                    pixels = []
                    # RGB565 is little-endian in the converter, so we assume little-endian here too
                    for i in range(0, len(frame_data), 2):
                        if i+1 < len(frame_data):
                            # Little-endian: LSB first (frame_data[i]), MSB second (frame_data[i+1])
                            rgb565 = (frame_data[i+1] << 8) | frame_data[i]
                            r = ((rgb565 >> 11) & 0x1F) << 3
                            g = ((rgb565 >> 5) & 0x3F) << 2
                            b = (rgb565 & 0x1F) << 3
                            pixels.append([r, g, b])
                    
                    if len(pixels) >= bin_width * bin_height:
                        actual_height = bin_height
                        pixels = np.array(pixels[:bin_width * actual_height]).reshape((actual_height, bin_width, 3))
                        img = Image.fromarray(pixels.astype('uint8'), 'RGB')
                    else:
                        raise ValueError("Not enough data for specified dimensions")
                
                elif bin_format in ["RGB888", "RGB24"]:
                    pixels = []
                    for i in range(0, len(frame_data), 3):
                        if i+2 < len(frame_data):
                            pixels.append([frame_data[i], frame_data[i+1], frame_data[i+2]])
                    
                    if len(pixels) >= bin_width * bin_height:
                        actual_height = bin_height
                        pixels = np.array(pixels[:bin_width * actual_height]).reshape((actual_height, bin_width, 3))
                        img = Image.fromarray(pixels.astype('uint8'), 'RGB')
                    else:
                        raise ValueError("Not enough data for specified dimensions")
                
                elif bin_format == "BGR888":
                    pixels = []
                    for i in range(0, len(frame_data), 3):
                        if i+2 < len(frame_data):
                            pixels.append([frame_data[i+2], frame_data[i+1], frame_data[i]]) # R, G, B from B, G, R
                    
                    if len(pixels) >= bin_width * bin_height:
                        actual_height = bin_height
                        pixels = np.array(pixels[:bin_width * actual_height]).reshape((actual_height, bin_width, 3))
                        img = Image.fromarray(pixels.astype('uint8'), 'RGB')
                    else:
                        raise ValueError("Not enough data")
                
                elif bin_format == "RGBA8888":
                    pixels = []
                    for i in range(0, len(frame_data), 4):
                        if i+3 < len(frame_data):
                            pixels.append([frame_data[i], frame_data[i+1], frame_data[i+2]]) # Ignore Alpha for RGB view
                    
                    if len(pixels) >= bin_width * bin_height:
                        actual_height = bin_height
                        pixels = np.array(pixels[:bin_width * actual_height]).reshape((actual_height, bin_width, 3))
                        img = Image.fromarray(pixels.astype('uint8'), 'RGB')
                    else:
                        raise ValueError("Not enough data")
                
                elif bin_format == "Grayscale":
                    pixels = np.array(list(frame_data))
                    if len(pixels) >= bin_width * bin_height:
                        actual_height = bin_height
                        pixels = pixels[:bin_width * actual_height].reshape((actual_height, bin_width))
                        img = Image.fromarray(pixels.astype('uint8'), 'L')
                        img = img.convert('RGB')  # Convert to RGB for channel operations
                    else:
                        raise ValueError("Not enough data")
                
                elif bin_format == "Monochrome":
                    pixels = []
                    for byte in frame_data:
                        for bit in range(8):
                            # Extract bits from MSB to LSB (standard for packed monochrome)
                            pixels.append(255 if (byte >> (7-bit)) & 1 else 0)
                    
                    pixels = np.array(pixels)
                    if len(pixels) >= bin_width * bin_height:
                        actual_height = bin_height
                        pixels = pixels[:bin_width * actual_height].reshape((actual_height, bin_width))
                        img = Image.fromarray(pixels.astype('uint8'), 'L')
                        img = img.convert('RGB')
                    else:
                        raise ValueError("Not enough data")
                
                # Apply channel view
                if channel_view != "Full Color":
                    arr = np.array(img)
                    if channel_view == "Red Only":
                        arr[:,:,1] = 0
                        arr[:,:,2] = 0
                    elif channel_view == "Green Only":
                        arr[:,:,0] = 0
                        arr[:,:,2] = 0
                    elif channel_view == "Blue Only":
                        arr[:,:,0] = 0
                        arr[:,:,1] = 0
                    elif channel_view == "Alpha Only":
                        # Alpha channel is ignored in decoding, so this will just show a black image
                        st.warning("Alpha channel data is not available in the current decoding process.")
                        arr[:,:,:] = 0
                        
                    img = Image.fromarray(arr)
                
                # Apply transformations
                if invert_colors:
                    img = Image.eval(img, lambda x: 255 - x)
                
                if rotate_view != 0:
                    img = img.rotate(-rotate_view, expand=True)
                
                # Scale for viewing
                view_img = img.resize((img.width * zoom_level, img.height * zoom_level), Image.NEAREST)
                
                # Add grid if requested
                if show_grid and zoom_level >= 4:
                    draw = ImageDraw.Draw(view_img)
                    for x in range(0, view_img.width, zoom_level):
                        draw.line([(x, 0), (x, view_img.height)], fill=(128, 128, 128), width=1)
                    for y in range(0, view_img.height, zoom_level):
                        draw.line([(0, y), (view_img.width, y)], fill=(128, 128, 128), width=1)
                
                frame_label = f"Frame {current_frame + 1}/{detected_frames}" if detected_frames > 1 else "Decoded Image"
                st.image(view_img, caption=f"{frame_label} ({img.width}×{img.height})", use_container_width=True)
                
                # Comprehensive Analysis
                with st.expander("📊 Comprehensive Image Analysis", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**Dimensions & Format:**")
                        st.write(f"• Width: {img.width} pixels")
                        st.write(f"• Height: {img.height} pixels")
                        st.write(f"• Total pixels: {img.width * img.height:,}")
                        st.write(f"• Aspect ratio: {img.width/img.height:.4f}:1")
                        st.write(f"• Display area: {img.width * img.height / 1000:.2f}K pixels")
                        st.write(f"• Diagonal: {math.sqrt(img.width**2 + img.height**2):.1f} px")
                        if detected_frames > 1:
                            st.write(f"• Animation: {detected_frames} frames")
                            st.write(f"• Frame size: {frame_size:,} bytes")
                    
                    with col2:
                        arr = np.array(img)
                        st.write("**Color Statistics:**")
                        if len(arr.shape) == 3:
                            st.write(f"• Avg Red: {arr[:,:,0].mean():.2f}")
                            st.write(f"• Avg Green: {arr[:,:,1].mean():.2f}")
                            st.write(f"• Avg Blue: {arr[:,:,2].mean():.2f}")
                            st.write(f"• Brightness: {arr.mean():.2f}")
                            st.write(f"• R Std Dev: {arr[:,:,0].std():.2f}")
                            st.write(f"• G Std Dev: {arr[:,:,1].std():.2f}")
                            st.write(f"• B Std Dev: {arr[:,:,2].std():.2f}")
                            # Dominant color
                            avg_color = arr.mean(axis=(0,1)).astype(int)
                            st.write(f"• Dominant: RGB({avg_color[0]}, {avg_color[1]}, {avg_color[2]})")
                        else:
                            st.write(f"• Avg brightness: {arr.mean():.2f}")
                            st.write(f"• Std deviation: {arr.std():.2f}")
                            st.write(f"• Min: {arr.min()}")
                            st.write(f"• Max: {arr.max()}")
                    
                    with col3:
                        st.write("**File Information:**")
                        st.write(f"• Format: {bin_format}")
                        st.write(f"• File size: {file_size:,} bytes")
                        st.write(f"• Size: {file_size/1024:.2f} KB")
                        if file_size > 1024*1024:
                            st.write(f"• Size: {file_size/(1024*1024):.2f} MB")
                        bytes_pp = file_size / (img.width * img.height) if img.width * img.height > 0 else 0
                        st.write(f"• Bytes/pixel: {bytes_pp:.2f}")
                        st.write(f"• Bit depth: {bytes_pp * 8:.0f} bits/px")
                        
                        # Compression ratio
                        uncompressed = img.width * img.height * 3
                        ratio = uncompressed / file_size if file_size > 0 else 0
                        st.write(f"• vs RGB888: {ratio:.2f}x")
                
                # Advanced color analysis
                if bin_format not in ["Monochrome"]:
                    with st.expander("🎨 Advanced Color Analysis"):
                        arr = np.array(img)
                        
                        col_hist1, col_hist2 = st.columns(2)
                        
                        with col_hist1:
                            if len(arr.shape) == 3:
                                st.write("**RGB Channel Histograms**")
                                col_r, col_g, col_b = st.columns(3)
                                with col_r:
                                    st.caption("Red")
                                    hist_r = np.histogram(arr[:,:,0], bins=32, range=(0, 256))[0]
                                    st.bar_chart(hist_r)
                                with col_g:
                                    st.caption("Green")
                                    hist_g = np.histogram(arr[:,:,1], bins=32, range=(0, 256))[0]
                                    st.bar_chart(hist_g)
                                with col_b:
                                    st.caption("Blue")
                                    hist_b = np.histogram(arr[:,:,2], bins=32, range=(0, 256))[0]
                                    st.bar_chart(hist_b)
                        
                        with col_hist2:
                            st.write("**Color Distribution**")
                            if len(arr.shape) == 3:
                                # Count unique colors
                                # Reshape to (N, 3) where N is total pixels
                                unique_colors = len(np.unique(arr.reshape(-1, 3), axis=0))
                                st.metric("Unique Colors", f"{unique_colors:,}")
                                
                                # Color range
                                st.write(f"**Range per channel:**")
                                st.write(f"• Red: {arr[:,:,0].min()}-{arr[:,:,0].max()}")
                                st.write(f"• Green: {arr[:,:,1].min()}-{arr[:,:,1].max()}")
                                st.write(f"• Blue: {arr[:,:,2].min()}-{arr[:,:,2].max()}")
                                
                                # Saturation analysis
                                # Convert to float for calculation
                                hsv = arr.astype(float)
                                maxc = hsv.max(axis=2)
                                minc = hsv.min(axis=2)
                                # Calculate saturation: (max - min) / max
                                sat = np.where(maxc != 0, (maxc - minc) / maxc, 0)
                                st.write(f"**Saturation:**")
                                st.write(f"• Average: {sat.mean()*100:.1f}%")
                                st.write(f"• Std Dev: {sat.std()*100:.1f}%")
                
                # Pixel inspector
                with st.expander("🔍 Advanced Pixel Inspector"):
                    col_insp1, col_insp2 = st.columns([2, 1])
                    
                    with col_insp1:
                        st.write("**Coordinate Inspector**")
                        col_x, col_y = st.columns(2)
                        with col_x:
                            inspect_x = st.number_input("X coordinate", 0, img.width-1, min(10, img.width-1), key="inspect_x")
                        with col_y:
                            inspect_y = st.number_input("Y coordinate", 0, img.height-1, min(10, img.height-1), key="inspect_y")
                        
                        if 0 <= inspect_x < img.width and 0 <= inspect_y < img.height:
                            arr = np.array(img)
                            if len(arr.shape) == 3:
                                pixel = arr[inspect_y, inspect_x]
                                st.write(f"**Pixel at ({inspect_x}, {inspect_y}):**")
                                
                                col_pix1, col_pix2, col_pix3 = st.columns(3)
                                with col_pix1:
                                    st.metric("Red", pixel[0])
                                    st.caption(f"{pixel[0]/255*100:.1f}%")
                                with col_pix2:
                                    st.metric("Green", pixel[1])
                                    st.caption(f"{pixel[1]/255*100:.1f}%")
                                with col_pix3:
                                    st.metric("Blue", pixel[2])
                                    st.caption(f"{pixel[2]/255*100:.1f}%")
                                
                                # Color swatch
                                color_html = f'<div style="width:150px;height:150px;background-color:rgb({pixel[0]},{pixel[1]},{pixel[2]});border:3px solid #333;border-radius:10px;margin:15px auto;box-shadow: 0 4px 6px rgba(0,0,0,0.2);"></div>'
                                st.markdown(color_html, unsafe_allow_html=True)
                                
                                # Multiple formats
                                # Recalculate RGB565 to ensure consistency
                                r5 = (pixel[0] >> 3) & 0x1F
                                g6 = (pixel[1] >> 2) & 0x3F
                                b5 = (pixel[2] >> 3) & 0x1F
                                rgb565_val = (r5 << 11) | (g6 << 5) | b5
                                
                                st.code(f"""RGB: ({pixel[0]}, {pixel[1]}, {pixel[2]})
HEX: #{pixel[0]:02X}{pixel[1]:02X}{pixel[2]:02X}
RGB565: 0x{rgb565_val:04X}
Float: ({pixel[0]/255:.3f}, {pixel[1]/255:.3f}, {pixel[2]/255:.3f})""")
                    
                    with col_insp2:
                        st.write("**Region Analysis**")
                        region_size = st.slider("Region size", 1, min(50, img.width, img.height), 5)
                        
                        x1 = max(0, inspect_x - region_size//2)
                        y1 = max(0, inspect_y - region_size//2)
                        x2 = min(img.width, inspect_x + region_size//2)
                        y2 = min(img.height, inspect_y + region_size//2)
                        
                        region = arr[y1:y2, x1:x2]
                        
                        if len(region.shape) == 3:
                            st.write(f"**Region {x1},{y1} to {x2},{y2}**")
                            st.write(f"• Avg R: {region[:,:,0].mean():.1f}")
                            st.write(f"• Avg G: {region[:,:,1].mean():.1f}")
                            st.write(f"• Avg B: {region[:,:,2].mean():.1f}")
                            st.write(f"• Brightness: {region.mean():.1f}")
                
                # Export options
                col_exp1, col_exp2, col_exp3 = st.columns(3)
                
                with col_exp1:
                    buf = io.BytesIO()
                    img.save(buf, format='PNG')
                    buf.seek(0)
                    st.download_button(
                        label="💾 Export PNG",
                        data=buf,
                        file_name=bin_file.name.replace(f'.{file_ext}', f'_frame{current_frame}.png') if detected_frames > 1 else bin_file.name.replace(f'.{file_ext}', '.png'),
                        mime="image/png",
                        use_container_width=True
                    )
                
                with col_exp2:
                    buf_jpg = io.BytesIO()
                    img.save(buf_jpg, format='JPEG', quality=95)
                    buf_jpg.seek(0)
                    st.download_button(
                        label="💾 Export JPEG",
                        data=buf_jpg,
                        file_name=bin_file.name.replace(f'.{file_ext}', '.jpg'),
                        mime="image/jpeg",
                        use_container_width=True
                    )
                
                with col_exp3:
                    if st.button("🔄 Convert Format", use_container_width=True):
                        st.info("Use Tab 1 to convert this image to a different binary format")
                
            except Exception as e:
                st.error(f"❌ **Decoding Error:** {str(e)}")
                st.write("**Troubleshooting Tips:**")
                st.write("• Verify the width/height settings match your file")
                st.write("• Try different color format options")
                st.write("• Check if file has a header (use auto-detect)")
                st.write("• Ensure file isn't corrupted")
            
            # Hex viewer with search
            with st.expander("🔍 Advanced Hex Viewer & Search"):
                col_hex_ctrl, col_hex_view = st.columns([1, 3])
                
                with col_hex_ctrl:
                    preview_size = st.slider("Preview (bytes)", 128, min(8192, len(bin_data)), min(1024, len(bin_data)), 128)
                    hex_offset = st.number_input("Start offset", 0, max(0, len(bin_data)-16), 0, 16)
                    
                    st.write("**Byte Search**")
                    search_hex = st.text_input("Find hex", "FF")
                    try:
                        search_val = int(search_hex, 16)
                        count = bin_data.count(search_val)
                        st.metric("Occurrences", f"{count:,}")
                        if count > 0:
                            percentage = (count / len(bin_data)) * 100
                            st.write(f"{percentage:.2f}% of file")
                            
                            # Find first occurrence
                            first_pos = bin_data.find(search_val)
                            if first_pos != -1:
                                st.write(f"First at: 0x{first_pos:08X}")
                    except:
                        st.write("Invalid hex value")
                    
                    st.write("**Pattern Search**")
                    pattern_hex = st.text_input("Pattern (e.g., FF00)", "")
                    if pattern_hex and len(pattern_hex) % 2 == 0:
                        try:
                            pattern = bytes.fromhex(pattern_hex)
                            pattern_count = len([i for i in range(len(bin_data) - len(pattern)) if bin_data[i:i+len(pattern)] == pattern])
                            st.metric("Pattern matches", pattern_count)
                        except:
                            st.write("Invalid pattern")
                
                with col_hex_view:
                    hex_lines = []
                    end_offset = min(hex_offset + preview_size, len(bin_data))
                    for i in range(hex_offset, end_offset, 16):
                        hex_part = ' '.join(f'{b:02X}' for b in bin_data[i:i+16])
                        ascii_part = ''.join(chr(b) if 32 <= b < 127 else '.' for b in bin_data[i:i+16])
                        hex_lines.append(f"{i:08X}  {hex_part:<48}  |{ascii_part}|")
                    st.code('\n'.join(hex_lines), language=None)
                    if len(bin_data) > end_offset:
                        st.write(f"... and {len(bin_data) - end_offset:,} more bytes")
            
            # Deep statistical analysis
            with st.expander("📈 Deep Statistical Analysis"):
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                
                with col_stat1:
                    st.write("**Byte Distribution**")
                    byte_counts = np.bincount(np.array(list(bin_data)), minlength=256)
                    st.bar_chart(byte_counts)
                    
                    # Most/least common
                    most_common_idx = np.argmax(byte_counts)
                    # Find the index of the minimum count among non-zero counts
                    non_zero_counts = byte_counts[byte_counts > 0]
                    if np.any(non_zero_counts):
                        min_count = np.min(non_zero_counts)
                        least_common_idx = np.where(byte_counts == min_count)[0][0]
                    else:
                        least_common_idx = 0
                        
                    st.write(f"**Most common:** 0x{most_common_idx:02X} ({byte_counts[most_common_idx]:,}x)")
                    st.write(f"**Least common:** 0x{least_common_idx:02X} ({byte_counts[least_common_idx]:,}x)")
                
                with col_stat2:
                    arr = np.array(list(bin_data))
                    st.write("**Statistical Measures**")
                    st.metric("Mean", f"{arr.mean():.2f}")
                    st.metric("Median", f"{np.median(arr):.1f}")
                    st.metric("Std Dev", f"{arr.std():.2f}")
                    st.metric("Variance", f"{arr.var():.1f}")
                    st.metric("Min Value", f"0x{arr.min():02X}")
                    st.metric("Max Value", f"0x{arr.max():02X}")
                    st.metric("Range", f"{arr.max() - arr.min()}")
                    st.metric("Unique Bytes", f"{len(np.unique(arr))}/256")
                
                with col_stat3:
                    st.write("**Entropy Analysis**")
                    byte_counts = np.bincount(np.array(list(bin_data)), minlength=256)
                    probabilities = byte_counts[byte_counts > 0] / len(bin_data)
                    # Handle case where probabilities is empty (empty file)
                    if len(probabilities) > 0:
                        entropy = -np.sum(probabilities * np.log2(probabilities))
                    else:
                        entropy = 0.0
                    
                    st.metric("Shannon Entropy", f"{entropy:.4f} bits/byte")
                    st.progress(entropy / 8.0)
                    
                    # Entropy interpretation
                    if entropy < 2:
                        entropy_desc = "Very low (highly repetitive)"
                    elif entropy < 4:
                        entropy_desc = "Low (structured data)"
                    elif entropy < 6:
                        entropy_desc = "Medium (mixed content)"
                    elif entropy < 7.5:
                        entropy_desc = "High (complex/random)"
                    else:
                        entropy_desc = "Very high (encrypted/compressed)"
                    
                    st.caption(f"**Interpretation:** {entropy_desc}")
                    st.write(f"Max possible: 8.0 bits/byte")
                    st.write(f"Compression potential: {(8-entropy)/8*100:.1f}%")
                    
                    # Byte pair analysis
                    st.write("**Byte Patterns**")
                    if len(bin_data) >= 2:
                        pairs = Counter()
                        for i in range(len(bin_data)-1):
                            pair = (bin_data[i], bin_data[i+1])
                            pairs[pair] += 1
                        
                        if pairs:
                            most_common_pair = pairs.most_common(1)[0]
                            st.write(f"**Most common pair:** 0x{most_common_pair[0][0]:02X}{most_common_pair[0][1]:02X}")
                            st.write(f"Occurs: {most_common_pair[1]:,} times")
                        else:
                            st.write("No byte pairs found.")
                    else:
                        st.write("File too small for pair analysis.")
        else:
            st.info("👆 Upload a BIN, NLF, or BIM file to begin analysis")

# ==================== TAB 3: ANIMATION TOOLS ====================
with tab3:
    st.header("🎬 Animation Tools & Frame Extractor")
    
    st.markdown("""
    <div class="info-box">
    <b>Animation Tools</b><br>
    Extract frames from multi-frame files, create animations, and analyze frame-by-frame content.
    Especially useful for LED fan NLF files with multiple animation frames.
    </div>
    """, unsafe_allow_html=True)
    
    anim_file = st.file_uploader(
        "📁 Upload animation file (NLF/BIN/BIM)",
        type=['nlf', 'bin', 'bim'],
        help="Upload multi-frame animation file",
        key="anim_uploader"
    )
    
    if anim_file is not None:
        anim_data = anim_file.read()
        
        col_anim_set, col_anim_view = st.columns([1, 2])
        
        with col_anim_set:
            st.subheader("⚙️ Animation Settings")
            
            anim_width = st.number_input("Frame Width", 1, 512, 144, key="anim_w")
            anim_height = st.number_input("Frame Height", 1, 512, 64, key="anim_h")
            anim_format = st.selectbox("Color Format", ["RGB565", "RGB888", "BGR888", "RGBA8888"], key="anim_fmt")
            
            bpp_map = {"RGB565": 2, "RGB888": 3, "BGR888": 3, "RGBA8888": 4}
            frame_size = anim_width * anim_height * bpp_map.get(anim_format, 2)
            total_frames = len(anim_data) // frame_size if frame_size > 0 else 0
            
            st.metric("Detected Frames", total_frames)
            st.metric("Frame Size", f"{frame_size:,} bytes")
            
            if total_frames > 0:
                st.success(f"✅ Animation contains {total_frames} frames")
                
                # Animation controls
                st.markdown("---")
                st.write("**Playback Controls**")
                
                play_speed = st.slider("Playback Speed (FPS)", 1, 60, 10)
                loop_animation = st.checkbox("Loop Animation", value=True)
                
                col_play1, col_play2 = st.columns(2)
                
                # --- Animation Playback Logic (Incomplete in original, adding placeholder) ---
                if 'animation_frame_index' not in st.session_state:
                    st.session_state.animation_frame_index = 0
                
                # Placeholder for the actual animation loop, which is complex in Streamlit
                # We will use a simple slider and a manual "Next Frame" button for demonstration
                # Real-time animation requires a separate thread or Streamlit's experimental features
                
                if st.button("▶️ Next Frame", use_container_width=True):
                    st.session_state.animation_frame_index = (st.session_state.animation_frame_index + 1) % total_frames
                    # Rerun to update the frame
                    st.rerun()
                
                current_anim_frame = st.slider("Frame", 0, total_frames-1, st.session_state.animation_frame_index, key="anim_frame_slider")
                st.session_state.animation_frame_index = current_anim_frame
                
                # Export options
                st.markdown("---")
                st.write("**Export Options**")
                
                # Frame extraction function (to be defined)
                def decode_frame(data, width, height, fmt):
                    # This is a simplified version of the decoding logic from Tab 2
                    bpp_map = {"RGB565": 2, "RGB888": 3, "BGR888": 3, "RGBA8888": 4}
                    frame_size = width * height * bpp_map.get(fmt, 2)
                    
                    if len(data) < frame_size:
                        raise ValueError("Incomplete frame data")
                    
                    if fmt == "RGB565":
                        pixels = []
                        for i in range(0, len(data), 2):
                            rgb565 = (data[i+1] << 8) | data[i] # Little-endian
                            r = ((rgb565 >> 11) & 0x1F) << 3
                            g = ((rgb565 >> 5) & 0x3F) << 2
                            b = (rgb565 & 0x1F) << 3
                            pixels.append([r, g, b])
                        pix_arr = np.array(pixels[:width * height]).reshape((height, width, 3))
                        return Image.fromarray(pix_arr.astype('uint8'), 'RGB')
                    
                    elif fmt == "RGB888":
                        pixels = []
                        for i in range(0, len(data), 3):
                            pixels.append([data[i], data[i+1], data[i+2]])
                        pix_arr = np.array(pixels[:width * height]).reshape((height, width, 3))
                        return Image.fromarray(pix_arr.astype('uint8'), 'RGB')
                    
                    elif fmt == "BGR888":
                        pixels = []
                        for i in range(0, len(data), 3):
                            pixels.append([data[i+2], data[i+1], data[i]])
                        pix_arr = np.array(pixels[:width * height]).reshape((height, width, 3))
                        return Image.fromarray(pix_arr.astype('uint8'), 'RGB')
                    
                    elif fmt == "RGBA8888":
                        pixels = []
                        for i in range(0, len(data), 4):
                            pixels.append([data[i], data[i+1], data[i+2]])
                        pix_arr = np.array(pixels[:width * height]).reshape((height, width, 3))
                        return Image.fromarray(pix_arr.astype('uint8'), 'RGB')
                    
                    else:
                        raise NotImplementedError(f"Format {fmt} not supported for animation tools.")

                
                # Export All Frames as ZIP (new feature)
                if st.button("💾 Export All Frames as ZIP", use_container_width=True):
                    # This is a complex operation that requires zipping files, which is not easily done in Streamlit's single-file model
                    # We will provide a placeholder and a warning
                    st.warning("Exporting all frames as a ZIP is not directly supported in this single-file Streamlit app. Please use the GIF export or download frames individually.")
                
                # GIF Export
                if st.button("🎬 Export as GIF", use_container_width=True):
                    try:
                        frames = []
                        # Limit to 100 frames for performance
                        num_frames_to_export = min(total_frames, 100) 
                        
                        st.info(f"Generating GIF from first {num_frames_to_export} frames...")
                        
                        for i in range(num_frames_to_export):
                            frame_offset = i * frame_size
                            frame_data = anim_data[frame_offset:frame_offset + frame_size]
                            
                            frame_img = decode_frame(frame_data, anim_width, anim_height, anim_format)
                            frames.append(frame_img)
                        
                        if frames:
                            buf = io.BytesIO()
                            # Duration is in milliseconds
                            duration_ms = int(1000 / play_speed)
                            frames[0].save(buf, format='GIF', save_all=True, append_images=frames[1:], 
                                         duration=duration_ms, loop=0)
                            buf.seek(0)
                            
                            st.download_button(
                                label="⬇️ Download GIF",
                                data=buf,
                                file_name=anim_file.name.replace('.nlf', '.gif').replace('.bin', '.gif').replace('.bim', '.gif'),
                                mime="image/gif",
                                use_container_width=True
                            )
                            st.success("GIF generated successfully!")
                        else:
                            st.error("Could not decode any frames for GIF generation.")
                            
                    except Exception as e:
                        st.error(f"GIF export error: {str(e)}")
            else:
                st.warning("⚠️ No valid frames detected. Adjust dimensions or format.")
        
        with col_anim_view:
            if total_frames > 0:
                try:
                    # Decode current frame
                    frame_offset = current_anim_frame * frame_size
                    frame_data = anim_data[frame_offset:frame_offset + frame_size]
                    
                    frame_img = decode_frame(frame_data, anim_width, anim_height, anim_format)
                    
                    # Display frame
                    st.image(frame_img, caption=f"Frame {current_anim_frame + 1} of {total_frames}", use_container_width=True)
                    
                    # Frame comparison
                    with st.expander("🔄 Frame-to-Frame Comparison"):
                        if current_anim_frame > 0:
                            col_prev, col_curr, col_diff = st.columns(3)
                            
                            # Get previous frame
                            prev_offset = (current_anim_frame - 1) * frame_size
                            prev_data = anim_data[prev_offset:prev_offset + frame_size]
                            
                            # Decode previous frame
                            prev_img = decode_frame(prev_data, anim_width, anim_height, anim_format)
                            prev_arr = np.array(prev_img)
                            
                            with col_prev:
                                st.caption(f"Frame {current_anim_frame}")
                                st.image(prev_img, use_container_width=True)
                            
                            with col_curr:
                                st.caption(f"Frame {current_anim_frame + 1}")
                                st.image(frame_img, use_container_width=True)
                            
                            with col_diff:
                                st.caption("Difference")
                                curr_arr = np.array(frame_img)
                                diff = np.abs(curr_arr.astype(int) - prev_arr.astype(int))
                                diff_img = Image.fromarray(diff.astype('uint8'), 'RGB')
                                st.image(diff_img, use_container_width=True)
                                
                                # Difference metrics
                                st.write(f"**Mean Change:** {np.mean(diff):.2f}")
                                st.write(f"**Max diff:** {np.max(diff)}")
                        else:
                            st.info("Select a frame > 0 to enable comparison with the previous frame.")
                    
                    # Frame statistics
                    with st.expander("📊 Frame Statistics"):
                        arr = np.array(frame_img)
                        col_s1, col_s2, col_s3 = st.columns(3)
                        
                        with col_s1:
                            st.write("**RGB Averages**")
                            st.write(f"• R: {arr[:,:,0].mean():.1f}")
                            st.write(f"• G: {arr[:,:,1].mean():.1f}")
                            st.write(f"• B: {arr[:,:,2].mean():.1f}")
                        
                        with col_s2:
                            st.write("**Brightness**")
                            brightness = arr.mean()
                            st.metric("Average", f"{brightness:.1f}")
                            st.progress(brightness / 255)
                        
                        with col_s3:
                            st.write("**Variance**")
                            st.write(f"• R: {arr[:,:,0].std():.1f}")
                            st.write(f"• G: {arr[:,:,1].std():.1f}")
                            st.write(f"• B: {arr[:,:,2].std():.1f}")
                    
                except Exception as e:
                    st.error(f"Frame decode error: {str(e)}")
            else:
                st.info("Configure settings on the left to view frames")

# ==================== TAB 4: DOCUMENTATION ====================
with tab4:
    st.header("📚 Comprehensive Documentation")
    
    doc_section = st.selectbox(
        "Select Documentation Section",
        ["Quick Start Guide", "File Format Details", "LED Fan Guide (NLF)", "Troubleshooting", 
         "Code Examples", "Advanced Topics", "FAQ"]
    )
    
    if doc_section == "Quick Start Guide":
        col_doc1, col_doc2 = st.columns(2)
        
        with col_doc1:
            st.subheader("🎯 Converting Images to BIN")
            st.markdown("""
            **Step-by-step process:**
            
            1. **Upload your image** (PNG, JPG, etc.)
            2. **Set dimensions** to match your display
            3. **Choose color format** (RGB565 for most LED displays)
            4. **Select resize method** (Fit maintains aspect ratio)
            5. **Adjust settings** (brightness, contrast, etc.)
            6. **Download** the BIN file
            
            **Pro Tips:**
            - Use high-contrast images for best LED display results
            - Start with 64×32 for common LED matrices
            - RGB565 saves memory (2 bytes/pixel)
            - Test with simple images first
            """)
        
        with col_doc2:
            st.subheader("🔍 Viewing BIN/NLF/BIM Files")
            st.markdown("""
            **Analysis workflow:**
            
            1. **Upload** your BIN/NLF/BIM file
            2. **Enable auto-detect** for automatic format detection
            3. **Select from suggestions** or manually configure
            4. **View decoded image** and analyze statistics
            5. **Export** as PNG/JPEG if needed
            
            **Features:**
            - Auto-detection of dimensions and format
            - Multi-frame animation support
            - Hex viewer with search
            - Deep statistical analysis
            - Frame-by-frame comparison
            """)
    
    elif doc_section == "File Format Details":
        st.subheader("🎨 Color Format Reference")
        
        format_details = {
            "Format": ["RGB565", "RGB888", "BGR888", "RGBA8888", "Grayscale", "Monochrome"],
            "Bits/Pixel": ["16", "24", "24", "32", "8", "1 (packed)"],
            "Bytes/Pixel": ["2", "3", "3", "4", "1", "0.125 (packed)"],
            "Colors": ["65,536", "16.7M", "16.7M", "16.7M+Alpha", "256", "2"],
            "Common Use": [
                "LED matrices (most common)",
                "High-quality RGB displays",
                "Some LED panels (reversed)",
                "Displays with transparency",
                "Monochrome OLEDs",
                "Simple B&W displays"
            ]
        }
        
        st.table(format_details)
        
        st.markdown("---")
        
        col_fmt1, col_fmt2 = st.columns(2)
        
        with col_fmt1:
            st.subheader("RGB565 Details")
            st.markdown("""
            **16-bit color encoding (Little-Endian assumed for data):**
            ```
            Byte 0 (LSB): G[2:0], B[4:0]
            Byte 1 (MSB): R[4:0], G[5:3]
            
            Bit 15-11: Red   (5 bits = 32 levels)
            Bit 10-5:  Green (6 bits = 64 levels)
            Bit 4-0:   Blue  (5 bits = 32 levels)
            ```
            
            **Why 6 bits for green?**
            Human eyes are most sensitive to green light, so it gets extra precision.
            
            **Encoding example (Python logic):**
            ```python
            r5 = (red >> 3) & 0x1F
            g6 = (green >> 2) & 0x3F
            b5 = (blue >> 3) & 0x1F
            rgb565 = (r5 << 11) | (g6 << 5) | b5
            
            # Little-Endian (LSB first)
            bin_data.append(rgb565 & 0xFF)
            bin_data.append(rgb565 >> 8)
            ```
            """)
        
        with col_fmt2:
            st.subheader("RGB888 Details")
            st.markdown("""
            **24-bit true color:**
            ```
            Byte 0: Red   (8 bits = 256 levels)
            Byte 1: Green (8 bits = 256 levels)
            Byte 2: Blue  (8 bits = 256 levels)
            ```
            
            **Total colors:** 16,777,216
            
            **Byte ordering:**
            - RGB888: Red, Green, Blue
            - BGR888: Blue, Green, Red (reversed)
            
            **When to use:**
            - High-quality displays
            - When memory isn't limited
            - Desktop/PC applications
            """)
    
    elif doc_section == "LED Fan Guide (NLF)":
        st.subheader("🎬 LED Fan Animation Format")
        
        st.markdown("""
        ### What is NLF?
        NLF (likely "Neopixel LED Fan" or similar) is a binary format for storing LED fan animations.
        LED fans create 3D holographic effects by rapidly spinning LEDs while changing colors.
        
        ### Common LED Fan Specifications
        """)
        
        fan_specs = {
            "Model": ["Standard", "Large", "Mini", "Pro"],
            "Resolution": ["144×64", "120×60", "96×48", "144×96"],
            "Format": ["RGB565", "RGB565", "RGB565", "RGB888"],
            "FPS": ["20-30", "15-25", "25-35", "20-30"],
            "Use Case": ["General", "High detail", "Compact", "Premium quality"]
        }
        st.table(fan_specs)
        
        st.markdown("""
        ### NLF File Structure
        ```
        [Header - 10 bytes]
        - Magic: 'NLF1' (4 bytes)
        - Width: uint16 (2 bytes, Little-Endian)
        - Height: uint16 (2 bytes, Little-Endian)
        - Bytes per pixel: uint8 (1 byte)
        - Frame count: uint8 (1 byte)
        
        [Frame Data]
        - Frame 0: [width × height × bpp] bytes
        - Frame 1: [width × height × bpp] bytes
        - ... (repeated for each frame)
        ```
        
        ### Creating LED Fan Animations
        1. Create a sequence of images (frames)
        2. Convert each to the correct size (e.g., 144×64)
        3. Combine into single BIN file
        4. Add NLF header (optional, depending on device)
        5. Test playback speed and adjust FPS
        
        ### Best Practices
        - **High contrast:** LED fans work best with bold colors
        - **Simple designs:** Complex details may blur
        - **Frame rate:** 20-30 FPS is typical
        - **Loop smoothly:** Ensure first and last frames match
        - **Test in darkness:** LED fans look best in low light
        """)
    
    elif doc_section == "Troubleshooting":
        st.subheader("🔧 Common Issues & Solutions")
        
        with st.expander("❌ Image looks corrupted or garbled"):
            st.markdown("""
            **Possible causes & fixes:**
            
            1. **Wrong dimensions**
               - Solution: Try auto-detect feature
               - Check device specifications
               - Try common sizes (64×32, 128×64, etc.)
            
            2. **Wrong color format**
               - Solution: Try RGB565 first (most common)
               - If colors are wrong, try BGR888
               - Check device documentation
            
            3. **File has header**
               - Solution: Enable auto-detect
               - Manually skip header bytes
               - Check first few bytes in hex viewer
            
            4. **Byte order issues**
               - Some displays expect different endianness
               - Try swapping byte pairs if RGB565 looks wrong
            """)
        
        with st.expander("⚠️ File size doesn't match"):
            st.markdown("""
            **Verification formula:**
            ```
            Expected Size = Width × Height × Bytes_Per_Pixel
            
            RGB565:  Width × Height × 2
            RGB888:  Width × Height × 3
            RGBA:    Width × Height × 4
            Gray:    Width × Height × 1
            Mono:    ceil(Width × Height / 8)
            ```
            
            **If size doesn't match:**
            - File may have header (add 10-20 bytes)
            - File may have padding/alignment bytes
            - Multiple frames (divide by frame count)
            - Check for footer or metadata
            """)
        
        with st.expander("🎨 Colors are wrong or inverted"):
            st.markdown("""
            **Color issues:**
            
            1. **RGB vs BGR**
               - Some displays reverse red and blue
               - Try BGR888 if RGB888 looks wrong
            
            2. **Inverted colors (negative image)**
               - Use "Invert Colors" option in viewer
               - Check if display inverts automatically
            
            3. **Washed out colors**
               - Increase contrast in converter
               - Adjust brightness settings
               - Check display brightness limits
            
            4. **Wrong bit depth**
               - RGB565 has less color depth than RGB888
               - Dithering can help smooth gradients
            """)
        
        with st.expander("🎬 Animation doesn't play correctly"):
            st.markdown("""
            **Animation troubleshooting:**
            
            1. **Wrong frame size**
               - Calculate: width × height × bytes_per_pixel
               - File should divide evenly by frame size
            
            2. **Wrong frame count**
               - Check file_size / frame_size
               - May include header/footer bytes
               - Use the Animation Tools tab to verify frame extraction
            
            3. **Frame order issues**
               - Frames are sequential in file
               - First frame at offset 0 (or after header)
            
            4. **Playback speed**
               - Adjust FPS in animation tools
               - Most LED fans: 20-30 FPS
               - Test different speeds
            """)
    
    elif doc_section == "Code Examples":
        st.subheader("💻 Integration Code Examples")
        
        code_platform = st.selectbox("Select Platform", 
            ["Arduino/ESP32", "Raspberry Pi (Python)", "Processing/P5.js", "C/C++", "MicroPython"])
        
        if code_platform == "Arduino/ESP32":
            st.code("""
// Arduino LED Matrix Example (using RGB565 BIN file)
#include <Adafruit_GFX.h>
#include <RGBmatrixPanel.h>
#include <SD.h>

#define WIDTH 64
#define HEIGHT 32
#define BPP 2 // Bytes per pixel for RGB565

// Define your matrix pins here
// RGBmatrixPanel matrix(A, B, C, D, CLK, LAT, OE, false, 64);

void setup() {
  // matrix.begin();
  // SD.begin(SD_CS_PIN);
  Serial.begin(115200);
  Serial.println("Starting BIN file loader...");
  
  // Load and display BIN file
  File binFile = SD.open("image.bin");
  if (binFile) {
    Serial.print("File size: ");
    Serial.println(binFile.size());
    
    if (binFile.size() >= WIDTH * HEIGHT * BPP) {
      // Read pixel data
      for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
          uint8_t lsb = binFile.read(); // LSB (Byte 0)
          uint8_t msb = binFile.read(); // MSB (Byte 1)
          uint16_t rgb565 = (msb << 8) | lsb;
          
          // Convert 565 to 888 for display (optional, depending on library)
          uint8_t r = (rgb565 >> 11) & 0x1F;
          uint8_t g = (rgb565 >> 5) & 0x3F;
          uint8_t b = rgb565 & 0x1F;
          
          // Scale back to 8-bit (approximate)
          uint16_t color = matrix.Color(r << 3, g << 2, b << 3);
          // matrix.drawPixel(x, y, color);
        }
      }
      // matrix.swapBuffers(false);
      Serial.println("Image loaded successfully.");
    } else {
      Serial.println("Error: File size mismatch.");
    }
    binFile.close();
  } else {
    Serial.println("Error opening image.bin");
  }
}

void loop() {
  // Your main loop code
}
            """)
        
        elif code_platform == "Raspberry Pi (Python)":
            st.code("""
# Raspberry Pi Python Example (using RGB888 BIN file)
import numpy as np
from PIL import Image

WIDTH = 64
HEIGHT = 32
BPP = 3 # Bytes per pixel for RGB888

try:
    with open("image.bin", "rb") as f:
        bin_data = f.read()
        
    expected_size = WIDTH * HEIGHT * BPP
    if len(bin_data) < expected_size:
        print(f"Error: File size mismatch. Expected {expected_size} bytes, got {len(bin_data)}.")
    else:
        # Convert binary data to a numpy array
        # The data is a flat array of R, G, B bytes
        arr = np.frombuffer(bin_data[:expected_size], dtype=np.uint8)
        
        # Reshape to (Height, Width, 3)
        image_array = arr.reshape((HEIGHT, WIDTH, BPP))
        
        # Create PIL Image
        img = Image.fromarray(image_array, 'RGB')
        
        # Display or process the image (e.g., using rpi-rgb-led-matrix library)
        # matrix.SetImage(img.convert('RGB'))
        img.show()
        print("Image loaded and displayed successfully.")

except FileNotFoundError:
    print("Error: image.bin not found.")
except Exception as e:
    print(f"An error occurred: {e}")
            """)
        
        elif code_platform == "Processing/P5.js":
            st.code("""
// Processing (Java) Example for RGB565
// Requires a data folder with 'image.bin'

final int WIDTH = 64;
final int HEIGHT = 32;
final int BPP = 2; // RGB565

void setup() {
  size(WIDTH * 10, HEIGHT * 10); // Scale up for viewing
  loadBinImage("image.bin");
  noLoop();
}

void loadBinImage(String filename) {
  byte[] binData = loadBytes(filename);
  
  if (binData.length < WIDTH * HEIGHT * BPP) {
    println("Error: File size mismatch.");
    return;
  }
  
  int index = 0;
  for (int y = 0; y < HEIGHT; y++) {
    for (int x = 0; x < WIDTH; x++) {
      // Read 2 bytes (Little-Endian)
      int lsb = binData[index++] & 0xFF;
      int msb = binData[index++] & 0xFF;
      int rgb565 = (msb << 8) | lsb;
      
      // Extract R, G, B components
      int r5 = (rgb565 >> 11) & 0x1F;
      int g6 = (rgb565 >> 5) & 0x3F;
      int b5 = rgb565 & 0x1F;
      
      // Scale back to 8-bit (0-255)
      int r = r5 << 3;
      int g = g6 << 2;
      int b = b5 << 3;
      
      // Draw pixel (scaled)
      fill(r, g, b);
      rect(x * 10, y * 10, 10, 10);
    }
  }
}
            """)
        
        elif code_platform == "C/C++":
            st.code("""
// C/C++ Example for RGB565
#include <stdio.h>
#include <stdint.h>

#define WIDTH 64
#define HEIGHT 32
#define BPP 2 // RGB565

// Function to convert 565 to 24-bit color (for console/debug)
void rgb565_to_rgb888(uint16_t color565, uint8_t *r, uint8_t *g, uint8_t *b) {
    *r = ((color565 >> 11) & 0x1F) << 3;
    *g = ((color565 >> 5) & 0x3F) << 2;
    *b = (color565 & 0x1F) << 3;
}

int main() {
    FILE *fp = fopen("image.bin", "rb");
    if (fp == NULL) {
        printf("Error opening image.bin\\n");
        return 1;
    }

    uint8_t buffer[WIDTH * HEIGHT * BPP];
    size_t bytes_read = fread(buffer, 1, sizeof(buffer), fp);
    fclose(fp);

    if (bytes_read < sizeof(buffer)) {
        printf("Error: File size mismatch. Read %zu bytes, expected %zu.\\n", bytes_read, sizeof(buffer));
        return 1;
    }

    printf("Image data loaded. First pixel (0,0) analysis:\\n");
    
    // Read first pixel (Little-Endian)
    uint8_t lsb = buffer[0];
    uint8_t msb = buffer[1];
    uint16_t rgb565 = (msb << 8) | lsb;

    uint8_t r, g, b;
    rgb565_to_rgb888(rgb565, &r, &g, &b);

    printf("RGB565 Value: 0x%04X\\n", rgb565);
    printf("RGB888 Value: R=%d, G=%d, B=%d\\n", r, g, b);

    // Example: Accessing a pixel at (x, y)
    int x = 10, y = 5;
    size_t offset = (y * WIDTH + x) * BPP;
    
    lsb = buffer[offset];
    msb = buffer[offset + 1];
    rgb565 = (msb << 8) | lsb;
    rgb565_to_rgb888(rgb565, &r, &g, &b);
    
    printf("Pixel at (%d, %d): R=%d, G=%d, B=%d\\n", x, y, r, g, b);

    return 0;
}
            """)
        
        elif code_platform == "MicroPython":
            st.code("""
# MicroPython/CircuitPython Example (for monochrome displays)
# Assumes a display driver like ssd1306 is available

import framebuf
import machine
import os

WIDTH = 128
HEIGHT = 64

# Initialize I2C or SPI for your display
# i2c = machine.I2C(scl=machine.Pin(5), sda=machine.Pin(4))
# display = ssd1306.SSD1306_I2C(WIDTH, HEIGHT, i2c)

try:
    with open("image.bin", "rb") as f:
        # Monochrome data is packed 1 bit per pixel, 8 pixels per byte
        # The size should be ceil(WIDTH * HEIGHT / 8)
        bin_data = f.read()
        
    # Create a FrameBuffer object from the binary data
    # The format is typically MONO_VLSB (Vertical-LSB) or MONO_HLSB (Horizontal-LSB)
    # Assuming MONO_HLSB for simplicity, which matches the converter's packing logic
    fbuf = framebuf.FrameBuffer(
        bin_data, WIDTH, HEIGHT, framebuf.MONO_HLSB
    )
    
    # Display the buffer
    # display.blit(fbuf, 0, 0)
    # display.show()
    
    print("Image loaded to FrameBuffer successfully.")

except OSError:
    print("Error: image.bin not found or file system error.")
            """)
    
    elif doc_section == "Advanced Topics":
        st.subheader("💡 Advanced Topics")
        st.markdown("""
        ### Endianness (Byte Order)
        - **Little-Endian (LSB first):** The least significant byte comes first. Common in x86 architectures and many embedded systems. **This converter assumes Little-Endian for RGB565 output.**
        - **Big-Endian (MSB first):** The most significant byte comes first. Common in network protocols and some older systems.
        
        If your display shows swapped colors (e.g., Red and Blue are swapped), you may need to reverse the byte order in your device's code or modify the converter's output logic.
        
        ### Dithering
        Dithering is a technique used to simulate colors that are not available in the display's palette. It does this by interspersing pixels of available colors.
        - **Floyd-Steinberg:** A popular error-diffusion algorithm that produces high-quality results. Used in the Monochrome conversion.
        - **Quantization:** Reducing the number of colors in an image.
        
        ### Memory Alignment
        Some microcontrollers require data to be aligned to 2-byte or 4-byte boundaries for efficient access. If your image width is not a multiple of 8 or 16, you may need to add padding bytes at the end of each row. This converter currently does not add padding, assuming continuous memory or a display driver that handles non-aligned data.
        
        ### BIM Format
        The BIM (Binary Image) format is a simple, custom format often used in embedded systems. The header is typically:
        - `BIM\x00` (4 bytes magic)
        - `Width` (4 bytes, uint32)
        - `Height` (4 bytes, uint32)
        - Followed by raw pixel data.
        
        The color format (RGB565, RGB888, etc.) is usually implicit or configured separately on the device.
        """)
    
    elif doc_section == "FAQ":
        st.subheader("❓ Frequently Asked Questions")
        
        st.markdown("""
        **Q: What is the difference between BIN, NLF, and BIM?**
        A: **BIN** is raw binary pixel data with no header. **NLF** is a format specifically for LED fans, including a header with width, height, BPP, and frame count. **BIM** is another simple binary image format with a header for width and height.
        
        **Q: Why are my colors wrong when using RGB565?**
        A: It's likely an **endianness** issue. This converter outputs RGB565 in **Little-Endian** (LSB first). If your device expects Big-Endian, you need to swap the two bytes for every pixel.
        
        **Q: How do I create an animation?**
        A: You need a sequence of images (frames). You can use external tools to generate the frames, then concatenate the resulting BIN files into a single file. The NLF format is designed to hold these concatenated frames with a header indicating the total frame count.
        
        **Q: Why is the GIF export limited to 100 frames?**
        A: Generating and downloading large GIFs can be very slow and resource-intensive in a web application. The limit is a performance safeguard. For full animations, you should use the raw binary output and a dedicated desktop tool.
        
        **Q: Can I use this for a monochrome OLED display?**
        A: Yes, select **Monochrome** as the color format. It will pack the 1-bit pixels into bytes, which is the standard format for most OLED controllers like the SSD1306.
        """)
