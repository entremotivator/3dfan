import streamlit as st
from PIL import Image
import io
import numpy as np
import struct

st.set_page_config(page_title="Image ⇄ BIN Converter", page_icon="🖼️", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🖼️ Image ⇄ BIN Converter Pro</h1><p>Convert images to BIN files for LED displays • View and analyze BIN files</p></div>', unsafe_allow_html=True)

# Create tabs for different functions
tab1, tab2, tab3 = st.tabs(["📤 Image to BIN", "📥 BIN Viewer & Analyzer", "📚 Documentation"])

# ==================== TAB 1: IMAGE TO BIN ====================
with tab1:
    st.header("Convert Image to BIN File")
    
    col_settings, col_upload = st.columns([1, 2])
    
    with col_settings:
        st.subheader("⚙️ Display Settings")
        
        matrix_width = st.number_input("Width (pixels)", min_value=8, max_value=512, value=64, step=8, key="width1")
        matrix_height = st.number_input("Height (pixels)", min_value=8, max_value=512, value=32, step=8, key="height1")
        
        st.markdown("---")
        
        color_format = st.selectbox(
            "Color Format",
            ["RGB565", "RGB888", "BGR888", "RGB24", "Grayscale", "Monochrome"],
            help="RGB565: 16-bit (most common)\nRGB888: 24-bit full color\nBGR888: 24-bit (reversed)\nGrayscale: 8-bit\nMonochrome: 1-bit"
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
            dithering = st.checkbox("Apply Dithering", value=False, help="Better for low-color displays")
            flip_h = st.checkbox("Flip Horizontal", value=False)
            flip_v = st.checkbox("Flip Vertical", value=False)
            rotate = st.selectbox("Rotate", [0, 90, 180, 270], help="Degrees clockwise")
        
        # Quick presets
        with st.expander("⚡ Quick Presets"):
            if st.button("32×32 RGB LED", use_container_width=True):
                st.session_state.width1 = 32
                st.session_state.height1 = 32
                st.rerun()
            if st.button("64×32 RGB LED", use_container_width=True):
                st.session_state.width1 = 64
                st.session_state.height1 = 32
                st.rerun()
            if st.button("64×64 RGB LED", use_container_width=True):
                st.session_state.width1 = 64
                st.session_state.height1 = 64
                st.rerun()
            if st.button("128×64 OLED", use_container_width=True):
                st.session_state.width1 = 128
                st.session_state.height1 = 64
                st.rerun()
    
    with col_upload:
        uploaded_file = st.file_uploader(
            "📁 Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'gif', 'webp'],
            help="Supported formats: PNG, JPG, BMP, GIF, WebP"
        )
        
        if uploaded_file is not None:
            # Display images side by side
            col_orig, col_proc = st.columns(2)
            
            with col_orig:
                st.markdown("**Original Image**")
                original_image = Image.open(uploaded_file)
                st.image(original_image, use_container_width=True)
                st.caption(f"📏 {original_image.size[0]}×{original_image.size[1]} | Mode: {original_image.mode}")
            
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
            
            # Apply brightness and contrast
            if brightness != 100 or contrast != 100:
                from PIL import ImageEnhance
                if brightness != 100:
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(brightness / 100)
                if contrast != 100:
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(contrast / 100)
            
            # Apply dithering if needed
            if dithering and color_format in ["Monochrome", "Grayscale"]:
                img = img.convert('L').convert('1', dither=Image.FLOYDSTEINBERG)
            
            with col_proc:
                st.markdown("**Processed Image**")
                st.image(img, use_container_width=True)
                st.caption(f"📏 {img.size[0]}×{img.size[1]} | Format: {color_format}")
            
            # Convert to binary data
            pixels = np.array(img)
            bin_data = bytearray()
            bytes_per_pixel = 0
            
            if color_format == "RGB565":
                for row in pixels:
                    for pixel in row:
                        r, g, b = pixel[:3] if len(pixel) >= 3 else (pixel[0], pixel[0], pixel[0])
                        r5 = (r >> 3) & 0x1F
                        g6 = (g >> 2) & 0x3F
                        b5 = (b >> 3) & 0x1F
                        rgb565 = (r5 << 11) | (g6 << 5) | b5
                        bin_data.append(rgb565 >> 8)
                        bin_data.append(rgb565 & 0xFF)
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
                            bin_data.extend([pixel[2], pixel[1], pixel[0]])
                        else:
                            bin_data.extend([pixel[0], pixel[0], pixel[0]])
                bytes_per_pixel = 3
            
            elif color_format == "Grayscale":
                gray_img = img.convert('L')
                gray_pixels = np.array(gray_img)
                for row in gray_pixels:
                    bin_data.extend(row)
                bytes_per_pixel = 1
            
            elif color_format == "Monochrome":
                mono_img = img.convert('1')
                mono_pixels = np.array(mono_img)
                for row in mono_pixels:
                    byte_val = 0
                    bit_count = 0
                    for pixel in row:
                        byte_val = (byte_val << 1) | (1 if pixel else 0)
                        bit_count += 1
                        if bit_count == 8:
                            bin_data.append(byte_val)
                            byte_val = 0
                            bit_count = 0
                    if bit_count > 0:
                        byte_val <<= (8 - bit_count)
                        bin_data.append(byte_val)
                bytes_per_pixel = 0.125
            
            # Success message and metrics
            st.markdown('<div class="success-box">✅ <b>Conversion Complete!</b></div>', unsafe_allow_html=True)
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Resolution", f"{matrix_width}×{matrix_height}")
            with col_m2:
                st.metric("Format", color_format)
            with col_m3:
                st.metric("File Size", f"{len(bin_data):,} bytes")
            with col_m4:
                st.metric("Pixels", f"{matrix_width * matrix_height:,}")
            
            # Download button
            filename = uploaded_file.name.rsplit('.', 1)[0] + '.bin'
            st.download_button(
                label="⬇️ Download BIN File",
                data=bytes(bin_data),
                file_name=filename,
                mime="application/octet-stream",
                use_container_width=True
            )
            
            # Additional details
            with st.expander("📊 Detailed Information"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**File Information:**")
                    st.write(f"• Total pixels: {matrix_width * matrix_height:,}")
                    st.write(f"• Bytes per pixel: {bytes_per_pixel}")
                    st.write(f"• Total bytes: {len(bin_data):,}")
                    st.write(f"• File format: Raw binary ({color_format})")
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
                        st.write("• Total colors: 16,777,216")
                    elif color_format == "Grayscale":
                        st.write("• 8 bits per pixel")
                        st.write("• 256 gray levels")
                    elif color_format == "Monochrome":
                        st.write("• 1 bit per pixel")
                        st.write("• Black and white only")
            
            # Hex preview
            with st.expander("🔍 Hex Data Preview (First 512 bytes)"):
                preview_bytes = min(512, len(bin_data))
                hex_lines = []
                for i in range(0, preview_bytes, 16):
                    hex_part = ' '.join(f'{b:02X}' for b in bin_data[i:i+16])
                    ascii_part = ''.join(chr(b) if 32 <= b < 127 else '.' for b in bin_data[i:i+16])
                    hex_lines.append(f"{i:08X}  {hex_part:<48}  {ascii_part}")
                st.code('\n'.join(hex_lines), language=None)
                if len(bin_data) > preview_bytes:
                    st.write(f"... and {len(bin_data) - preview_bytes:,} more bytes")
        else:
            st.info("👆 Upload an image file to begin conversion")

# ==================== TAB 2: BIN VIEWER ====================
with tab2:
    st.header("BIN File Viewer & Analyzer")
    
    col_settings2, col_viewer = st.columns([1, 2])
    
    with col_settings2:
        st.subheader("⚙️ Display Settings")
        
        bin_width = st.number_input("Width (pixels)", min_value=8, max_value=512, value=64, step=8, key="width2")
        bin_height = st.number_input("Height (pixels)", min_value=8, max_value=512, value=32, step=8, key="height2")
        
        st.markdown("---")
        
        bin_format = st.selectbox(
            "Color Format",
            ["RGB565", "RGB888", "BGR888", "RGB24", "Grayscale", "Monochrome"],
            help="Format of the BIN file",
            key="bin_format"
        )
        
        st.markdown("---")
        
        with st.expander("🔧 View Options"):
            zoom_level = st.slider("Zoom Level", 1, 10, 4, help="Pixel size multiplier")
            show_grid = st.checkbox("Show Pixel Grid", value=False)
            auto_detect = st.checkbox("Auto-detect dimensions", value=True, help="Guess dimensions from file size")
    
    with col_viewer:
        bin_file = st.file_uploader(
            "📁 Upload BIN file",
            type=['bin'],
            help="Upload a .bin file to view and analyze",
            key="bin_uploader"
        )
        
        if bin_file is not None:
            bin_data = bin_file.read()
            file_size = len(bin_data)
            
            st.markdown('<div class="info-box">📄 <b>File loaded successfully!</b></div>', unsafe_allow_html=True)
            
            # Auto-detect dimensions
            if auto_detect:
                possible_dims = []
                for fmt_name, bpp in [("RGB565", 2), ("RGB888", 3), ("BGR888", 3), ("Grayscale", 1)]:
                    total_pixels = file_size / bpp
                    if total_pixels == int(total_pixels):
                        total_pixels = int(total_pixels)
                        # Common aspect ratios
                        for ratio_w, ratio_h in [(1, 1), (2, 1), (4, 3), (16, 9), (16, 10)]:
                            w = int((total_pixels * ratio_w / ratio_h) ** 0.5)
                            h = int(total_pixels / w)
                            if w * h == total_pixels and w * bpp * h == file_size:
                                possible_dims.append((w, h, fmt_name))
                
                if possible_dims:
                    st.info(f"💡 **Possible dimensions detected:** " + ", ".join([f"{w}×{h} ({fmt})" for w, h, fmt in possible_dims[:5]]))
            
            # File analysis
            col_a1, col_a2, col_a3, col_a4 = st.columns(4)
            with col_a1:
                st.metric("File Size", f"{file_size:,} bytes")
            with col_a2:
                st.metric("Format", bin_format)
            with col_a3:
                expected_size = bin_width * bin_height * (2 if bin_format == "RGB565" else 3 if "888" in bin_format else 1)
                st.metric("Expected Size", f"{expected_size:,} bytes")
            with col_a4:
                match = "✅ Match" if file_size == expected_size else "⚠️ Mismatch"
                st.metric("Size Check", match)
            
            # Try to decode and display
            try:
                if bin_format == "RGB565":
                    pixels = []
                    for i in range(0, len(bin_data), 2):
                        if i+1 < len(bin_data):
                            rgb565 = (bin_data[i] << 8) | bin_data[i+1]
                            r = ((rgb565 >> 11) & 0x1F) << 3
                            g = ((rgb565 >> 5) & 0x3F) << 2
                            b = (rgb565 & 0x1F) << 3
                            pixels.append([r, g, b])
                    
                    total_pixels = len(pixels)
                    actual_height = total_pixels // bin_width
                    pixels = np.array(pixels[:bin_width * actual_height]).reshape((actual_height, bin_width, 3))
                    img = Image.fromarray(pixels.astype('uint8'), 'RGB')
                
                elif bin_format in ["RGB888", "RGB24"]:
                    pixels = []
                    for i in range(0, len(bin_data), 3):
                        if i+2 < len(bin_data):
                            pixels.append([bin_data[i], bin_data[i+1], bin_data[i+2]])
                    
                    total_pixels = len(pixels)
                    actual_height = total_pixels // bin_width
                    pixels = np.array(pixels[:bin_width * actual_height]).reshape((actual_height, bin_width, 3))
                    img = Image.fromarray(pixels.astype('uint8'), 'RGB')
                
                elif bin_format == "BGR888":
                    pixels = []
                    for i in range(0, len(bin_data), 3):
                        if i+2 < len(bin_data):
                            pixels.append([bin_data[i+2], bin_data[i+1], bin_data[i]])
                    
                    total_pixels = len(pixels)
                    actual_height = total_pixels // bin_width
                    pixels = np.array(pixels[:bin_width * actual_height]).reshape((actual_height, bin_width, 3))
                    img = Image.fromarray(pixels.astype('uint8'), 'RGB')
                
                elif bin_format == "Grayscale":
                    pixels = np.array(list(bin_data))
                    total_pixels = len(pixels)
                    actual_height = total_pixels // bin_width
                    pixels = pixels[:bin_width * actual_height].reshape((actual_height, bin_width))
                    img = Image.fromarray(pixels.astype('uint8'), 'L')
                
                elif bin_format == "Monochrome":
                    pixels = []
                    for byte in bin_data:
                        for bit in range(8):
                            pixels.append(255 if (byte >> (7-bit)) & 1 else 0)
                    pixels = np.array(pixels)
                    total_pixels = len(pixels)
                    actual_height = total_pixels // bin_width
                    pixels = pixels[:bin_width * actual_height].reshape((actual_height, bin_width))
                    img = Image.fromarray(pixels.astype('uint8'), 'L')
                
                # Scale up for viewing
                view_img = img.resize((bin_width * zoom_level, img.height * zoom_level), Image.NEAREST)
                
                st.image(view_img, caption=f"Decoded Image ({img.width}×{img.height})", use_container_width=True)
                
                # Analysis
                with st.expander("📊 Image Analysis"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Dimensions:**")
                        st.write(f"• Width: {img.width} pixels")
                        st.write(f"• Height: {img.height} pixels")
                        st.write(f"• Total pixels: {img.width * img.height:,}")
                        st.write(f"• Aspect ratio: {img.width/img.height:.2f}:1")
                    
                    with col2:
                        if bin_format != "Monochrome":
                            arr = np.array(img)
                            st.write("**Color Statistics:**")
                            if len(arr.shape) == 3:
                                st.write(f"• Avg Red: {arr[:,:,0].mean():.1f}")
                                st.write(f"• Avg Green: {arr[:,:,1].mean():.1f}")
                                st.write(f"• Avg Blue: {arr[:,:,2].mean():.1f}")
                            else:
                                st.write(f"• Avg brightness: {arr.mean():.1f}")
                            st.write(f"• Min value: {arr.min()}")
                            st.write(f"• Max value: {arr.max()}")
                
                # Download as PNG
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                buf.seek(0)
                
                st.download_button(
                    label="💾 Export as PNG",
                    data=buf,
                    file_name=bin_file.name.replace('.bin', '.png'),
                    mime="image/png",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"❌ Error decoding BIN file: {str(e)}")
                st.write("Try adjusting the width/height or format settings.")
            
            # Hex viewer
            with st.expander("🔍 Raw Hex Data (First 512 bytes)"):
                preview_bytes = min(512, len(bin_data))
                hex_lines = []
                for i in range(0, preview_bytes, 16):
                    hex_part = ' '.join(f'{b:02X}' for b in bin_data[i:i+16])
                    ascii_part = ''.join(chr(b) if 32 <= b < 127 else '.' for b in bin_data[i:i+16])
                    hex_lines.append(f"{i:08X}  {hex_part:<48}  {ascii_part}")
                st.code('\n'.join(hex_lines), language=None)
                if len(bin_data) > preview_bytes:
                    st.write(f"... and {len(bin_data) - preview_bytes:,} more bytes")
            
            # Statistics
            with st.expander("📈 Byte Distribution"):
                byte_counts = np.bincount(np.array(list(bin_data)), minlength=256)
                st.bar_chart(byte_counts)
        else:
            st.info("👆 Upload a BIN file to view and analyze")

# ==================== TAB 3: DOCUMENTATION ====================
with tab3:
    st.header("📚 Documentation & Guide")
    
    col_doc1, col_doc2 = st.columns(2)
    
    with col_doc1:
        st.subheader("🎯 Getting Started")
        st.markdown("""
        ### Image to BIN Conversion
        1. **Upload** your image (PNG, JPG, BMP, GIF, WebP)
        2. **Set** display dimensions (width × height)
        3. **Choose** color format based on your display
        4. **Configure** resize method and advanced options
        5. **Download** the generated .bin file
        
        ### Viewing BIN Files
        1. **Upload** your .bin file
        2. **Configure** dimensions and format
        3. **View** the decoded image
        4. **Analyze** file structure and statistics
        5. **Export** as PNG if needed
        """)
        
        st.subheader("💡 Tips & Best Practices")
        st.markdown("""
        - **High contrast images** work best on LED displays
        - **RGB565** is most common for LED matrices (saves memory)
        - **Test with small images** first to verify format
        - **Square images** typically work best with most displays
        - **Use presets** for quick common configurations
        - **Dithering** helps with low-color displays
        """)
    
    with col_doc2:
        st.subheader("🎨 Color Format Guide")
        
        format_data = {
            "Format": ["RGB565", "RGB888/RGB24", "BGR888", "Grayscale", "Monochrome"],
            "Bits/Pixel": ["16", "24", "24", "8", "1"],
            "Colors": ["65K", "16.7M", "16.7M", "256", "2"],
            "Use Case": [
                "LED matrices, most common",
                "High-quality displays",
                "Some LED panels (reversed RGB)",
                "Monochrome displays, OLEDs",
                "Simple black/white displays"
            ]
        }
        
        st.table(format_data)
        
        st.subheader("📐 Common Display Sizes")
        st.markdown("""
        **LED Matrix Panels:**
        - 32×32 pixels (P3, P4, P5 modules)
        - 64×32 pixels (popular for signs)
        - 64×64 pixels (larger displays)
        - 128×64 pixels (wide panels)
        
        **OLED Displays:**
        - 128×64 pixels (common size)
        - 128×32 pixels (compact)
        - 256×64 pixels (wide)
        
        **LED Fans:**
        - Various custom sizes
        - Usually 64×32 or similar
        """)
    
    st.markdown("---")
    
    st.subheader("🔧 Technical Details")
    
    col_tech1, col_tech2 = st.columns(2)
    
    with col_tech1:
        st.markdown("""
        ### RGB565 Format
        16-bit color encoding:
        - **Bits 15-11:** Red (5 bits, 32 levels)
        - **Bits 10-5:** Green (6 bits, 64 levels)
        - **Bits 4-0:** Blue (5 bits, 32 levels)
        
        Green gets extra bit (human eye more sensitive).
        
        ### RGB888 Format
        24-bit full color:
        - **Byte 1:** Red (8 bits, 256 levels)
        - **Byte 2:** Green (8 bits, 256 levels)
        - **Byte 3:** Blue (8 bits, 256 levels)
        
        Total: 16,777,216 colors
        """)
    
    with col_tech2:
        st.markdown("""
        ### Byte Order
        - **RGB565:** Big-endian (MSB first)
        - **RGB888:** Sequential R-G-B
        - **BGR888:** Sequential B-G-R
        
        ### File Structure
        Raw binary data with no headers:
        ```
        [Pixel 0][Pixel 1][Pixel 2]...
        [Row 0 continues...]
        [Row 1][Row 2]...
        ```
        
        Sequential pixel data, row by row.
        """)
    
    st.markdown("---")
    
    st.subheader("❓ Troubleshooting")
    
    with st.expander("Image looks corrupted or wrong colors"):
        st.markdown("""
        **Possible solutions:**
        - Try different color format (RGB vs BGR)
        - Verify width/height match your display
        - Check if file size matches expected size
        - Some displays use different byte ordering
        """)
    
    with st.expander("File size doesn't match expected"):
        st.markdown("""
        **Check:**
        - Width × Height × BytesPerPixel should equal file size
        - RGB565: 2 bytes per pixel
        - RGB888: 3 bytes per pixel
        - Grayscale: 1 byte per pixel
        - File might have header or padding (not supported)
        """)
    
    with st.expander("Display shows garbled/shifted image"):
        st.markdown("""
        **Solutions:**
        - Incorrect dimensions set
        - Try auto-detect feature in BIN Viewer
        - Some displays use row/column scanning patterns
        - Check if display expects data in different order
        """)
    
    with st.expander("Colors are too dim/bright"):
        st.markdown("""
        **Adjust:**
        - Use Brightness/Contrast sliders in converter
        - Some displays have gamma correction
        - LED displays may need hardware brightness adjustment
        - Consider display's power limitations
        """)
    
    st.markdown("---")
    
    st.subheader("🚀 Example Use Cases")
    
    col_ex1, col_ex2 = st.columns(2)
    
    with col_ex1:
        st.markdown("""
        ### Arduino LED Matrix
        ```cpp
        // Load BIN file from SD card
        File binFile = SD.open("image.bin");
        uint8_t buffer[2048];
        binFile.read(buffer, 2048);
        
        // Display on matrix
        matrix.drawRGBBitmap(0, 0, 
          (uint16_t*)buffer, 64, 32);
        ```
        
        ### ESP32 LED Panel
        ```cpp
        #include <SD.h>
        #include <PxMatrix.h>
        
        // Read BIN file
        File f = SD.open("/image.bin");
        uint16_t pixel;
        for(int y=0; y<32; y++) {
          for(int x=0; x<64; x++) {
            f.read((uint8_t*)&pixel, 2);
            display.drawPixel(x, y, pixel);
          }
        }
        ```
        """)
    
    with col_ex2:
        st.markdown("""
        ### Raspberry Pi Display
        ```python
        from PIL import Image
        import numpy as np
        
        # Read BIN file
        with open('image.bin', 'rb') as f:
            data = f.read()
        
        # Convert RGB565 to RGB888
        pixels = []
        for i in range(0, len(data), 2):
            rgb565 = (data[i] << 8) | data[i+1]
            r = ((rgb565 >> 11) & 0x1F) << 3
            g = ((rgb565 >> 5) & 0x3F) << 2
            b = (rgb565 & 0x1F) << 3
            pixels.extend([r, g, b])
        
        # Create image
        img = Image.frombytes('RGB', 
          (64, 32), bytes(pixels))
        ```
        
        ### Processing/P5.js
        ```javascript
        // Load binary file
        let data = loadBytes('image.bin');
        
        loadPixels();
        for(let i=0; i<data.bytes.length; i+=2){
          let rgb565 = (data.bytes[i]<<8) | 
                       data.bytes[i+1];
          let r = ((rgb565>>11)&0x1F)<<3;
          let g = ((rgb565>>5)&0x3F)<<2;
          let b = (rgb565&0x1F)<<3;
          // Set pixel...
        }
        ```
        """)
    
    st.markdown("---")
    
    st.subheader("📦 Supported File Formats")
    
    col_fmt1, col_fmt2, col_fmt3 = st.columns(3)
    
    with col_fmt1:
        st.markdown("""
        **Input Images:**
        - ✅ PNG
        - ✅ JPG/JPEG
        - ✅ BMP
        - ✅ GIF
        - ✅ WebP
        """)
    
    with col_fmt2:
        st.markdown("""
        **Output Formats:**
        - ✅ .bin (raw binary)
        - ✅ RGB565 (16-bit)
        - ✅ RGB888 (24-bit)
        - ✅ BGR888 (24-bit)
        - ✅ Grayscale (8-bit)
        - ✅ Monochrome (1-bit)
        """)
    
    with col_fmt3:
        st.markdown("""
        **Export Options:**
        - ✅ Download BIN
        - ✅ Export PNG
        - ✅ Hex preview
        - ✅ Statistics
        - ✅ Analysis tools
        """)
    
    st.markdown("---")
    
    st.subheader("⚡ Performance Tips")
    st.markdown("""
    - **Large files (>1MB):** May take a few seconds to process
    - **Very high resolution:** Consider downscaling first
    - **Batch conversion:** Process multiple images one at a time
    - **Memory:** Close other tabs if processing very large files
    - **Browser:** Chrome/Edge recommended for best performance
    """)
    
    st.markdown("---")
    
    st.info("""
    💡 **Pro Tip:** Save your settings by bookmarking the page after configuring your preferences. 
    The app will remember your matrix dimensions and format choices!
    """)
    
    st.success("""
    ✨ **Need Help?** This tool is designed to be intuitive, but if you're stuck:
    1. Try the Quick Presets for common configurations
    2. Use Auto-detect in BIN Viewer to find correct dimensions
    3. Check the Troubleshooting section above
    4. Start with small test images to verify your setup
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p><b>Image ⇄ BIN Converter Pro</b></p>
    <p>Perfect for LED matrices, OLED displays, LED fans, and embedded displays</p>
    <p style='font-size: 0.9em;'>Supports RGB565, RGB888, BGR888, Grayscale, and Monochrome formats</p>
</div>
""", unsafe_allow_html=True)
