# Home page

import streamlit as st

def show_page():
    # Displaying image with text overlay
    try:
        from PIL import Image, ImageDraw, ImageFont
        import io

        # Loading the image
        img = Image.open("app/assets/churn.png")

        # Converting to RGBA if not already
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        # Making the original image slightly more opaque (25% opacity for better visibility)
        img_data = img.getdata()
        new_data = []
        for item in img_data:
            if len(item) == 4:  # RGBA
                new_data.append((item[0], item[1], item[2], int(item[3] * 0.25)))
            else:  # RGB
                new_data.append((item[0], item[1], item[2], int(255 * 0.25)))
        img.putdata(new_data)

        # Creating a drawing context
        draw = ImageDraw.Draw(img)

        # Getting image dimensions
        img_width, img_height = img.size

        # Text to overlay - professional formatting
        text_lines = [
            "WELCOME",
            "TO THE",
            "CUSTOMER CHURN",
            "PREDICTION"
        ]

        # Loading font with professional sizing
        try:
            # Professional font size calculation
            base_font_size = max(min(img_width // 12, img_height // 8), 50)
            font_size = base_font_size
            
            # Professional font selection - prioritizing clean, readable fonts
            font_paths = [
                # Windows professional fonts
                "C:/Windows/Fonts/calibrib.ttf",  # Calibri Bold
                "C:/Windows/Fonts/arialbd.ttf",  # Arial Bold
                "C:/Windows/Fonts/segoeui.ttf",  # Segoe UI
                "C:/Windows/Fonts/arial.ttf",
                # macOS professional fonts  
                "/System/Library/Fonts/Arial Bold.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
                "/System/Library/Fonts/Arial.ttf",
                # Linux professional fonts
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ]
            
            main_font = None
            for font_path in font_paths:
                try:
                    main_font = ImageFont.truetype(font_path, font_size)
                    break
                except:
                    continue
                    
            if not main_font:
                main_font = ImageFont.load_default()
                
        except Exception as e:
            main_font = ImageFont.load_default()

        # Calculate total text height and individual line widths
        line_heights = []
        line_widths = []
        
        for line in text_lines:
            bbox = draw.textbbox((0, 0), line, font=main_font)
            line_width = bbox[2] - bbox[0]
            line_height = bbox[3] - bbox[1]
            line_widths.append(line_width)
            line_heights.append(line_height)
        
        # Check if any line is too wide and reduce font size if needed
        max_line_width = max(line_widths)
        margin = img_width * 0.15  # 15% margin for professional spacing
        available_width = img_width - (2 * margin)
        
        if max_line_width > available_width:
            scale_factor = available_width / max_line_width
            font_size = int(font_size * scale_factor * 0.9)
            
            # Reload font with new size
            main_font = None
            for font_path in font_paths:
                try:
                    main_font = ImageFont.truetype(font_path, font_size)
                    break
                except:
                    continue
                    
            if not main_font:
                main_font = ImageFont.load_default()
            
            # Recalculate with new font size
            line_heights = []
            line_widths = []
            
            for line in text_lines:
                bbox = draw.textbbox((0, 0), line, font=main_font)
                line_width = bbox[2] - bbox[0]
                line_height = bbox[3] - bbox[1]
                line_widths.append(line_width)
                line_heights.append(line_height)
        
        # Professional spacing between lines
        line_spacing = max(line_heights) * 0.2
        total_text_height = sum(line_heights) + (len(text_lines) - 1) * line_spacing
        
        # Calculate starting Y position to center all text vertically
        start_y = (img_height - total_text_height) // 2
        
        # Draw each line centered horizontally with professional styling
        current_y = start_y
        
        for i, line in enumerate(text_lines):
            line_width = line_widths[i]
            line_height = line_heights[i]
            
            # Center this line horizontally
            text_x = (img_width - line_width) // 2
            text_y = current_y
            
            # Professional shadow effect
            shadow_offsets = [(3, 3), (2, 2)]
            for offset_x, offset_y in shadow_offsets:
                draw.text((text_x + offset_x, text_y + offset_y), line,
                          font=main_font, fill=(20, 20, 20, 180))  # Dark gray shadow
            
            # Subtle outline for clarity
            outline_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            for offset_x, offset_y in outline_offsets:
                draw.text((text_x + offset_x, text_y + offset_y), line,
                          font=main_font, fill=(40, 40, 40, 200))  # Subtle dark outline
            
            # Main text in professional navy blue
            draw.text((text_x, text_y), line,
                      font=main_font, fill=(25, 55, 109, 255))  # Professional navy blue
            
            # Move to next line position
            current_y += line_height + line_spacing

        # Converting PIL image to bytes for Streamlit
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)

        # Displaying the image
        st.image(img_buffer, use_container_width=True)

    except Exception as e:
        st.warning("Image not found or could not be loaded. Please check the path.")
        st.info(f"Image load error: {str(e)}")
        
        # Professional fallback header
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;">
            <h1 style="color: white; margin: 0; font-size: 2.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                🏢 Customer Churn Prediction Platform
            </h1>
        </div>
        """, unsafe_allow_html=True)

    # Professional description with better styling
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 2rem; border-radius: 10px; margin: 2rem 0;">
        <h3 style="color: #2c3e50; margin-top: 0;">About This Platform</h3>
        <p style="color: #34495e; font-size: 1.1rem; line-height: 1.6;">
            Customer churn prediction is the process of analyzing customer data 
            to anticipate which individuals are likely to discontinue using a company's 
            products or services. Accurately predicting churn enables businesses to take proactive 
            measures to retain at-risk customers, reducing customer loss and supporting long-term growth.
        </p>
        <p style="color: #34495e; font-size: 1.1rem; line-height: 1.6;">
            This application uses machine learning to predict the likelihood of a telecom customer churning 
            based on their personal profile, subscribed services, and billing patterns. It provides instant, 
            data-driven predictions along with clear explanations, helping businesses make informed and 
            timely decisions to improve customer retention.
        </p>
    </div>
    """, unsafe_allow_html=True)

    try:
        image = Image.open("app/assets/example.png")
        image = image.resize((1200, 600), resample=Image.LANCZOS)
        st.image(image, use_container_width=True)
    except Exception as e:
        st.info("Example image not found")

    # Professional key sections with card-like styling
    st.markdown("""
<h3 style="color: #2c3e50; margin-top: 0; border-bottom: 3px solid #3498db; padding-bottom: 0.5rem;">
    Key Platform Features
</h3>

<div style="background: linear-gradient(135deg, #e8f4fd 0%, #d1e7dd 100%); margin: 1.5rem 0; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #2980b9;">
    <h4 style="color: #2980b9; margin-bottom: 0.5rem;">📊 Dataset Viewer</h4>
    <p style="color: #34495e; margin-left: 1rem; line-height: 1.5;">
        Explore comprehensive customer information and understand churn trends and patterns 
        through interactive visualizations and statistical analysis.
    </p>
</div>

<div style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); margin: 1.5rem 0; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #27ae60;">
    <h4 style="color: #27ae60; margin-bottom: 0.5rem;">🤖 Models</h4>
    <p style="color: #34495e; margin-left: 1rem; line-height: 1.5;">
        Compare different machine learning models and review their performance metrics 
        to identify which algorithms provide the most accurate predictions for your data.
    </p>
</div>

<div style="background: linear-gradient(135deg, #e2d5f1 0%, #d6c7e8 100%); margin: 1.5rem 0; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #8e44ad;">
    <h4 style="color: #8e44ad; margin-bottom: 0.5rem;">🎯 Predictor</h4>
    <p style="color: #34495e; margin-left: 1rem; line-height: 1.5;">
        Enter customer details to receive instant churn predictions with detailed 
        explanations and actionable insights for customer retention strategies.
    </p>
</div>
""", unsafe_allow_html=True)
