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
        # Making the original image semi-transparent
        img_data = img.getdata()
        new_data = []
        for item in img_data:
            if len(item) == 4: # RGBA
                new_data.append((item[0], item[1], item[2], int(item[3] * 0.5)))
            else: # RGB
                new_data.append((item[0], item[1], item[2], int(255 * 0.5)))
        img.putdata(new_data)
        # Creating a drawing context
        draw = ImageDraw.Draw(img)
        # Getting image dimensions
        img_width, img_height = img.size
        # Text to overlay
        full_text = "Customer Churn Prediction App"
        # Loading font with better sizing logic - prioritizing bold fonts
        try:
            # More conservative font size calculation
            font_size = min(img_width // 15, img_height // 8, 60) # Cap at 60px
            # Trying bold fonts first for better visibility
            bold_fonts_to_try = [
                "arialbd.ttf", "Arial-Bold.ttf", "calibrib.ttf", "Calibri-Bold.ttf",
                "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"
            ]

            # Regular fonts as fallback
            regular_fonts_to_try = [
                "arial.ttf", "Arial.ttf", "calibri.ttf", "Calibri.ttf",
                "DejaVuSans.ttf", "LiberationSans-Regular.ttf"
            ]
            main_font = None
            # Trying bold fonts first
            for font_name in bold_fonts_to_try:
                try:
                    main_font = ImageFont.truetype(font_name, font_size)
                    break
                except:
                    continue
            # If no bold font found, try regular fonts
            if not main_font:
                for font_name in regular_fonts_to_try:
                    try:
                        main_font = ImageFont.truetype(font_name, font_size)
                        break
                    except:
                        continue
            # Fallback to default font with size
            if not main_font:
                try:
                    main_font = ImageFont.load_default()
                except:
                    main_font = None
        except Exception as e:
            main_font = None
        # Calculating text positioning more accurately
        if main_font:
            # Using textbbox for accurate measurements
            bbox = draw.textbbox((0, 0), full_text, font=main_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Account for bbox offset (some fonts have negative offsets)
            text_offset_x = -bbox[0] # Adjust for any negative left offset
            text_offset_y = -bbox[1] # Adjust for any negative top offset
        else:
            # Fallback measurements
            text_width = len(full_text) * 12 # Rough estimate
            text_height = 20
            text_offset_x = 0
            text_offset_y = 0
        # Center the text with proper offsets
        text_x = (img_width - text_width) // 2 + text_offset_x
        text_y = (img_height - text_height) // 2 + text_offset_y
        # Ensuring text stays within image bounds
        text_x = max(10, min(text_x, img_width - text_width - 10))
        text_y = max(10, min(text_y, img_height - text_height - 10))
        # Adding enhanced shadow effect for bold appearance
        if main_font:
            # Multiple shadow layers for depth and boldness
            shadow_offsets = [(3, 3), (2, 2), (1, 1)]
            shadow_colors = [(0, 0, 0, 100), (0, 0, 0, 120), (0, 0, 0, 140)]
            # Drawing multiple shadow layers
            for (offset_x, offset_y), shadow_color in zip(shadow_offsets, shadow_colors):
                draw.text((text_x + offset_x, text_y + offset_y), full_text,
                        font=main_font, fill=shadow_color)
            # For extra boldness, draw the text multiple times with slight offsets
            bold_offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]
            for offset_x, offset_y in bold_offsets:
                draw.text((text_x + offset_x, text_y + offset_y), full_text,
                        font=main_font, fill=(0, 0, 0, 255))
        # Converting PIL image to bytes for Streamlit
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)

        # Displaying the image
        st.image(img_buffer, use_container_width=True)
    except Exception as e:
        st.warning("Image not found or could not be loaded. Please check the path.")
        st.info(f"Image load error: {str(e)}")
        # Fallback: display title without image
        st.title("🏠 Customer Churn Prediction App")

    # Description below the image
    st.markdown("""
    ### Welcome to the Customer Churn Prediction App!
    This comprehensive app helps you predict whether a customer will churn based on
    their information.
    **Features:**
    - 📊 **Dataset Viewer**: Explore the raw dataset used for training
    - 🧩 **Models**: View performance metrics of different machine learning models
    - 🔍 **Predictor**: Make predictions with SHAP explanations
    **How to use:**
    1. Start by exploring the **Dataset Viewer** to understand the data
    2. Check the **Models** section to see accuracy metrics
    3. Use the **Predictor** to make churn predictions
    Navigate using the sidebar menu to get started!
    """)