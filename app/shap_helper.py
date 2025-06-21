import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import shap

def explain_prediction(model, user_df, background_df):
    """
    Generate model explanations using SHAP only.
    """
    
    try:
        print("ATTEMPTING SHAP EXPLANATION")
        print(f"User data shape: {user_df.shape}")
        print(f"Background data shape: {background_df.shape}")
        
        # Preparing clean data
        common_columns = list(user_df.columns)
        user_clean = user_df[common_columns].copy()
        bg_clean = background_df[common_columns].copy()
        
        # Converting to numeric
        for col in common_columns:
            user_clean[col] = pd.to_numeric(user_clean[col], errors='coerce')
            bg_clean[col] = pd.to_numeric(bg_clean[col], errors='coerce')
        
        # Filling the NaN values
        user_clean = user_clean.fillna(bg_clean.mean())
        bg_clean = bg_clean.fillna(bg_clean.mean())
        
        print("Creating SHAP explainer...")
        
        # Trying the simplest SHAP approach first
        explainer = None
        shap_values = None
        
        try:
            # Method 1: TreeExplainer without background (most compatible)
            print("Trying TreeExplainer without background...")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(user_clean.iloc[[0]])
            expected_value = explainer.expected_value
            print("TreeExplainer without background succeeded")
            
        except Exception as e1:
            print(f"TreeExplainer without background failed: {e1}")
            
            try:
                # Method 2: TreeExplainer with minimal background
                print("Trying TreeExplainer with small background...")
                small_bg = bg_clean.sample(n=min(50, len(bg_clean)), random_state=42)
                explainer = shap.TreeExplainer(model, data=small_bg, feature_perturbation='interventional')
                shap_values = explainer.shap_values(user_clean.iloc[[0]])
                expected_value = explainer.expected_value
                print("TreeExplainer with background succeeded")
                
            except Exception as e2:
                print(f"TreeExplainer with background failed: {e2}")
                
                try:
                    # Method 3: Try Explainer (auto-detect)
                    print("Trying auto-detection explainer...")
                    explainer = shap.Explainer(model)
                    shap_values = explainer(user_clean.iloc[[0]])
                    expected_value = explainer.expected_value if hasattr(explainer, 'expected_value') else 0
                    print("✓ Auto-detection explainer succeeded")
                    
                except Exception as e3:
                    print(f"Auto-detection explainer failed: {e3}")
                    raise Exception(f"All SHAP methods failed. Last error: {e3}")
        
        if shap_values is None:
            raise Exception("No SHAP values computed")
        
        # Processing SHAP results
        st.subheader("💡 Prediction Explanation")
        
        # Handling different SHAP output formats
        values, base_value = process_shap_values(shap_values, expected_value)
        
        # Ensuring array compatibility
        values = np.array(values).flatten()
        feature_values = user_clean.iloc[0].values.flatten()
        feature_names = list(user_clean.columns)
        
        # Syncing array lengths
        min_length = min(len(values), len(feature_values), len(feature_names))
        values = values[:min_length]
        feature_values = feature_values[:min_length]
        feature_names = feature_names[:min_length]
        
        print(f"Final arrays - values: {len(values)}, features: {len(feature_names)}")
        
        # Creating visualizations
        create_simple_shap_plot(base_value, values, feature_names, feature_values)
        
        st.markdown("""<hr style="margin: 2rem 0;">""", unsafe_allow_html=True)

        # Feature contributions table
        st.subheader("⚖️ Feature Contributions")
        shap_df = create_shap_dataframe(values, feature_names, feature_values)
        st.dataframe(shap_df.head(15), use_container_width=True)
        
        st.markdown("""<hr style="margin: 2rem 0;">""", unsafe_allow_html=True)
        
        # Summary statistics
        create_shap_summary_stats(shap_df, base_value)
        
        # Downloading option
        st.download_button(
            label="📥 Download SHAP Contributions CSV",
            data=shap_df.to_csv(index=False),
            file_name="shap_contributions.csv",
            mime="text/csv"
        )
        
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"SHAP Error: {error_msg}")
        st.error(f"🚫 SHAP explanation failed: {error_msg}")
        return False


def process_shap_values(shap_values, expected_value):
    """Extract values and base value from different SHAP formats"""
    
    if hasattr(shap_values, 'values'):
        values = shap_values.values[0]
        base_value = shap_values.base_values[0] if hasattr(shap_values, 'base_values') else expected_value
    elif isinstance(shap_values, list):
        if len(shap_values) == 2:
            values = shap_values[1][0]
            base_value = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) else expected_value
        else:
            values = shap_values[0][0]
            base_value = expected_value[0] if isinstance(expected_value, (list, np.ndarray)) else expected_value
    else:
        values = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        base_value = expected_value[0] if isinstance(expected_value, (list, np.ndarray)) else expected_value
    
    # Ensuring base_value is scalar
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(base_value[0]) if len(base_value) > 0 else 0.0
    else:
        base_value = float(base_value)
    
    return values, base_value


def create_simple_shap_plot(base_value, shap_values, feature_names, feature_values):
    """Creating a simple SHAP-like plot"""
    
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Getting top features by absolute impact
        indices = np.argsort(np.abs(shap_values))[::-1][:10]
        top_values = shap_values[indices]
        top_names = [feature_names[i] for i in indices]
        top_feature_vals = [feature_values[i] for i in indices]
        
        # Creating horizontal bar chart
        colors = ['red' if x > 0 else 'blue' for x in top_values]
        bars = ax.barh(range(len(top_values)), top_values, color=colors, alpha=0.7)
        
        # Customizing plot
        ax.set_yticks(range(len(top_values)))
        ax.set_yticklabels([f"{name}\n(value: {val:.2f})" for name, val in zip(top_names, top_feature_vals)])
        ax.set_xlabel('SHAP Value (Impact on Prediction)')
        ax.set_title(f'Top 10 Feature Impacts\nBase Prediction: {base_value:.3f}')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.invert_yaxis()
        
        # Adding value labels
        for i, (bar, value) in enumerate(zip(bars, top_values)):
            width = bar.get_width()
            offset = 0.01 * max(np.abs(top_values))
            x_pos = width + offset if width > 0 else width - offset
            ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}', 
                   ha='left' if width > 0 else 'right', 
                   va='center', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
    except Exception as e:
        print(f"Simple SHAP plot failed: {e}")
        st.error("Could not create SHAP visualization")


def create_shap_dataframe(values, feature_names, feature_values):
    """Creating SHAP contributions dataframe"""
    
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'Feature_Value': feature_values,
        'SHAP_Value': values,
        'Abs_SHAP': np.abs(values)
    }).sort_values('Abs_SHAP', ascending=False)
    
    # Round for display
    shap_df['Feature_Value'] = shap_df['Feature_Value'].round(3)
    shap_df['SHAP_Value'] = shap_df['SHAP_Value'].round(4)
    shap_df['Abs_SHAP'] = shap_df['Abs_SHAP'].round(4)
    
    return shap_df


def create_shap_summary_stats(shap_df, base_value):
    """Creating summary statistics display"""
    
    try:
        st.subheader("🔢 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Base Value", f"{base_value:.3f}")
        with col2:
            prediction_value = base_value + shap_df['SHAP_Value'].sum()
            st.metric("Final Prediction", f"{prediction_value:.3f}")
        with col3:
            positive_impact = shap_df[shap_df['SHAP_Value'] > 0]['SHAP_Value'].sum()
            st.metric("Positive Impact", f"+{positive_impact:.3f}")
        with col4:
            negative_impact = shap_df[shap_df['SHAP_Value'] < 0]['SHAP_Value'].sum()
            st.metric("Negative Impact", f"{negative_impact:.3f}")
            
    except Exception as e:
        print(f"Summary stats failed: {e}")