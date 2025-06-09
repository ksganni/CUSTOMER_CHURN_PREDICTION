import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import SHAP in a try-catch to handle PyArrow issues
SHAP_AVAILABLE = False
try:
    import shap
    SHAP_AVAILABLE = True
    print("SHAP imported successfully")
except ImportError as e:
    print(f"SHAP import failed: {e}")
    SHAP_AVAILABLE = False

def explain_prediction(model, user_df, background_df):
    """
    Generating model explanations, avoiding pyarrow completely.
    Falls back gracefully when SHAP is unavailable.
    """
    
    # Skip SHAP entirely if not available or if PyArrow issues detected
    if not SHAP_AVAILABLE:
        st.info("🔄 SHAP not available, using built-in model explanations...")
        return safe_model_explanation(model, user_df)
    
    try:
        print("=== ATTEMPTING SHAP EXPLANATION ===")
        print(f"User data shape: {user_df.shape}")
        print(f"Background data shape: {background_df.shape}")
        
        # Prepare clean data
        common_columns = list(user_df.columns)
        user_clean = user_df[common_columns].copy()
        bg_clean = background_df[common_columns].copy()
        
        # Convert to numeric
        for col in common_columns:
            user_clean[col] = pd.to_numeric(user_clean[col], errors='coerce')
            bg_clean[col] = pd.to_numeric(bg_clean[col], errors='coerce')
        
        # Fill NaN values
        user_clean = user_clean.fillna(bg_clean.mean())
        bg_clean = bg_clean.fillna(bg_clean.mean())
        
        print("Creating SHAP explainer...")
        
        # Try the simplest SHAP approach first
        explainer = None
        shap_values = None
        
        try:
            # Method 1: TreeExplainer without background (most compatible)
            print("Trying TreeExplainer without background...")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(user_clean.iloc[[0]])
            expected_value = explainer.expected_value
            print("✓ TreeExplainer without background succeeded")
            
        except Exception as e1:
            print(f"TreeExplainer without background failed: {e1}")
            
            try:
                # Method 2: TreeExplainer with minimal background
                print("Trying TreeExplainer with small background...")
                small_bg = bg_clean.sample(n=min(50, len(bg_clean)), random_state=42)
                explainer = shap.TreeExplainer(model, data=small_bg, feature_perturbation='interventional')
                shap_values = explainer.shap_values(user_clean.iloc[[0]])
                expected_value = explainer.expected_value
                print("✓ TreeExplainer with background succeeded")
                
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
        
        # Process SHAP results
        st.subheader("🔍 Prediction Explanation")
        
        # Handle different SHAP output formats
        values, base_value = process_shap_values(shap_values, expected_value)
        
        # Ensure array compatibility
        values = np.array(values).flatten()
        feature_values = user_clean.iloc[0].values.flatten()
        feature_names = list(user_clean.columns)
        
        # Sync array lengths
        min_length = min(len(values), len(feature_values), len(feature_names))
        values = values[:min_length]
        feature_values = feature_values[:min_length]
        feature_names = feature_names[:min_length]
        
        print(f"Final arrays - values: {len(values)}, features: {len(feature_names)}")
        
        # Create visualizations
        create_simple_shap_plot(base_value, values, feature_names, feature_values)
        
        # Feature contributions table
        st.subheader("📊 Feature Contributions")
        shap_df = create_shap_dataframe(values, feature_names, feature_values)
        st.dataframe(shap_df.head(15), use_container_width=True)
        
        # Summary statistics
        create_shap_summary_stats(shap_df, base_value)
        
        # Download option
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
        
        # Check for PyArrow in any part of the error
        if "pyarrow" in error_msg.lower():
            st.error("🚫 PyArrow dependency detected in SHAP")
            st.info("Using built-in model explanation instead.")
        else:
            st.error(f"🚫 SHAP explanation failed: {error_msg}")
        
        return safe_model_explanation(model, user_df)


def process_shap_values(shap_values, expected_value):
    """Extract values and base value from different SHAP formats"""
    
    if hasattr(shap_values, 'values'):  # New SHAP Explanation object
        values = shap_values.values[0]
        base_value = shap_values.base_values[0] if hasattr(shap_values, 'base_values') else expected_value
    elif isinstance(shap_values, list):
        if len(shap_values) == 2:  # Binary classification
            values = shap_values[1][0]
            base_value = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) else expected_value
        else:
            values = shap_values[0][0]
            base_value = expected_value[0] if isinstance(expected_value, (list, np.ndarray)) else expected_value
    else:
        values = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        base_value = expected_value[0] if isinstance(expected_value, (list, np.ndarray)) else expected_value
    
    # Ensure base_value is scalar
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(base_value[0]) if len(base_value) > 0 else 0.0
    else:
        base_value = float(base_value)
    
    return values, base_value


def create_simple_shap_plot(base_value, shap_values, feature_names, feature_values):
    """Create a simple SHAP-like plot without using SHAP plotting functions"""
    
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get top features by absolute impact
        indices = np.argsort(np.abs(shap_values))[::-1][:10]
        top_values = shap_values[indices]
        top_names = [feature_names[i] for i in indices]
        top_feature_vals = [feature_values[i] for i in indices]
        
        # Create horizontal bar chart
        colors = ['red' if x > 0 else 'blue' for x in top_values]
        bars = ax.barh(range(len(top_values)), top_values, color=colors, alpha=0.7)
        
        # Customize plot
        ax.set_yticks(range(len(top_values)))
        ax.set_yticklabels([f"{name}\n(value: {val:.2f})" for name, val in zip(top_names, top_feature_vals)])
        ax.set_xlabel('SHAP Value (Impact on Prediction)')
        ax.set_title(f'Top 10 Feature Impacts\nBase Prediction: {base_value:.3f}')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.invert_yaxis()
        
        # Add value labels
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
    """Create SHAP contributions dataframe"""
    
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
    """Create summary statistics display"""
    
    try:
        st.subheader("📈 Summary Statistics")
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


def safe_model_explanation(model, user_df):
    """
    PyArrow-free model explanation using only built-in model attributes
    """
    
    try:
        st.subheader("📈 Model Feature Importance")
        st.info("Using model's built-in feature importance (PyArrow-free method)")

        # Method 1: Feature importances (tree-based models)
        if hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
            
            importance_values = model.feature_importances_
            feature_names = list(user_df.columns)
            user_values = user_df.iloc[0].values
            
            # Ensure arrays match
            min_len = min(len(importance_values), len(feature_names), len(user_values))
            importance_values = importance_values[:min_len]
            feature_names = feature_names[:min_len]
            user_values = user_values[:min_len]
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance_values,
                'Your_Value': user_values
            }).sort_values('Importance', ascending=False)

            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Bar chart of top features
            top_features = importance_df.head(10)
            bars = ax1.barh(range(len(top_features)), top_features['Importance'], 
                           color='skyblue', alpha=0.8)
            ax1.set_yticks(range(len(top_features)))
            ax1.set_yticklabels(top_features['Feature'])
            ax1.set_xlabel('Feature Importance')
            ax1.set_title('Top 10 Most Important Features')
            ax1.invert_yaxis()
            
            # Add user values
            for i, (importance, user_val) in enumerate(zip(top_features['Importance'], 
                                                         top_features['Your_Value'])):
                ax1.text(importance * 1.02, i, f'Your: {user_val:.2f}', 
                        va='center', fontsize=9)
            
            # Distribution histogram
            ax2.hist(importance_df['Importance'], bins=15, alpha=0.7, 
                    color='lightgreen', edgecolor='black')
            ax2.set_xlabel('Feature Importance')
            ax2.set_ylabel('Count')
            ax2.set_title('Importance Distribution')
            mean_imp = importance_df['Importance'].mean()
            ax2.axvline(x=mean_imp, color='red', linestyle='--', 
                       label=f'Mean: {mean_imp:.3f}')
            ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Display table
            display_df = importance_df.copy()
            display_df['Importance'] = display_df['Importance'].round(4)
            display_df['Your_Value'] = display_df['Your_Value'].round(3)
            st.dataframe(display_df, use_container_width=True)
            
            # Download button
            st.download_button(
                label="📥 Download Feature Importance",
                data=display_df.to_csv(index=False),
                file_name="feature_importance.csv",
                mime="text/csv"
            )
            
            return True

        # Method 2: Model coefficients (linear models)  
        elif hasattr(model, 'coef_') and model.coef_ is not None:
            
            st.info("Showing linear model coefficients")
            
            coefficients = model.coef_
            if hasattr(coefficients, 'shape') and len(coefficients.shape) > 1:
                coefficients = coefficients[0] if coefficients.shape[0] == 1 else coefficients.flatten()
            
            feature_names = list(user_df.columns)
            user_values = user_df.iloc[0].values
            
            # Match array lengths
            min_len = min(len(coefficients), len(feature_names), len(user_values))
            coefficients = coefficients[:min_len]
            feature_names = feature_names[:min_len]
            user_values = user_values[:min_len]
            
            coef_df = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefficients,
                'Abs_Coefficient': np.abs(coefficients),
                'Your_Value': user_values
            }).sort_values('Abs_Coefficient', ascending=False)
            
            # Visualization
            fig, ax = plt.subplots(figsize=(12, 8))
            top_coef = coef_df.head(10)
            colors = ['red' if x > 0 else 'blue' for x in top_coef['Coefficient']]
            bars = ax.barh(range(len(top_coef)), top_coef['Coefficient'], 
                          color=colors, alpha=0.7)
            
            ax.set_yticks(range(len(top_coef)))
            ax.set_yticklabels([f"{feat}\n(your: {val:.2f})" for feat, val in 
                               zip(top_coef['Feature'], top_coef['Your_Value'])])
            ax.set_xlabel('Model Coefficient')
            ax.set_title('Top 10 Model Coefficients\n(Red=Positive Impact, Blue=Negative Impact)')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.invert_yaxis()
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Display table
            display_df = coef_df.copy()
            display_df['Coefficient'] = display_df['Coefficient'].round(4)
            display_df['Abs_Coefficient'] = display_df['Abs_Coefficient'].round(4)
            display_df['Your_Value'] = display_df['Your_Value'].round(3)
            st.dataframe(display_df, use_container_width=True)
            
            # Download button
            st.download_button(
                label="📥 Download Coefficients",
                data=display_df.to_csv(index=False),
                file_name="model_coefficients.csv",
                mime="text/csv"
            )
            
            return True

        # Method 3: Basic input display
        else:
            st.warning("⚠️ Model doesn't provide interpretability features")
            st.subheader("📋 Your Input Values")
            
            input_df = pd.DataFrame({
                'Feature': user_df.columns,
                'Your_Value': user_df.iloc[0].values
            })
            input_df['Your_Value'] = input_df['Your_Value'].round(3)
            st.dataframe(input_df, use_container_width=True)
            
            # Simple bar chart of input values
            fig, ax = plt.subplots(figsize=(12, 6))
            top_inputs = input_df.head(15)
            ax.barh(range(len(top_inputs)), top_inputs['Your_Value'], 
                   color='lightcoral', alpha=0.7)
            ax.set_yticks(range(len(top_inputs)))
            ax.set_yticklabels(top_inputs['Feature'])
            ax.set_xlabel('Input Value')
            ax.set_title('Your Input Values')
            ax.invert_yaxis()
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            return True

    except Exception as e:
        print(f"Safe model explanation error: {e}")
        st.error(f"❌ Could not create any explanation: {str(e)}")
        
        # Absolute fallback - just show raw data
        try:
            st.subheader("📊 Raw Input Data")
            st.dataframe(user_df, use_container_width=True)
            return True
        except:
            st.error("❌ Complete failure - unable to display any information")
            return False