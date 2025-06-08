import shap
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def explain_prediction(model, user_df, background_df):
    """
    Generating SHAP explanations using TreeExplainer, avoiding pyarrow usage.
    Fixed to handle expected_value array issues and shape mismatches.
    """
    try:
        print("=== SHAP EXPLANATION START ===")
        print(f"User data shape: {user_df.shape}")
        print(f"Background data shape: {background_df.shape}")
        
        # Ensure consistent column order and types
        common_columns = list(user_df.columns)
        user_clean = user_df[common_columns].copy()
        bg_clean = background_df[common_columns].copy()
        
        # Convert to numeric, handling any conversion issues
        for col in common_columns:
            user_clean[col] = pd.to_numeric(user_clean[col], errors='coerce')
            bg_clean[col] = pd.to_numeric(bg_clean[col], errors='coerce')
        
        # Fill any NaN values with column means
        user_clean = user_clean.fillna(bg_clean.mean())
        bg_clean = bg_clean.fillna(bg_clean.mean())
        
        print(f"Cleaned user data shape: {user_clean.shape}")
        print(f"Cleaned background data shape: {bg_clean.shape}")
        
        # Create explainer with limited background data to avoid memory issues
        max_background = min(100, len(bg_clean))
        background_sample = bg_clean.sample(n=max_background, random_state=42)
        
        print("Creating TreeExplainer...")
        # Force TreeExplainer to avoid automatic model type detection that might use PyArrow
        explainer = shap.TreeExplainer(
            model, 
            data=background_sample,
            feature_perturbation='tree_path_dependent'  # Avoid interventional mode
        )
        
        print("Computing SHAP values...")
        # Compute SHAP values for single prediction
        shap_values = explainer.shap_values(user_clean.iloc[[0]])  # Keep as DataFrame slice
        expected_value = explainer.expected_value
        
        st.subheader("🔍 Prediction Explanation")
        
        # Debug information
        print(f"SHAP values type: {type(shap_values)}")
        print(f"Expected value type: {type(expected_value)}")
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            print(f"SHAP values is list with {len(shap_values)} elements")
            if len(shap_values) == 2:  # Binary classification
                values = shap_values[1][0]  # Positive class, first sample
                base_value = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) else expected_value
            else:  # Multi-class or single output
                values = shap_values[0][0]  # First class, first sample
                base_value = expected_value[0] if isinstance(expected_value, (list, np.ndarray)) else expected_value
        else:
            print(f"SHAP values shape: {shap_values.shape}")
            values = shap_values[0] if len(shap_values.shape) > 1 else shap_values
            base_value = expected_value[0] if isinstance(expected_value, (list, np.ndarray)) else expected_value
        
        # Ensure arrays are 1D and same length
        values = np.array(values).flatten()
        feature_values = user_clean.iloc[0].values.flatten()
        feature_names = list(user_clean.columns)
        
        print(f"Final array lengths - values: {len(values)}, feature_values: {len(feature_values)}, feature_names: {len(feature_names)}")
        
        # Validate array lengths match
        min_length = min(len(values), len(feature_values), len(feature_names))
        if not (len(values) == len(feature_values) == len(feature_names)):
            print(f"Trimming arrays to common length: {min_length}")
            values = values[:min_length]
            feature_values = feature_values[:min_length]
            feature_names = feature_names[:min_length]
        
        # Convert base_value to scalar
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(base_value[0]) if len(base_value) > 0 else 0.0
        else:
            base_value = float(base_value)
        
        # Create waterfall plot
        success = create_waterfall_plot(base_value, values, feature_names, feature_values)
        
        if not success:
            create_manual_waterfall(base_value, values, feature_names, feature_values)
        
        # Feature contributions table
        st.subheader("📊 Feature Contributions")
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
        
        st.dataframe(shap_df.head(15), use_container_width=True)
        
        # Summary plots
        create_shap_summary_plot(shap_df, base_value)
        
        # Download button
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
        
        if "pyarrow" in error_msg.lower():
            st.error("🚫 PyArrow dependency detected in SHAP")
            st.info("Using fallback explanation method instead.")
        elif "length" in error_msg.lower() or "shape" in error_msg.lower():
            st.error("🚫 Array dimension mismatch in SHAP calculation")
            st.info("The model prediction was successful, but explanation requires compatible data shapes.")
        else:
            st.error(f"🚫 SHAP explanation failed: {error_msg}")
        
        return safe_shap_fallback(model, user_df)


def create_waterfall_plot(base_value, shap_values, feature_names, feature_values):
    """
    Create SHAP waterfall plot with fallback options
    """
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Try modern SHAP waterfall first
        try:
            explanation = shap.Explanation(
                values=shap_values,
                base_values=base_value,
                data=feature_values,
                feature_names=feature_names
            )
            shap.plots.waterfall(explanation, show=False)
            st.pyplot(fig)
            plt.close()
            return True
            
        except Exception as modern_error:
            print(f"Modern waterfall failed: {modern_error}")
            plt.close()
            
            # Try legacy waterfall
            try:
                fig, ax = plt.subplots(figsize=(12, 8))
                shap.plots._waterfall.waterfall_legacy(
                    base_value,
                    shap_values,
                    feature_values,
                    feature_names=feature_names,
                    show=False
                )
                st.pyplot(fig)
                plt.close()
                return True
                
            except Exception as legacy_error:
                print(f"Legacy waterfall failed: {legacy_error}")
                plt.close()
                return False
                
    except Exception as e:
        print(f"Waterfall plot creation failed: {e}")
        plt.close()
        return False


def create_manual_waterfall(base_value, shap_values, feature_names, feature_values):
    """
    Create manual waterfall-style plot when SHAP waterfall fails
    """
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get top features by absolute SHAP value
        indices = np.argsort(np.abs(shap_values))[::-1][:10]  # Top 10
        top_shap = shap_values[indices]
        top_names = [feature_names[i] for i in indices]
        top_values = [feature_values[i] for i in indices]
        
        # Create horizontal bar chart
        colors = ['red' if x > 0 else 'blue' for x in top_shap]
        bars = ax.barh(range(len(top_shap)), top_shap, color=colors, alpha=0.7)
        
        ax.set_yticks(range(len(top_shap)))
        ax.set_yticklabels([f"{name}\n(value: {val:.2f})" for name, val in zip(top_names, top_values)])
        ax.set_xlabel('SHAP Value (Impact on Prediction)')
        ax.set_title(f'Top Feature Impacts on Prediction\nBase Value: {base_value:.3f}')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, top_shap)):
            width = bar.get_width()
            ax.text(width + (0.01 * max(np.abs(top_shap)) if width > 0 else -0.01 * max(np.abs(top_shap))), 
                   bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}', 
                   ha='left' if width > 0 else 'right', 
                   va='center')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
    except Exception as e:
        print(f"Manual waterfall creation failed: {e}")
        # Super simple fallback
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(range(min(10, len(shap_values))), shap_values[:10])
            ax.set_title("SHAP Values (Top 10 Features)")
            ax.set_xlabel("Feature Index")
            ax.set_ylabel("SHAP Value")
            if len(feature_names) >= 10:
                ax.set_xticks(range(10))
                ax.set_xticklabels(feature_names[:10], rotation=45)
            st.pyplot(fig)
            plt.close()
        except:
            st.error("Unable to create any visualization")


def create_shap_summary_plot(shap_df, base_value):
    """
    Create summary plots and statistics
    """
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            positive_contrib = shap_df[shap_df['SHAP_Value'] > 0]['SHAP_Value'].sum()
            negative_contrib = abs(shap_df[shap_df['SHAP_Value'] < 0]['SHAP_Value'].sum())
            
            categories = ['Positive\nContributions', 'Negative\nContributions']
            values = [positive_contrib, negative_contrib]
            colors = ['red', 'blue']
            
            bars = ax.bar(categories, values, color=colors, alpha=0.7)
            ax.set_title('Total Feature Contributions')
            ax.set_ylabel('Sum of SHAP Values')
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            top_features = shap_df.head(5)
            if len(top_features) > 0:
                ax.pie(top_features['Abs_SHAP'], labels=top_features['Feature'], 
                       autopct='%1.1f%%', startangle=90)
                ax.set_title('Top 5 Feature Contributions')
            else:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            
            st.pyplot(fig)
            plt.close()
            
        # Summary statistics
        st.subheader("📈 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Base Value", f"{base_value:.3f}")
        with col2:
            prediction_value = base_value + shap_df['SHAP_Value'].sum()
            st.metric("Final Prediction", f"{prediction_value:.3f}")
        with col3:
            positive_impact = shap_df[shap_df['SHAP_Value'] > 0]['SHAP_Value'].sum()
            st.metric("Total Positive Impact", f"{positive_impact:.3f}")
        with col4:
            negative_impact = shap_df[shap_df['SHAP_Value'] < 0]['SHAP_Value'].sum()
            st.metric("Total Negative Impact", f"{negative_impact:.3f}")
            
    except Exception as e:
        print(f"Summary plot creation failed: {e}")
        st.error("Could not create summary plots")


def safe_shap_fallback(model, user_df):
    """
    Enhanced fallback feature importance when SHAP completely fails
    """
    try:
        st.subheader("📈 Model Feature Importance (Fallback)")
        st.info("SHAP explanation unavailable. Showing model's built-in feature importance.")

        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': user_df.columns,
                'Importance': model.feature_importances_,
                'User_Value': user_df.iloc[0].values
            }).sort_values('Importance', ascending=False)

            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Top features bar chart
            top_features = importance_df.head(10)
            bars = ax1.barh(range(len(top_features)), top_features['Importance'], 
                           color='skyblue', alpha=0.8)
            ax1.set_yticks(range(len(top_features)))
            ax1.set_yticklabels(top_features['Feature'])
            ax1.set_xlabel('Feature Importance')
            ax1.set_title('Top 10 Most Important Features')
            ax1.invert_yaxis()
            
            # Add user values as text
            for i, (importance, user_val) in enumerate(zip(top_features['Importance'], 
                                                         top_features['User_Value'])):
                ax1.text(importance + max(top_features['Importance']) * 0.01, i,
                        f'Your value: {user_val:.2f}', va='center', fontsize=9)
            
            # Importance distribution histogram
            ax2.hist(importance_df['Importance'], bins=15, alpha=0.7, 
                    color='lightgreen', edgecolor='black')
            ax2.set_xlabel('Feature Importance')
            ax2.set_ylabel('Number of Features')
            ax2.set_title('Feature Importance Distribution')
            mean_importance = importance_df['Importance'].mean()
            ax2.axvline(x=mean_importance, color='red', linestyle='--', 
                       label=f'Mean: {mean_importance:.3f}')
            ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Display table
            display_df = importance_df.copy()
            display_df['Importance'] = display_df['Importance'].round(4)
            display_df['User_Value'] = display_df['User_Value'].round(3)
            st.dataframe(display_df, use_container_width=True)
            
            return True
            
        elif hasattr(model, 'coef_'):
            # Linear model coefficients
            st.info("Showing linear model coefficients as feature importance.")
            coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            
            coef_df = pd.DataFrame({
                'Feature': user_df.columns,
                'Coefficient': coef,
                'Abs_Coefficient': np.abs(coef),
                'User_Value': user_df.iloc[0].values
            }).sort_values('Abs_Coefficient', ascending=False)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            top_coef = coef_df.head(10)
            colors = ['red' if x > 0 else 'blue' for x in top_coef['Coefficient']]
            bars = ax.barh(range(len(top_coef)), top_coef['Coefficient'], 
                          color=colors, alpha=0.7)
            ax.set_yticks(range(len(top_coef)))
            ax.set_yticklabels(top_coef['Feature'])
            ax.set_xlabel('Model Coefficient')
            ax.set_title('Top 10 Model Coefficients')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
            st.dataframe(coef_df, use_container_width=True)
            return True
            
        else:
            st.warning("⚠️ Model does not provide feature importance or coefficients")
            st.info("This model type doesn't provide interpretability features.")
            return False

    except Exception as e:
        print(f"Fallback explanation error: {e}")
        st.error("❌ Feature explanation not available due to an internal error.")
        return False