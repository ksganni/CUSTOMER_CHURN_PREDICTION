# Models page

import streamlit as st
import pandas as pd

def show_page(model_scores, reference_columns):
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 2rem; border-radius: 10px; margin: 2rem 0;">
        <h3 style="color: #2c3e50;font-weight: 600;">
            🧩 Model Performance Overview
        </h3>
        <p style="color: #34495e; font-size: 1.1rem; line-height: 1.6;">
            Machine learning models were compared to evaluate their effectiveness in predicting customer churn. 
            The comparison is based on actual evaluation results from the training process.
        </p>        
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""<hr style="margin: 2rem 0;">""", unsafe_allow_html=True)
    
    # Checking if we have actual evaluation results
    if model_scores is not None:
        # Using actual evaluation results
        st.success("Displaying actual evaluation results from your training session")
        # Converting the loaded scores to DataFrame format
        model_names = list(model_scores.keys())
        roc_auc_means = [model_scores[name]['roc_auc_mean'] for name in model_names]
        roc_auc_stds = [model_scores[name]['roc_auc_std'] for name in model_names]
        model_performance = {
            'Model': model_names,
            'ROC-AUC Mean': roc_auc_means,
            'ROC-AUC Std': roc_auc_stds
        }
    else:
        # Using example values with a warning
        st.warning("Using example evaluation results. Run model training to see actual scores.")
        model_performance = {
            'Model': [
                'Logistic Regression',
                'Decision Tree',
                'Random Forest',
                'XGBoost',
                'CatBoost'
            ],
            'ROC-AUC Mean': [0.918437, 0.798340, 0.929505, 0.929416, 0.930626],
            'ROC-AUC Std': [0.065427, 0.065994, 0.047551, 0.061658, 0.058482]
        }
    
    performance_df = pd.DataFrame(model_performance)
    
    # Displaying performance table with pastel yellow highlighting
    st.subheader("📊 Model Comparison")
    # Creating a custom styling function for highlight
    def highlight(s):
        is_max = s == s.max()
        return ['background-color: #a2bffe' if v else '' for v in is_max]
    
    styled_df = performance_df.style.apply(highlight, subset=['ROC-AUC Mean'])
    st.dataframe(styled_df, use_container_width=True)
    
    # Best model highlights
    best_model_idx = performance_df['ROC-AUC Mean'].idxmax()
    best_model = performance_df.iloc[best_model_idx]['Model']
    best_score = performance_df.iloc[best_model_idx]['ROC-AUC Mean']
    st.success(f"🏆 **Best Performing Model:** {best_model} with {best_score:.3f} ROC-AUC score")

    st.markdown("""<hr style="margin: 2rem 0;">""", unsafe_allow_html=True)
    
    # Model details
    st.subheader("📋 MODEL DETAILS")
    st.markdown(f"""
    **Selected Model:** {best_model}
    - **ROC-AUC Score:** {best_score:.3f}
    - **Why this model?** {best_model} achieved the highest ROC-AUC score, indicating
    excellent performance in distinguishing between churning and non-churning customers.
    - **Cross-Validation:** All models were evaluated using 5-fold cross-validation.
    - **Training Features:** {len(reference_columns) if reference_columns else 'N/A'} features
    
    **Models Performance Summary:**
    - **CatBoost:** {performance_df.iloc[4]['ROC-AUC Mean']:.3f} ± {performance_df.iloc[4]['ROC-AUC Std']:.3f}
    - **Random Forest:** {performance_df.iloc[2]['ROC-AUC Mean']:.3f} ± {performance_df.iloc[2]['ROC-AUC Std']:.3f}
    - **XGBoost:** {performance_df.iloc[3]['ROC-AUC Mean']:.3f} ± {performance_df.iloc[3]['ROC-AUC Std']:.3f}
    - **Logistic Regression:** {performance_df.iloc[0]['ROC-AUC Mean']:.3f} ± {performance_df.iloc[0]['ROC-AUC Std']:.3f}
    - **Decision Tree:** {performance_df.iloc[1]['ROC-AUC Mean']:.3f} ± {performance_df.iloc[1]['ROC-AUC Std']:.3f}
    
    **Note:** ROC-AUC (Receiver Operating Characteristic - Area Under Curve) measures
    the model's ability to distinguish between classes. A score of 1.0 is perfect, while
    0.5 is random guessing.
    """)

    st.markdown("""<hr style="margin: 2rem 0;">""", unsafe_allow_html=True)
    
    # Performance visualization
    try:
        import matplotlib.pyplot as plt
        st.subheader("📈 MODEL PERFORMANCE VISUALIZATION")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # ROC-AUC comparison with error bars for best model
        colors = ['#FFFACD' if model == best_model else 'lightblue' for model in performance_df['Model']]
        ax1.bar(performance_df['Model'], performance_df['ROC-AUC Mean'],
               yerr=performance_df['ROC-AUC Std'], capsize=5, color=colors, alpha=0.8)
        ax1.set_title('Model ROC-AUC Comparison (with Standard Deviation)')
        ax1.set_ylabel('ROC-AUC Score')
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_ylim(0.7, 1.0)
        
        # ROC-AUC ranking for best model
        sorted_df = performance_df.sort_values('ROC-AUC Mean', ascending=True)
        colors_ranking = ['#FFFACD' if model == best_model else 'lightblue' for model in sorted_df['Model']]
        ax2.barh(sorted_df['Model'], sorted_df['ROC-AUC Mean'], color=colors_ranking, alpha=0.8)
        ax2.set_title('Model Ranking by ROC-AUC Score')
        ax2.set_xlabel('ROC-AUC Score')
        ax2.set_xlim(0.7, 1.0)
        
        plt.tight_layout()
        st.pyplot(fig)
    except ImportError:
        st.info("Install matplotlib to see performance visualizations: `pip install matplotlib`")