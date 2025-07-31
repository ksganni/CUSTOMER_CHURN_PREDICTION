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

    # === Load model performance ===
    if model_scores is not None:
        st.success("✅ Displaying actual evaluation results from your training session.")
        model_names = list(model_scores.keys())
        roc_auc_means = [model_scores[name]['roc_auc_mean'] for name in model_names]
        roc_auc_stds = [model_scores[name]['roc_auc_std'] for name in model_names]
        accuracy_means = [model_scores[name]['accuracy_mean'] for name in model_names]
        accuracy_stds = [model_scores[name]['accuracy_std'] for name in model_names]

        model_performance = {
            'Model': model_names,
            'ROC-AUC Mean': roc_auc_means,
            'ROC-AUC Std': roc_auc_stds,
            'Accuracy Mean': accuracy_means,
            'Accuracy Std': accuracy_stds
        }
    else:
        st.warning("⚠️ Using example evaluation results. Run model training to see actual scores.")
        model_performance = {
            'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost', 'CatBoost'],
            'ROC-AUC Mean': [0.918, 0.798, 0.929, 0.929, 0.931],
            'ROC-AUC Std': [0.065, 0.066, 0.048, 0.062, 0.058],
            'Accuracy Mean': [0.824, 0.790, 0.848, 0.837, 0.836],
            'Accuracy Std': [0.079, 0.066, 0.067, 0.080, 0.078]
        }

    df = pd.DataFrame(model_performance).reset_index(drop=True)

    # === Sort and show table ===
    sort_metric = st.selectbox("Sort models by:", ["ROC-AUC Mean", "Accuracy Mean"])
    df_sorted = df.sort_values(sort_metric, ascending=False)

    # Highlight best model
    best_model_row = df_sorted.iloc[0]
    best_model = best_model_row['Model']
    best_score = best_model_row[sort_metric]

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: #a2bffe' if v else '' for v in is_max]

    st.subheader(f"📈 Models Ranked by {sort_metric}")
    styled_df = df_sorted.style.apply(highlight_max, subset=["ROC-AUC Mean", "Accuracy Mean"])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # === Best model summary ===
    st.success(f"🏆 Best Performing Model: **{best_model}** with {sort_metric} = {best_score:.3f}")

    st.subheader("📋 MODEL DETAILS")
    st.markdown(f"""
    **Selected Model:** {best_model}  
    - **ROC-AUC Score:** {best_model_row['ROC-AUC Mean']:.3f} ± {best_model_row['ROC-AUC Std']:.3f}  
    - **Accuracy Score:** {best_model_row['Accuracy Mean']:.3f} ± {best_model_row['Accuracy Std']:.3f}  
    - **Why this model?** It had the highest {sort_metric}, meaning it performed best based on that metric.  
    - **Cross-Validation:** 5-fold cross-validation was used.  
    - **Training Features Used:** {len(reference_columns) if reference_columns else 'N/A'} features  
    """)

    st.markdown(f"""
    **🔎 Models Performance Summary (ROC-AUC):**

    - **CatBoost:** {df[df['Model'] == 'CatBoost']['ROC-AUC Mean'].values[0]:.3f} ± {df[df['Model'] == 'CatBoost']['ROC-AUC Std'].values[0]:.3f}
    - **Random Forest:** {df[df['Model'] == 'Random Forest']['ROC-AUC Mean'].values[0]:.3f} ± {df[df['Model'] == 'Random Forest']['ROC-AUC Std'].values[0]:.3f}
    - **XGBoost:** {df[df['Model'] == 'XGBoost']['ROC-AUC Mean'].values[0]:.3f} ± {df[df['Model'] == 'XGBoost']['ROC-AUC Std'].values[0]:.3f}
    - **Logistic Regression:** {df[df['Model'] == 'Logistic Regression']['ROC-AUC Mean'].values[0]:.3f} ± {df[df['Model'] == 'Logistic Regression']['ROC-AUC Std'].values[0]:.3f}
    - **Decision Tree:** {df[df['Model'] == 'Decision Tree']['ROC-AUC Mean'].values[0]:.3f} ± {df[df['Model'] == 'Decision Tree']['ROC-AUC Std'].values[0]:.3f}

    ---
    **🧠 Metric Explanations:**

    - **ROC-AUC (Receiver Operating Characteristic – Area Under Curve):**
        - Measures how well the model distinguishes between classes (churn vs no churn).
        - Score ranges from **0.5 (random guessing)** to **1.0 (perfect separation)**.
        - The higher the better — especially useful for imbalanced datasets like churn.

    - **Accuracy:**
        - Measures the percentage of correct predictions overall.
        - Can be misleading if the dataset is imbalanced.
        - That’s why ROC-AUC is often more reliable in churn problems.
    """)

    st.markdown("""<hr style="margin: 2rem 0;">""", unsafe_allow_html=True)

    # === Visualization ===
    try:
        import matplotlib.pyplot as plt
        st.subheader("📊 MODEL PERFORMANCE VISUALIZATION")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        colors_roc = ['#FFFACD' if m == best_model else 'lightblue' for m in df['Model']]
        bars1 = ax1.bar(df['Model'], df['ROC-AUC Mean'], yerr=df['ROC-AUC Std'],
                        capsize=5, color=colors_roc, alpha=0.8)
        ax1.set_title('ROC-AUC Comparison')
        ax1.set_ylabel('ROC-AUC')
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_ylim(0.7, 1.0)
        ax1.grid(True, alpha=0.3)
        for bar, val in zip(bars1, df['ROC-AUC Mean']):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', fontsize=8)

        colors_acc = ['#FFFACD' if m == best_model else 'lightblue' for m in df['Model']]
        bars2 = ax2.bar(df['Model'], df['Accuracy Mean'], yerr=df['Accuracy Std'],
                        capsize=5, color=colors_acc, alpha=0.8)
        ax2.set_title('Accuracy Comparison')
        ax2.set_ylabel('Accuracy')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0.7, 1.0)
        ax2.grid(True, alpha=0.3)
        for bar, val in zip(bars2, df['Accuracy Mean']):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', fontsize=8)

        plt.tight_layout()
        st.pyplot(fig)

    except ImportError:
        st.info("📦 Install matplotlib to see visualizations: `pip install matplotlib`")
