import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing

st.set_page_config(page_title="Bias Audit Report", layout="wide")
st.title("📊 AI Bias Audit Dashboard with Reweighing")

uploaded_file = st.file_uploader("📁 Upload your dataset (CSV format)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("🧾 Sample of Your Data")
    st.dataframe(df.head())

    st.markdown(f"**🔢 Rows:** {df.shape[0]} | **📌 Columns:** {df.shape[1]}")

    label_col = st.selectbox("🎯 Select the label column", df.columns, index=df.columns.get_loc("outcome") if "outcome" in df.columns else 0)
    protected_attr = st.selectbox("🛡️ Select protected attribute", df.columns)

    privileged_val = st.text_input(f"✔️ Privileged group value for '{protected_attr}'", "Male")
    unprivileged_val = st.text_input(f"❌ Unprivileged group value for '{protected_attr}'", "Female")

    df.dropna(inplace=True)
    df['protected_orig'] = df[protected_attr]

    # Map numeric protected attribute to string labels for consistent coloring
    def map_protected(val):
        if val == privileged_val:
            return "Male"
        elif val == unprivileged_val:
            return "Female"
        else:
            return str(val)

    df['protected_str'] = df['protected_orig'].apply(map_protected)

    favorable_rate = df[label_col].mean() * 100
    st.subheader("📋 Dataset Statistics Summary")
    st.markdown(f"""
    - **Favorable outcome rate:** `{favorable_rate:.2f}%` (`{label_col} = 1`)
    - Indicates class balance in your dataset.
    """)

    # Outcome Rate by Group
    st.subheader("📊 Outcome Rate by Group")
    plot_df = df[df['protected_str'].isin(['Male', 'Female'])]
    group_means = plot_df.groupby('protected_str')[label_col].mean().reset_index()

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    palette = {'Male': 'blue', 'Female': 'red'}
    sns.barplot(x='protected_str', y=label_col, data=group_means, palette=palette, ax=ax1)
    for i, row in group_means.iterrows():
        ax1.text(i, row[label_col] + 0.02, f"{row[label_col]*100:.1f}%", ha='center')
    ax1.set_ylim(0, 1)
    ax1.set_xlabel(protected_attr)
    ax1.set_ylabel(f"Mean {label_col}")
    ax1.set_title("Mean Outcome by Group")
    st.pyplot(fig1)

    # Outcome Distribution
    st.subheader("📊 Outcome Distribution")
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    sns.countplot(x=label_col, data=df, palette='pastel', ax=ax2)
    for p in ax2.patches:
        ax2.text(p.get_x() + p.get_width() / 2., p.get_height() + 3, int(p.get_height()), ha="center")
    ax2.set_title("Favorable vs. Unfavorable Outcomes")
    st.pyplot(fig2)

    if st.button("🔍 Run Bias Analysis"):
        try:
            # Map to binary for AIF360 dataset creation
            bin_map = {privileged_val: 1, unprivileged_val: 0}
            df[protected_attr] = df[protected_attr].map(bin_map)
            df_encoded = pd.get_dummies(df, drop_first=True)
            df_encoded[protected_attr] = df[protected_attr]
            df_encoded.dropna(inplace=True)

            dataset = BinaryLabelDataset(
                df=df_encoded,
                label_names=[label_col],
                protected_attribute_names=[protected_attr]
            )
            dataset_pred = dataset.copy()
            dataset_pred.labels = dataset.labels

            privileged_groups = [{protected_attr: 1}]
            unprivileged_groups = [{protected_attr: 0}]

            metric = ClassificationMetric(dataset, dataset_pred,
                                          unprivileged_groups=unprivileged_groups,
                                          privileged_groups=privileged_groups)

            st.subheader("📈 Fairness Metrics (Before Reweighing)")
            st.write(f"- Statistical Parity Difference: `{metric.statistical_parity_difference():.3f}`")
            st.write(f"- Disparate Impact: `{metric.disparate_impact():.3f}`")

            # Count per group with consistent labels & colors
            st.subheader("📊 Count of Individuals per Group")
            count_df = df[df['protected_str'].isin(['Male', 'Female'])]
            group_counts = count_df['protected_str'].value_counts().reset_index()
            group_counts.columns = ['Group', 'Count']

            fig3, ax3 = plt.subplots()
            sns.barplot(data=group_counts, x='Group', y='Count', palette=palette, ax=ax3)
            for i, row in group_counts.iterrows():
                ax3.text(i, row['Count'] + 5, row['Count'], ha='center')
            ax3.set_title("Number of Records per Group")
            st.pyplot(fig3)

            # Reweighing
            RW = Reweighing(unprivileged_groups=unprivileged_groups,
                            privileged_groups=privileged_groups)
            dataset_transf = RW.fit_transform(dataset)

            metric_rw = ClassificationMetric(dataset_transf, dataset_pred,
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)

            st.subheader("📉 Fairness Metrics (After Reweighing)")
            st.write(f"- Statistical Parity Difference: `{metric_rw.statistical_parity_difference():.3f}`")
            st.write(f"- Disparate Impact: `{metric_rw.disparate_impact():.3f}`")

            st.success(f"""
            ✅ Reweighing applied successfully.
            - Bias metrics improved: SPD changed from `{metric.statistical_parity_difference():.3f}` to `{metric_rw.statistical_parity_difference():.3f}`
            - Label distribution preserved (~{favorable_rate:.2f}% favorable)
            """)

            # Outcome Rate Comparison Before vs After Reweighing
            st.subheader("📉 Outcome Rate by Group: Before vs After Reweighing")
            original_rates = count_df.groupby('protected_str')[label_col].mean()
            reweighed_df = dataset_transf.convert_to_dataframe()[0]
            # Map numeric back to strings for plotting
            reweighed_df['protected_str'] = reweighed_df[protected_attr].map({1: 'Male', 0: 'Female'})
            reweighed_rates = reweighed_df.groupby('protected_str')[label_col].mean()

            compare_df = pd.DataFrame({
                'Before Reweighting': original_rates,
                'After Reweighting': reweighed_rates
            }).reset_index().melt(id_vars='protected_str', var_name='Stage', value_name='Outcome Rate')

            fig4, ax4 = plt.subplots()
            sns.barplot(data=compare_df, x='protected_str', y='Outcome Rate', hue='Stage', palette=palette, ax=ax4)
            for bar in ax4.patches:
                ax4.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                         f"{bar.get_height()*100:.1f}%", ha='center', fontsize=9)
            ax4.set_ylim(0, 1)
            ax4.set_title("Outcome Rate Before vs After Reweighing")
            st.pyplot(fig4)

            # Fairness Metric Comparison
            st.subheader("📊 Fairness Metric Comparison")
            fairness_data = pd.DataFrame({
                'Metric': ['Statistical Parity Difference', 'Disparate Impact'],
                'Before': [metric.statistical_parity_difference(), metric.disparate_impact()],
                'After': [metric_rw.statistical_parity_difference(), metric_rw.disparate_impact()]
            }).melt(id_vars='Metric', var_name='Stage', value_name='Value')

            fig5, ax5 = plt.subplots()
            sns.barplot(data=fairness_data, x='Metric', y='Value', hue='Stage', palette=['blue','red'], ax=ax5)
            for bar in ax5.patches:
                ax5.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                         f"{bar.get_height():.2f}", ha='center', fontsize=9)
            ax5.set_title("Fairness Metrics Before vs After Reweighing")
            st.pyplot(fig5)

        except Exception as e:
            st.error(f"⚠️ Error during analysis: {e}")
else:
    st.info("👈 Upload a CSV file to begin.")
