
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import zipfile

st.set_page_config(
    page_title="Auto Data Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Automatic Data Analysis & Visualization")
st.write(
    "Upload a CSV dataset to automatically explore data, "
    "calculate statistics, and generate charts."
)

uploaded_file = st.file_uploader(
    "Upload your CSV dataset",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if df.empty or len(df.columns) == 0:
            st.error("The uploaded CSV has no data.")
            st.stop()

        numeric_cols = df.select_dtypes(
            include="number"
        ).columns.tolist()

        categorical_cols = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        st.success("Dataset loaded successfully!")

        # Key metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing cells", int(df.isna().sum().sum()))
        c4.metric("Duplicate rows", int(df.duplicated().sum()))

        # Dataset preview
        st.header("1. Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)

        # Data types and missing values
        st.header("2. Data Quality")
        quality = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isna().sum().values,
            "Missing (%)": (
                df.isna().mean().values * 100
            ).round(2),
            "Unique Values": df.nunique().values
        })
        st.dataframe(quality, use_container_width=True)

        # Statistics
        st.header("3. Statistical Summary")
        if numeric_cols:
            st.dataframe(
                df[numeric_cols].describe().T,
                use_container_width=True
            )
        else:
            st.info("No numeric columns were detected.")

        # Automatic chart generation
        st.header("4. Automatic Visualizations")
        chart_files = {}

        for col in numeric_cols:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.histplot(
                data=df, x=col, kde=len(df[col].dropna()) > 1,
                ax=ax
            )
            ax.set_title(f"Distribution of {col}")
            fig.tight_layout()

            st.pyplot(fig)
            buffer = BytesIO()
            fig.savefig(buffer, format="png", dpi=120)
            chart_files[f"histogram_{col}.png"] = buffer.getvalue()
            plt.close(fig)

        if numeric_cols:
            # Correlation heatmap
            if len(numeric_cols) >= 2:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.heatmap(
                    df[numeric_cols].corr(),
                    annot=True,
                    cmap="coolwarm",
                    fmt=".2f",
                    ax=ax
                )
                ax.set_title("Correlation Heatmap")
                fig.tight_layout()
                st.pyplot(fig)

                buffer = BytesIO()
                fig.savefig(buffer, format="png", dpi=120)
                chart_files["correlation_heatmap.png"] = buffer.getvalue()
                plt.close(fig)

            # Scatter plot using the first two numeric columns
            if len(numeric_cols) >= 2:
                fig, ax = plt.subplots(figsize=(7, 4))
                sns.scatterplot(
                    data=df,
                    x=numeric_cols[0],
                    y=numeric_cols[1],
                    ax=ax
                )
                ax.set_title(
                    f"{numeric_cols[0]} vs {numeric_cols[1]}"
                )
                fig.tight_layout()
                st.pyplot(fig)

                buffer = BytesIO()
                fig.savefig(buffer, format="png", dpi=120)
                chart_files["scatter_plot.png"] = buffer.getvalue()
                plt.close(fig)

        # Categorical charts
        for col in categorical_cols:
            if df[col].nunique(dropna=True) <= 15:
                counts = df[col].fillna("Missing").astype(str).value_counts()

                fig, ax = plt.subplots(figsize=(7, 4))
                counts.plot(kind="bar", ax=ax)
                ax.set_title(f"Count by {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Count")
                ax.tick_params(axis="x", rotation=45)
                fig.tight_layout()
                st.pyplot(fig)

                buffer = BytesIO()
                fig.savefig(buffer, format="png", dpi=120)
                chart_files[f"category_{col}.png"] = buffer.getvalue()
                plt.close(fig)

        # Automatically generated observations
        st.header("5. Automatic Insights")
        st.write(f"The dataset contains {len(df)} rows and {len(df.columns)} columns.")

        if df.isna().sum().sum():
            st.warning(
                f"There are {int(df.isna().sum().sum())} missing cells. "
                "Consider how to handle them before further analysis."
            )
        else:
            st.info("No missing cells were detected.")

        if df.duplicated().sum():
            st.warning(
                f"{int(df.duplicated().sum())} duplicate rows were found."
            )
        else:
            st.info("No duplicate rows were detected.")

        if numeric_cols:
            for col in numeric_cols:
                st.write(
                    f"**{col}:** mean = {df[col].mean():.2f}, "
                    f"minimum = {df[col].min()}, maximum = {df[col].max()}."
                )

        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            pairs = corr.where(
                pd.DataFrame(
                    [[i != j for j in range(len(corr))]
                     for i in range(len(corr))],
                    index=corr.index,
                    columns=corr.columns
                )
            ).stack()

            if not pairs.empty:
                strongest = pairs.abs().idxmax()
                value = corr.loc[strongest[0], strongest[1]]
                st.write(
                    f"Strongest numeric correlation: **{strongest[0]}** "
                    f"and **{strongest[1]}** (r = {value:.2f}). "
                    "Correlation does not prove causation."
                )

        # Download report and charts together
        st.header("6. Download Results")
        report = [
            "AUTOMATIC DATA ANALYSIS REPORT",
            "",
            f"Rows: {len(df)}",
            f"Columns: {len(df.columns)}",
            f"Missing cells: {int(df.isna().sum().sum())}",
            f"Duplicate rows: {int(df.duplicated().sum())}",
            "",
            "STATISTICAL SUMMARY",
            df.describe().to_string() if numeric_cols
            else "No numeric columns found."
        ]

        st.download_button(
            "Download analysis report",
            data="\n".join(report),
            file_name="analysis_report.txt",
            mime="text/plain"
        )

        zip_buffer = BytesIO()
        with zipfile.ZipFile(
            zip_buffer, "w", zipfile.ZIP_DEFLATED
        ) as archive:
            for name, content in chart_files.items():
                archive.writestr(name, content)

        st.download_button(
            "Download all charts (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="analysis_charts.zip",
            mime="application/zip"
        )

    except Exception as e:
        st.error(f"Could not analyze this CSV: {e}")

else:
    st.info("Upload a CSV file above to begin automatic analysis.")
