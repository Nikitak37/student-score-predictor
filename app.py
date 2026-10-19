
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Example dataset for learning purposes
data = {
    "Hours_Studied": [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 11
    ],
    "Exam_Score": [
        35, 42, 50, 61, 68, 78, 82, 88, 94, 98,
        39, 46, 55, 64, 73, 79, 86, 91, 96, 99
    ]
}

df = pd.DataFrame(data)

# Train a simple prediction model
X = df[["Hours_Studied"]]
y = df["Exam_Score"]

model = LinearRegression()
model.fit(X, y)

# App interface
st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Student Score Predictor")
st.write(
    "Explore how study hours relate to exam scores "
    "using a simple machine learning model."
)

st.info(
    "This demo uses illustrative sample data. "
    "Predictions are estimates, not guaranteed results."
)

hours = st.slider(
    "How many hours do you plan to study?",
    min_value=0.0,
    max_value=12.0,
    value=5.0,
    step=0.5
)

if st.button("Predict my score"):
    new_data = pd.DataFrame({
        "Hours_Studied": [hours]
    })

    prediction = model.predict(new_data)[0]
    prediction = max(0, min(100, prediction))

    st.subheader("Your estimated exam score")
    st.metric("Predicted score", f"{prediction:.1f} / 100")

    if prediction >= 80:
        st.success("Your estimated score is in the high range.")
    elif prediction >= 60:
        st.info("Keep practising to improve your estimated score.")
    else:
        st.warning(
            "Consider a study plan and more practice."
        )

st.subheader("Explore the sample dataset")

st.dataframe(df, use_container_width=True)

st.subheader("Study Hours vs Exam Scores")

fig, ax = plt.subplots(figsize=(8, 4))

ax.scatter(
    df["Hours_Studied"],
    df["Exam_Score"],
    label="Sample data"
)

ax.set_xlabel("Hours studied")
ax.set_ylabel("Exam score")
ax.set_title("Study Hours vs Exam Scores")
ax.grid(True)
ax.legend()

st.pyplot(fig)
plt.close(fig)

st.caption(
    "Educational demo only. Real exam results depend on "
    "many factors beyond study hours."
)
