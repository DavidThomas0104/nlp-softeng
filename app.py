import gradio as gr
from predict import predict_comment

def classify_text(comment):
    return {"Toxic": predict_comment(comment) == "Toxic",
            "Not Toxic": predict_comment(comment) == "Not Toxic"}

demo = gr.Interface(
    fn=predict_comment,
    inputs="text",
    outputs="text",
    title="Toxic Comment Classifier",
    description="Enter a comment and check if it is toxic or not."
)

if __name__ == "__main__":
    demo.launch()