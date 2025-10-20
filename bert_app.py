import gradio as gr
from bert_predict import predict_comment

demo = gr.Interface(
    fn=predict_comment,
    inputs="text",
    outputs="text",
    title="Toxic Comment Classifier (BERT)",
    description="Enter a comment to check if it is toxic or not."
)

if __name__ == "__main__":
    demo.launch()
