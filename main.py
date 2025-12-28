from src.ui.gradio_app import build_demo

def main():
    demo = build_demo()
    demo.launch(share=True)

if __name__ == "__main__":
    main()