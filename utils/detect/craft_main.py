from utils.detect.craft_pytorch.main import main


def craft_main(images, trained_model_path):
    return main(images, trained_model=trained_model_path)
