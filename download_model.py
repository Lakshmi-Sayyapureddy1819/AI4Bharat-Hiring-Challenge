import nemo.collections.asr as nemo_asr
import os


def download_en_model():
    # Define a local path to save the model
    model_dir = os.path.join(os.getcwd(), "models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # This automatically fetches the model from NVIDIA's cloud and saves it locally
    model_name = "stt_en_conformer_ctc_medium"
    print(f"Downloading {model_name}...")
    model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name=model_name)

    # Save the model as a .nemo file for the asr_service.py to load
    save_path = os.path.join(model_dir, f"{model_name}.nemo")
    model.save_to(save_path)
    print(f"Model saved successfully at: {save_path}")


if __name__ == "__main__":
    download_en_model()
