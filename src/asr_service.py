from fastapi import FastAPI, UploadFile, File
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Configuration: point `ASR_MODEL_PATH` at a NeMo .nemo file if available.
MODEL_PATH = os.getenv("ASR_MODEL_PATH", "")

# Lazy imports / model handle
_model = None
_model_type = None


def _load_nemo_model(path: str):
    """Attempt to load a NeMo EncDecCTCModel if nemo is installed and path exists.

    This is lazy and returns the model instance or raises an informative ImportError.
    """
    import nemo.collections.asr as nemo_asr
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path=path).to(device)
    model.freeze()
    return model


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), lang_id: str = "hi"):
    """Transcribes uploaded audio using a configured ASR backend.

    Behavior:
    - If `ASR_MODEL_PATH` points to a NeMo `.nemo` file and `nemo_toolkit` is installed,
      it will attempt to load and run that model (lazy-loaded on first request).
    - If no model is configured or loading fails, returns a helpful message so the
      caller can fall back to other approaches.
    """
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())

    global _model, _model_type
    if MODEL_PATH:
        try:
            if _model is None:
                # Try to load NeMo model (if available)
                _model = _load_nemo_model(MODEL_PATH)
                _model_type = "nemo"

            if _model_type == "nemo":
                # NeMo's transcribe expects list of file paths
                text = _model.transcribe([temp_path])[0]
                os.remove(temp_path)
                return {"transcription": text}

        except Exception as e:
            # Clean up and return an informative message rather than failing silently.
            try:
                os.remove(temp_path)
            except Exception:
                pass
            return {"error": "Failed to run ASR model", "detail": str(e)}

    # No model configured or couldn't run it
    try:
        os.remove(temp_path)
    except Exception:
        pass
    return {"transcription": "[ASR model not configured on server]", "note": "Set ASR_MODEL_PATH to a valid .nemo file and install nemo_toolkit[all] to enable transcription."}
