import requests
import os
from dotenv import load_dotenv

load_dotenv()


def translate_to_english(text: str, source_lang: str = "hi-IN") -> str:
    """Translate `text` to English.

    Primary: Sarvam API if `SARVAM_API_KEY` is set.
    Fallback: OpenAI if `OPENAI_API_KEY` is set.
    Raises EnvironmentError if no translation backend is configured.
    """
    api_key = os.getenv("SARVAM_API_KEY")
    if api_key:
        url = "https://api.sarvam.ai/translate"
        headers = {"api-subscription-key": api_key, "Content-Type": "application/json"}
        payload = {
            "input": text,
            "source_language_code": source_lang,
            "target_language_code": "en-IN",
            "model": "sarvam-translate:v1",
        }
        resp = requests.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data.get("translated_text") or data.get("data") or ""

    # Fallback to OpenAI translate via a simple prompt if available
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            import openai

            openai.api_key = openai_key
            prompt = f"Translate the following text to English:\n\n{text}"
            resp = openai.Completion.create(model="text-davinci-003", prompt=prompt, max_tokens=1024)
            return resp.choices[0].text.strip()
        except Exception:
            return ""

    raise EnvironmentError("No translation backend configured. Set SARVAM_API_KEY or OPENAI_API_KEY in environment.")
