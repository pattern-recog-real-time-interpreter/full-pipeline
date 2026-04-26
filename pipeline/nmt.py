import time
from dataclasses import dataclass

from .config import PipelineConfig


@dataclass
class NMTResult:
    english_text: str
    latency_s: float


class NMTEngine:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._translator = None
        self._tokenizer = None

    def load(self) -> None:
        import ctranslate2
        from transformers import AutoTokenizer

        print(f"[NMT] Loading NLLB-600M INT8 from {self.config.nmt_model_dir} ...")
        self._translator = ctranslate2.Translator(
            self.config.nmt_model_dir,
            device=self.config.device,
            compute_type="int8",
        )
        print(f"[NMT] Loading tokenizer from {self.config.nmt_tokenizer_name} ...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.nmt_tokenizer_name)
        print("[NMT] NLLB-600M loaded.")

    def translate(self, thai_text: str) -> NMTResult:
        assert self._translator is not None, "Call load() first"

        if not thai_text.strip():
            return NMTResult(english_text="", latency_s=0.0)

        self._tokenizer.src_lang = "tha_Thai"
        tokens = self._tokenizer.convert_ids_to_tokens(
            self._tokenizer.encode(thai_text)
        )

        t0 = time.perf_counter()
        results = self._translator.translate_batch(
            [tokens],
            target_prefix=[["eng_Latn"]],
            beam_size=4,
            max_decoding_length=256,
        )
        latency_s = time.perf_counter() - t0

        # Skip the leading language token
        out_tokens = results[0].hypotheses[0][1:]
        out_ids = self._tokenizer.convert_tokens_to_ids(out_tokens)
        english_text = self._tokenizer.decode(out_ids, skip_special_tokens=True).strip()

        return NMTResult(english_text=english_text, latency_s=latency_s)

    @property
    def is_loaded(self) -> bool:
        return self._translator is not None
