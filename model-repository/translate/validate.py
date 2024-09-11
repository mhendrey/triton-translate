from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from datetime import datetime
import numpy as np
from pprint import pprint
import requests
from sacrebleu.metrics import CHRF


def get_translations(
    texts: list[str],
    src_langs: list[str],
    tgt_langs: list[str],
    translation_model: str = "seamlessm4t_text2text",
    language_id_threshold: float = None,
    max_workers: int = 50,
) -> list[str]:
    results = [None] * len(texts)
    errors = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, (text, src_lang, tgt_lang) in enumerate(
            zip(texts, src_langs, tgt_langs)
        ):
            inference_request = {
                "parameters": {"tgt_lang": tgt_lang, "translation_model": translation_model},
                "inputs": [
                    {
                        "name": "INPUT_TEXT",
                        "shape": [1, 1],
                        "datatype": "BYTES",
                        "data": [text],
                    },
                ],
            }
            if src_lang:
                inference_request["parameters"]["src_lang"] = src_lang
            if language_id_threshold:
                inference_request["parameters"][
                    "language_id_threshold"
                ] = language_id_threshold
            future = executor.submit(
                requests.post,
                url="http://localhost:8000/v2/models/translate/infer",
                json=inference_request,
            )
            futures[future] = i
        # Wait for results to come back
        for future in as_completed(futures):
            i = futures[future]
            try:
                response_json = future.result().json()
            except Exception as exc:
                raise ValueError(f"{texts[i]} threw {exc}")

            try:
                translated_text = response_json["outputs"][0]["data"][0]
            except:
                errors.append(f"{texts[i]} threw {response_json}")
                # raise ValueError(f"{texts[i]} threw {response_json}")
            else:
                results[i] = translated_text
    return results, errors


def test_pair(src, tgt, use_src: bool = True, translation_model: str = "seamlessm4t_text2text", language_id_threshold: float = None):
    flores = load_dataset("facebook/flores", "all", split="devtest")
    chrf = CHRF(word_order=2, eps_smoothing=True)
    if src == "cmn_Hant":
        flores_src = "zho_Hant"
    elif src == "cmn":
        flores_src = "zho_Hans"
    elif src == "sat_Beng":
        flores_src = "sat_Olck"
    else:
        flores_src = src
    src_sentence = [
        c for c in flores.column_names if c.startswith(f"sentence_{flores_src}")
    ]
    if src_sentence:
        src_sentence = src_sentence[0]
    else:
        raise ValueError(f"{src=:} {flores_src=:} not in flores")

    if tgt == "cmn_Hant":
        flores_tgt = "zho_Hant"
    elif tgt == "cmn":
        flores_tgt = "zho_Hans"
    elif tgt == "sat_Beng":
        flores_tgt = "sat_Olck"
    else:
        flores_tgt = tgt
    tgt_sentence = [
        c for c in flores.column_names if c.startswith(f"sentence_{flores_tgt}")
    ]
    if tgt_sentence:
        tgt_sentence = tgt_sentence[0]
    else:
        raise ValueError(f"{tgt=:} {flores_tgt=:} not in flores")

    tgt_texts = []
    translations = []
    errors = defaultdict(list)
    for batch in flores.iter(batch_size=60):
        texts = []
        src_langs = []
        tgt_langs = []
        for text_chunk in np.array_split(batch[src_sentence], 3):
            texts.append(" ".join(text_chunk))
            if use_src:
                src_langs.append(src)
            else:
                src_langs.append(None)
            tgt_langs.append(tgt)
        for text_chunk in np.array_split(batch[tgt_sentence], 3):
            tgt_texts.append(" ".join(text_chunk))
        results, errs = get_translations(
            texts, src_langs, tgt_langs, translation_model, language_id_threshold
        )
        if errs:
            errors[src] += errs
        for t in results:
            if t is not None:
                translations.append(t)

    return chrf.corpus_score(translations, [tgt_texts]).score, errors


def main():
    # These are the valid language codes in SeamlessM4Tv2Large
    # Given as (SeamlessM4T, NLLB)
    language_codes = [
        ("afr", "afr_Latn"),
        ("amh", "amh_Ethi"),
        ("arb", "arb_Arab"),
        ("ary", "ary_Arab"),
        ("arz", "arz_Arab"),
        ("asm", "asm_Beng"),
        ("azj", "azj_Latn"),
        ("bel", "bel_Cyrl"),
        ("ben", "ben_Beng"),
        ("bos", "bos_Latn"),
        ("bul", "bul_Cyrl"),
        ("cat", "cat_Latn"),
        ("ceb", "ceb_Latn"),
        ("ces", "ces_Latn"),
        ("ckb", "ckb_Arab"),
        ("cmn", "zho_Hans"),
        ("cmn_Hant", "zho_Hant"),
        ("cym", "cym_Latn"),
        ("dan", "dan_Latn"),
        ("deu", "deu_Latn"),
        ("ell", "ell_Grek"),
        # "eng", # Skip English
        ("est", "est_Latn"),
        ("eus", "eus_Latn"),
        ("fin", "fin_Latn"),
        ("fra", "fra_Latn"),
        ("fuv", "fuv_Latn"),
        ("gaz", "gaz_Latn"),
        ("gle", "gle_Latn"),
        ("glg", "glg_Latn"),
        ("guj", "guj_Gujr"),
        ("heb", "heb_Hebr"),
        ("hin", "hin_Deva"),
        ("hrv", "hrv_Latn"),
        ("hun", "hun_Latn"),
        ("hye", "hye_Armn"),
        ("ibo", "ibo_Latn"),
        ("ind", "ind_Latn"),
        ("isl", "isl_Latn"),
        ("ita", "ita_Latn"),
        ("jav", "jav_Latn"),
        ("jpn", "jpn_Jpan"),
        ("kan", "kan_Knda"),
        ("kat", "kat_Geor"),
        ("kaz", "kaz_Cyrl"),
        ("khk", "khk_Cyrl"),
        ("khm", "khm_Khmr"),
        ("kir", "kir_Cyrl"),
        ("kor", "kor_Hang"),
        ("lao", "lao_Laoo"),
        ("lit", "lit_Latn"),
        ("lug", "lug_Latn"),
        ("luo", "luo_Latn"),
        ("lvs", "lvs_Latn"),
        ("mai", "mai_Deva"),
        ("mal", "mal_Mlym"),
        ("mar", "mar_Deva"),
        ("mkd", "mkd_Cyrl"),
        ("mlt", "mlt_Latn"),
        ("mni", "mni_Beng"),
        ("mya", "mya_Mymr"),
        ("nld", "nld_Latn"),
        ("nno", "nno_Latn"),
        ("nob", "nob_Latn"),
        ("npi", "npi_Deva"),
        ("nya", "nya_Latn"),
        ("ory", "ory_Orya"),
        ("pan", "pan_Guru"),
        ("pbt", "pbt_Arab"),
        ("pes", "pes_Arab"),
        ("pol", "pol_Latn"),
        ("por", "por_Latn"),
        ("ron", "ron_Latn"),
        ("rus", "rus_Cyrl"),
        ("sat", "sat_Beng"), # Flores has sat_Olck, but not sat_Beng
        ("slk", "slk_Latn"),
        ("slv", "slv_Latn"),
        ("sna", "sna_Latn"),
        ("snd", "snd_Arab"),
        ("som", "som_Latn"),
        ("spa", "spa_Latn"),
        ("srp", "srp_Cyrl"),
        ("swe", "swe_Latn"),
        ("swh", "swh_Latn"),
        ("tam", "tam_Taml"),
        ("tel", "tel_Telu"),
        ("tgk", "tgk_Cyrl"),
        ("tgl", "tgl_Latn"),
        ("tha", "tha_Thai"),
        ("tur", "tur_Latn"),
        ("ukr", "ukr_Cyrl"),
        ("urd", "urd_Arab"),
        ("uzn", "uzn_Latn"),
        ("vie", "vie_Latn"),
        ("yor", "yor_Latn"),
        ("yue", "yue_Hant"),
        ("zsm", "zsm_Latn"),
        ("zul", "zul_Latn"),
    ]

    errors = {}
    chrf2 = []
    print("| SeamlessM4T Language ", end="")
    print("| SeamlessM4T chrF2++ w/ src_lang ", end="")
    print("| SeamlessM4T chrF2++ no src_lang ", end="")
    print("| NLLB Language ", end="")
    print("| NLLB chrF2++ w/ src_lang ", end="")
    print("| NLLB chrF2++ no src_lang |")
    print("| :--: | :--: | :--: | :--: | :--: | :--: |", flush=True)
    for language_code in language_codes:
        errors_lang = []
        chrf2_lang = []
        for i, src in enumerate(language_code):
            if i == 0: # SeamlessM4T
                tgt = "eng"
                translation_model = "seamlessm4t"
            else: # NLLB
                tgt = "eng_Latn"
                translation_model = "nllb"
            print(f"| {src} ", end="", flush=True)
            for use_src in [True, False]:
                try:
                    triton_score, errors_dict = test_pair(
                        src, tgt, use_src=use_src, translation_model=translation_model
                    )
                except Exception as exc:
                    errors_lang.append(f"{src} threw {exc}\n")
                    chrf2_lang.append(np.nan)
                    print(f"| - ", end="", flush=True)
                else:
                    print(f"| {triton_score:.1f} ", end="", flush=True)
                    chrf2_lang.append(triton_score)
                    errors_lang.append(errors_dict)
        print("|", flush=True)
        errors[language_code] = errors_lang
        chrf2.append(chrf2_lang)

    chrf2_mean = np.nanmean(np.array(chrf2), axis=0)
    print(f"| **Mean** | **{chrf2_mean[0]:.2f}** | **{chrf2_mean[1]:.2f}** ", end="")
    print(f"| | **{chrf2_mean[2]:.2f}** | **{chrf2_mean[3]:.2f}** |")


    for language_code, error_lang in errors.items():
        for error_dict in error_lang:
            if len(error_dict) > 0:
                print(f"{language_code=}")
                pprint(error_dict)



if __name__ == "__main__":
    main()
