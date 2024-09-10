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
    # Notice that zho is not here. Need to rename that to cmn
    language_codes_seamless = [
        "afr",
        "amh",
        "arb",
        "ary",
        "arz",
        "asm",
        "azj",
        "bel",
        "ben",
        "bos",
        "bul",
        "cat",
        "ceb",
        "ces",
        "ckb",
        "cmn",
        "cmn_Hant",
        "cym",
        "dan",
        "deu",
        "ell",
        # "eng", # Skip English
        "est",
        "eus",
        "fin",
        "fra",
        "fuv",
        "gaz",
        "gle",
        "glg",
        "guj",
        "heb",
        "hin",
        "hrv",
        "hun",
        "hye",
        "ibo",
        "ind",
        "isl",
        "ita",
        "jav",
        "jpn",
        "kan",
        "kat",
        "kaz",
        "khk",
        "khm",
        "kir",
        "kor",
        "lao",
        "lit",
        "lug",
        "luo",
        "lvs",
        "mai",
        "mal",
        "mar",
        "mkd",
        "mlt",
        "mni",
        "mya",
        "nld",
        "nno",
        "nob",
        "npi",
        "nya",
        "ory",
        "pan",
        "pbt",
        "pes",
        "pol",
        "por",
        "ron",
        "rus",
        "sat",
        "slk",
        "slv",
        "sna",
        "snd",
        "som",
        "spa",
        "srp",
        "swe",
        "swh",
        "tam",
        "tel",
        "tgk",
        "tgl",
        "tha",
        "tur",
        "ukr",
        "urd",
        "uzn",
        "vie",
        "yor",
        "yue",
        "zsm",
        "zul",
    ]

    language_codes_nllb = [
        "afr_Latn",
        "amh_Ethi",
        "arb_Arab",
        "ary_Arab",
        "arz_Arab",
        "asm_Beng",
        "azj_Latn",
        "bel_Cyrl",
        "ben_Beng",
        "bos_Latn",
        "bul_Cyrl",
        "cat_Latn",
        "ceb_Latn",
        "ces_Latn",
        "ckb_Arab",
        "cym_Latn",
        "dan_Latn",
        "deu_Latn",
        "ell_Grek",
        # "eng_Latn", # Skip English
        "est_Latn",
        "eus_Latn",
        "fin_Latn",
        "fra_Latn",
        "fuv_Latn",
        "gaz_Latn",
        "gle_Latn",
        "glg_Latn",
        "guj_Gujr",
        "heb_Hebr",
        "hin_Deva",
        "hrv_Latn",
        "hun_Latn",
        "hye_Armn",
        "ibo_Latn",
        "ind_Latn",
        "isl_Latn",
        "ita_Latn",
        "jav_Latn",
        "jpn_Jpan",
        "kan_Knda",
        "kat_Geor",
        "kaz_Cyrl",
        "khk_Cyrl",
        "khm_Khmr",
        "kir_Cyrl",
        "kor_Hang",
        "lao_Laoo",
        "lit_Latn",
        "lug_Latn",
        "luo_Latn",
        "lvs_Latn",
        "mai_Deva",
        "mal_Mlym",
        "mar_Deva",
        "mkd_Cyrl",
        "mlt_Latn",
        "mni_Beng",
        "mya_Mymr",
        "nld_Latn",
        "nno_Latn",
        "nob_Latn",
        "npi_Deva",
        "nya_Latn",
        "ory_Orya",
        "pan_Guru",
        "pbt_Arab",
        "pes_Arab",
        "pol_Latn",
        "por_Latn",
        "ron_Latn",
        "rus_Cyrl",
        # "sat_Beng", # Flores has sat_Olck, but not sat_Beng
        "slk_Latn",
        "slv_Latn",
        "sna_Latn",
        "snd_Arab",
        "som_Latn",
        "spa_Latn",
        "srp_Cyrl",
        "swe_Latn",
        "swh_Latn",
        "tam_Taml",
        "tel_Telu",
        "tgk_Cyrl",
        "tgl_Latn",
        "tha_Thai",
        "tur_Latn",
        "ukr_Cyrl",
        "urd_Arab",
        "uzn_Latn",
        "vie_Latn",
        "yor_Latn",
        "yue_Hant",
        "zho_Hans",
        "zho_Hant",
        "zsm_Latn",
        "zul_Latn",
    ]

    errors = []
    chrf2 = []
    errors_no_src = []
    chrf2_no_src = []
    start_time = datetime.now()
    print(f"| Language | SeamlessM4T chrF2++ w/ src_lang | SeamlessM4T chrF2++ no src_lang |")
    print(f"| :------: | :-----------------------------: | :-----------------------------: |")
    for i, src in enumerate(language_codes_seamless):
        print(f"| {src} |", end="", flush=True)
        try:
            triton_score, errors_dict = test_pair(src, "eng", translation_model="seamlessm4t_text2text")
        except Exception as exc:
            errors.append(f"{src} threw {exc}\n")
        else:
            print(f" {triton_score:.1f} |", end="", flush=True)
            chrf2.append(triton_score)
            errors.append(errors_dict)

        try:
            triton_score, errors_dict = test_pair(src, "eng", use_src=False, translation_model="seamlessm4t_text2text")
        except Exception as exc:
            errors_no_src.append(f"{src} threw {exc}\n")
        else:
            print(f" {triton_score:.1f} |", flush=True)
            chrf2_no_src.append(triton_score)
            errors_no_src.append(errors_dict)

    end_time = datetime.now()

    mean_score = sum(chrf2) / len(chrf2)
    mean_no_score = sum(chrf2_no_src) / len(chrf2_no_src)
    print(f"| **Mean** | **{mean_score:.2f}** | **{mean_no_score:.2f}** |")
    print(f"\nTime taken: {end_time - start_time}")

    print(f"Errors when using src_lang in SeamlessM4T")
    for errors_dict in errors:
        if errors_dict:
            pprint(errors_dict)
    print("\n\nErrors when no src_lang in SeamlessM4T")
    for errors_dict in errors_no_src:
        if errors_dict:
            pprint(errors_dict)

    errors = []
    chrf2 = []
    errors_no_src = []
    chrf2_no_src = []
    start_time = datetime.now()
    print(f"| Language | NLLB chrF2++ w/ src_lang | NLLB chrF2++ no src_lang |")
    print(f"| :------: | :----------------------: | :----------------------: |")
    for i, src in enumerate(language_codes_nllb):
        print(f"| {src} |", end="", flush=True)
        try:
            triton_score, errors_dict = test_pair(src, "eng_Latn", translation_model="nllb_200_distilled_600M")
        except Exception as exc:
            errors.append(f"{src} threw {exc}\n")
        else:
            print(f" {triton_score:.1f} |", end="", flush=True)
            chrf2.append(triton_score)
            errors.append(errors_dict)

        try:
            triton_score, errors_dict = test_pair(src, "eng_Latn", use_src=False, translation_model="nllb_200_distilled_600M")
        except Exception as exc:
            errors_no_src.append(f"{src} threw {exc}\n")
        else:
            print(f" {triton_score:.1f} |", flush=True)
            chrf2_no_src.append(triton_score)
            errors_no_src.append(errors_dict)


    end_time = datetime.now()

    mean_score = sum(chrf2) / len(chrf2)
    mean_no_score = sum(chrf2_no_src) / len(chrf2_no_src)
    print(f"| **Mean** | **{mean_score:.2f}** | **{mean_no_score:.2f}** |")
    print(f"\nTime taken: {end_time - start_time}")

    print(f"Errors when using src_lang in NLLB")
    for errors_dict in errors:
        if errors_dict:
            pprint(errors_dict)
    print("\n\nErrors when no src_lang in NLLB")
    for errors_dict in errors_no_src:
        if errors_dict:
            pprint(errors_dict)



if __name__ == "__main__":
    main()
