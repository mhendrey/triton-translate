# Translate
This is a BLS deployment for clients to sent text to be translated. This BLS is
composed of three subtasks, each of which is it's own deployment:

1. Language Identification
   Identify language of the source text sent if not provided by the client. Default
   model is the [fastText Language Identification model](./fasttext_language_identification.md).
   Other options that may be added in the future include [Lingua](https://github.com/pemistahl/lingua-py)
2. Sentence Segmenter
   Nearly all translation models were trained on sentence level text and thus input
   text needs to be broken up into sentence chunks. Default segmenter is the
   [sentencex](./sentencex.md). Other options that may be added in the future include
   [PySBD](https://github.com/nipunsadvilkar/pySBD).
3. Translation
   Currently using [SeamlessM4Tv2Large](./seamlessm4t_text2text.md) as the default
   translation model given its wide coverage of languages. In the next release,
   [NLLB](https://huggingface.co/facebook/nllb-200-distilled-600M) will be added given
   its faster speed with same or even slightly higher average performance across many
   languages.

General workflow organized by the BLS. If the `src_lang` is provided by the client,
then any language detection step is skipped and sentence segmentation is performed
followed by translation of each of the sentences. The translated results are bundled
together using a simple `" ".join(translated_sentences_array)` and then sent back.

If the `src_lang` is not provided by the client, the entire text provided is sent to
the language identification model to provide the necessary `src_lang` for the sentence
segmentation step. If the probability associated with the top result from the language
identifcation model is below `language_id_threshold`, then the language identification
is run again on each sentence after segmentation before translation occurs. The
translated results are bundled together as above and sent back to the client.

Because dynamic batching has been enabled for these Triton Inference Server
deployments, clients simply send each request separately. This simplifies the code for
the client, see examples below, yet they reap the benefits of batched processing. In
addition, this allows for controlling the GPU RAM consumed by the server.

## Optional Request Parameters
* `src_lang`: ISO 639-3 Language Code for submitted text. Default is `None` which
  triggers using language identification model.
* `tgt_lang`: ISO 639-3 Language Code for translated text. Default is `eng`
* `language_id_threshold`: Run language id for each sentence if document level language
  probability for top prediction is below this threshold. Default is 0.30.
* `translation_model`: Translation model to use. Default is `seamlessm4t`. Other
  option is `nllb`.

## Send Single Request
```
import requests

base_url = "http://localhost:8000/v2/models"

text = (
    """Dans les ruelles sombres de Neo-Paris, l'année 2077 étale son ombre numérique sur les derniers vestiges d'une humanité en déclin. La ville, désormais contrôlée par des corporations omnipotentes, brille de mille lumières artificielles, cachant la misère de ceux qui errent dans ses interstices numériques. Au cœur de ce chaos urbain, un hacker solitaire, connu sous le pseudonyme de Phoenix, se faufile à travers les réseaux informatiques, laissant sa marque dans le vaste univers virtuel qui enveloppe la réalité. Avec ses yeux augmentés par la cybernétique, il perçoit le monde tel un flux de données, dévoilant les secrets que les puissants cherchent à garder enfouis."""
)
inference_json = {
    "parameters": {"src_lang": "fra"}, # Optional src_lang provided
    "inputs": [
        {
            "name": "INPUT_TEXT",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [text],
        }
    ]
}
translated_response = requests.post(
    url=f"{base_url}/translate/infer",
    json=inference_json,
)

response_json = translated_response.json()
"""
{
    "model_name": "translate",
    "model_version": "1",
    "outputs": [
        {
            "name": "TRANSLATED_TEXT",
            "shape": [1],
            "datatype": "BYTES",
            "data": [
                'In the dark alleys of Neo-Paris, the year 2077 spreads its digital shadow over the last remnants of a declining humanity. The city, now controlled by omnipotent corporations, shines with a thousand artificial lights, hiding the misery of those who wander in its digital interstices. At the heart of this urban chaos, a lone hacker, known by the pseudonym Phoenix, sneaks through computer networks, leaving his mark in the vast virtual universe that envelops reality. With his cybernetically enhanced eyes, he perceives the world as a flow of data, revealing the secrets that the powerful seek to keep hidden.'
            ]
        }
    ]
}
"""
```

### Sending Many Requests
To submit multiple requests, use multithreading to send the requests in parallel to
take advantage of the dynamic batching on the server end to maximize throughput.

NOTE: You will encounter an "OSError: Too many open files" if you send a lot of
requests. Typically the default ulimit is 1024 on most system. Either increase this
using `ulimit -n {n_files}`, or don't create too many futures before you processing
some of them.

```
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

base_url = "http://localhost:8000/v2/models"

# First is in French, second is in Spanish
texts = [
    """Dans les ruelles sombres de Neo-Paris, l'année 2077 étale son ombre numérique sur les derniers vestiges d'une humanité en déclin. La ville, désormais contrôlée par des corporations omnipotentes, brille de mille lumières artificielles, cachant la misère de ceux qui errent dans ses interstices numériques. Au cœur de ce chaos urbain, un hacker solitaire, connu sous le pseudonyme de Phoenix, se faufile à travers les réseaux informatiques, laissant sa marque dans le vaste univers virtuel qui enveloppe la réalité. Avec ses yeux augmentés par la cybernétique, il perçoit le monde tel un flux de données, dévoilant les secrets que les puissants cherchent à garder enfouis.""",
    """Las luces de neón arrojaban arcoíris digitales a través de los callejones goteantes de Neo-París, una sinfonía caótica de luces y sombras donde el acero tosco se entrelazaba con hologramas relucientes. El viento, saturado de vapores químicos y sueños erróneos, silbaba entre los rascacielos, llevando consigo el murmullo de una ciudad donde los humanos se disolvían en la matriz, buscando un escape hacia los algoritmos y las sombras digitales. Fue en este océano de datos y desilusión donde yo, Kaï, un cazador de fallas a sueldo de un jefe enigmático, me lancé hacia una misión que sacudiría los cimientos mismos de nuestra fracturada realidad.""",
]

futures = {}
translated = {}
with ThreadPoolExecutor(max_workers=60) as executor:
    for i, text in enumerate(texts):
        infer_request = {
            "inputs": [
                {
                    "name": "INPUT_TEXT",
                    "shape": [1, 1],
                    "datatype": "BYTES",
                    "data": [text],
                }
            ]
        }
        future = executor.submit(requests.post,
            url=f"{base_url}/translate/infer",
            json=infer_request,
        )
        futures[future] = i
    
    for future in as_completed(futures):
        try:
            response = future.result()
        except Exception as exc:
            print(f"{futures[future]} threw {exc}")
        try:
            translated_text = response.json()["outputs"][0]["data"]
        except Exception as exc:
            raise ValueError(f"Error getting data from response: {exc}")
        translated[futures[future]] = translated_text
print(translated)
```

### Performance Analysis
There is some data in [data/translate](../data/translate/load_sample_one.json)
which can be used with the `perf_analyzer` CLI in the Triton Inference Server SDK
container to measure the throughput. This data contains a single spanish news
article with 21 sentences.

```
sdk-container:/workspace perf_analyzer \
    -m translate \
    -v \
    --input-data data/translate/load_sample_one.json \
    --measurement-mode=time_windows \
    --measurement-interval=20000 \
    --max-trials=4 \
    --concurrency-range=3 \
    --bls-composing=fasttext_language_identification,sentencex,seamlessm4t_text2text
```
Gives the following result on an RTX4090 GPU

* Request concurrency: 3
  * Pass [1] throughput: 1.4166 infer/sec. Avg latency: 2063479 usec (std 445241 usec). 
  * Pass [2] throughput: 1.41662 infer/sec. Avg latency: 2064923 usec (std 411890 usec). 
  * Pass [3] throughput: 1.37496 infer/sec. Avg latency: 2138226 usec (std 537251 usec). 
  * Client: 
    * Request count: 101
    * Throughput: 1.40272 infer/sec
    * Avg client overhead: 0.00%
    * Avg latency: 2088387 usec (standard deviation 174699 usec)
    * p50 latency: 1965212 usec
    * p90 latency: 2791654 usec
    * p95 latency: 2804253 usec
    * p99 latency: 3171326 usec
    * Avg HTTP time: 2088377 usec (send 52 usec + response wait 2088325 usec + receive 0 usec)
  * Server: 
    * Inference count: 101
    * Execution count: 100
    * Successful request count: 101
    * Avg request latency: 2088165 usec (overhead 223773 usec + queue 916062 usec + compute 948330 usec)

  * Composing models: 
  * fasttext_language_identification, version: 1
      * Inference count: 104
      * Execution count: 100
      * Successful request count: 104
      * Avg request latency: 1644 usec (overhead 2 usec + queue 339 usec + compute input 13 usec + compute infer 1277 usec + compute output 11 usec)

  * seamlessm4t_text2text, version: 1
      * Inference count: 2149
      * Execution count: 81
      * Successful request count: 2149
      * Avg request latency: 1859923 usec (overhead 28 usec + queue 915307 usec + compute input 188 usec + compute infer 944142 usec + compute output 257 usec)

  * sentencex, version: 1
      * Inference count: 104
      * Execution count: 98
      * Successful request count: 104
      * Avg request latency: 2858 usec (overhead 3 usec + queue 416 usec + compute input 14 usec + compute infer 2413 usec + compute output 10 usec)

* Inferences/Second vs. Client Average Batch Latency
* Concurrency: 3, throughput: 1.40272 infer/sec, latency 2088387 usec

### Validation
We use the same [Flores dataset](https://huggingface.co/datasets/facebook/flores)
used to validate the
[SeamlessM4Tv2ForTextToText](seamlessm4t_text2text.md), but this time, we aggregate up
15 sentences for a given language at one time and submit these to the `translate`
deployment endpoint that is using SeamlessM4T under the hood. Of course, the
`translate` deployment is using the `sentencex` deployment to split the text up into
sentences again. However, the chF2++ metric uses the block of 15 sentences for
comparison. For each language, we perform translation first by providing the
`src_lang` as a request parameter. This causes `translate` to skip language detection.
We then repeat doing the translation, but without providing the `src_lang`. This
causes `translate` to use the language detection deployment before performing sentence
segmentation followed by translation. In addition, if the probability assigned to the
top predicted language is less than the `language_id_threhold` (0.30), then each
sentence in the segmenter is sent for language detection before being translated.

The validation is run over a total of 96 languages. The results for each language are
listed below.

| SeamlessM4T Language | SeamlessM4T chrF2++ w/ src_lang | SeamlessM4T chrF2++ no src_lang | NLLB Language | NLLB chrF2++ w/ src_lang | NLLB chrF2++ no src_lang |
| :--: | :--: | :--: | :--: | :--: | :--: |
| afr | 67.7 | 67.7 | afr_Latn | 68.9 | 68.9 |
| amh | 64.0 | 64.0 | amh_Ethi | 59.6 | 59.6 |
| arb | 68.6 | 68.6 | arb_Arab | 66.3 | 66.3 |
| ary | 59.9 | 58.6 | ary_Arab | 57.8 | 56.0 |
| arz | 64.2 | 63.3 | arz_Arab | 62.0 | 60.9 |
| asm | 61.2 | 61.2 | asm_Beng | 58.8 | 58.8 |
| azj | 60.0 | 60.0 | azj_Latn | 58.7 | 58.7 |
| bel | 59.9 | 59.9 | bel_Cyrl | 57.8 | 57.8 |
| ben | 65.2 | 65.1 | ben_Beng | 63.2 | 63.2 |
| bos | 70.7 | 70.7 | bos_Latn | 68.3 | 68.1 |
| bul | 70.4 | 70.4 | bul_Cyrl | 68.1 | 68.1 |
| cat | 72.6 | 72.6 | cat_Latn | 71.2 | 71.2 |
| ceb | 69.6 | 69.6 | ceb_Latn | 67.4 | 67.4 |
| ces | 68.8 | 68.8 | ces_Latn | 66.7 | 66.7 |
| ckb | 61.5 | 61.5 | ckb_Arab | 60.5 | 60.5 |
| cmn | 62.4 | 61.8 | zho_Hans | 59.4 | 59.2 |
| cmn_Hant | 60.6 | 55.9 | zho_Hant | 56.0 | 56.9 |
| cym | 74.7 | 74.7 | cym_Latn | 71.4 | 71.4 |
| dan | 72.6 | 72.6 | dan_Latn | 71.4 | 71.4 |
| deu | 71.7 | 71.7 | deu_Latn | 69.6 | 69.6 |
| ell | 66.3 | 66.3 | ell_Grek | 64.8 | 64.8 |
| est | 65.6 | 65.6 | est_Latn | 63.7 | 63.7 |
| eus | 64.5 | 64.5 | eus_Latn | 62.1 | 62.1 |
| fin | 63.9 | 63.9 | fin_Latn | 62.1 | 62.1 |
| fra | 72.2 | 72.2 | fra_Latn | 70.1 | 70.0 |
| fuv | 41.9 | 41.9 | fuv_Latn | 43.2 | 43.2 |
| gaz | 56.0 | 56.0 | gaz_Latn | 52.9 | 52.9 |
| gle | 65.5 | 65.5 | gle_Latn | 63.5 | 63.5 |
| glg | 70.8 | 70.8 | glg_Latn | 69.1 | 69.1 |
| guj | 68.6 | 68.6 | guj_Gujr | 66.9 | 66.9 |
| heb | 68.8 | 68.7 | heb_Hebr | 66.6 | 66.6 |
| hin | 67.6 | 67.6 | hin_Deva | 67.0 | 67.0 |
| hrv | 67.4 | 67.4 | hrv_Latn | 65.0 | 65.0 |
| hun | 66.1 | 66.1 | hun_Latn | 63.7 | 63.7 |
| hye | 68.2 | 68.2 | hye_Armn | 65.1 | 65.1 |
| ibo | 60.4 | 60.3 | ibo_Latn | 56.5 | 56.5 |
| ind | 68.6 | 68.7 | ind_Latn | 68.2 | 68.2 |
| isl | 61.6 | 61.6 | isl_Latn | 58.5 | 58.5 |
| ita | 66.3 | 66.3 | ita_Latn | 65.5 | 65.5 |
| jav | 66.8 | 66.7 | jav_Latn | 65.3 | 65.3 |
| jpn | 54.1 | 54.1 | jpn_Jpan | 57.8 | 57.8 |
| kan | 64.7 | 64.8 | kan_Knda | 62.6 | 62.6 |
| kat | 62.3 | 62.3 | kat_Geor | 59.4 | 59.4 |
| kaz | 64.4 | 64.4 | kaz_Cyrl | 61.9 | 61.9 |
| khk | 60.3 | 60.4 | khk_Cyrl | 56.2 | 56.2 |
| khm | 9.9 | 9.9 | khm_Khmr | 24.6 | 24.4 |
| kir | 58.8 | 58.8 | kir_Cyrl | 56.0 | 56.0 |
| kor | 59.9 | 59.9 | kor_Hang | 59.0 | 59.0 |
| lao | 64.9 | 64.9 | lao_Laoo | 62.2 | 62.2 |
| lit | 63.5 | 63.5 | lit_Latn | 61.4 | 61.4 |
| lug | 52.7 | 52.7 | lug_Latn | 50.6 | 50.6 |
| luo | 55.6 | 55.6 | luo_Latn | 51.5 | 51.5 |
| lvs | 63.9 | 63.9 | lvs_Latn | 61.5 | 61.6 |
| mai | 69.7 | 69.7 | mai_Deva | 67.2 | 67.2 |
| mal | 65.7 | 65.7 | mal_Mlym | 64.2 | 64.1 |
| mar | 66.9 | 66.9 | mar_Deva | 63.8 | 63.9 |
| mkd | 70.9 | 70.9 | mkd_Cyrl | 68.5 | 68.6 |
| mlt | 75.4 | 75.4 | mlt_Latn | 74.3 | 74.3 |
| mni | 58.6 | 58.6 | mni_Beng | 56.4 | 56.4 |
| mya | 58.1 | 58.1 | mya_Mymr | 54.6 | 54.6 |
| nld | 64.3 | 64.3 | nld_Latn | 63.2 | 63.2 |
| nno | 70.9 | 70.9 | nno_Latn | 68.4 | 68.3 |
| nob | 70.5 | 70.5 | nob_Latn | 67.5 | 67.5 |
| npi | 68.3 | 68.3 | npi_Deva | 65.7 | 65.6 |
| nya | 58.4 | 58.4 | nya_Latn | 56.0 | 56.0 |
| ory | 66.7 | 66.7 | ory_Orya | 63.7 | 63.7 |
| pan | 56.6 | 56.5 | pan_Guru | 58.1 | 58.1 |
| pbt | 61.6 | 61.6 | pbt_Arab | 59.5 | 59.5 |
| pes | 66.7 | 66.8 | pes_Arab | 64.4 | 64.4 |
| pol | 63.1 | 63.1 | pol_Latn | 61.5 | 61.5 |
| por | 74.0 | 74.0 | por_Latn | 73.3 | 73.3 |
| ron | 70.7 | 70.7 | ron_Latn | 70.5 | 70.6 |
| rus | 66.6 | 66.6 | rus_Cyrl | 64.8 | 64.8 |
| sat | 40.9 | 41.0 | sat_Beng | 47.1 | - |
| slk | 68.5 | 68.5 | slk_Latn | 66.6 | 66.7 |
| slv | 65.2 | 65.2 | slv_Latn | 63.1 | 63.1 |
| sna | 58.2 | 58.2 | sna_Latn | 56.0 | 56.0 |
| snd | 65.1 | 65.1 | snd_Arab | 64.7 | 64.7 |
| som | 57.9 | 57.9 | som_Latn | 56.5 | 56.5 |
| spa | 64.8 | 64.8 | spa_Latn | 64.2 | 64.2 |
| srp | 70.9 | 70.9 | srp_Cyrl | 68.1 | 68.1 |
| swe | 72.6 | 72.6 | swe_Latn | 70.6 | 70.6 |
| swh | 66.5 | 66.5 | swh_Latn | 65.1 | 65.1 |
| tam | 62.9 | 62.9 | tam_Taml | 61.5 | 61.5 |
| tel | 67.0 | 67.0 | tel_Telu | 65.6 | 65.5 |
| tgk | 63.7 | 63.7 | tgk_Cyrl | 61.2 | 61.2 |
| tgl | 69.6 | 69.6 | tgl_Latn | 68.4 | 68.4 |
| tha | 15.4 | 15.6 | tha_Thai | 28.1 | 27.6 |
| tur | 66.8 | 66.8 | tur_Latn | 65.5 | 65.5 |
| ukr | 67.9 | 67.9 | ukr_Cyrl | 66.5 | 66.5 |
| urd | 63.9 | 63.9 | urd_Arab | 62.4 | 62.4 |
| uzn | 64.0 | 64.0 | uzn_Latn | 62.3 | 62.3 |
| vie | 64.5 | 64.5 | vie_Latn | 64.0 | 64.0 |
| yor | 51.0 | 51.0 | yor_Latn | 49.1 | 49.1 |
| yue | 57.6 | 57.6 | yue_Hant | 58.1 | 58.1 |
| zsm | - | - | zsm_Latn | 69.1 | 69.1 |
| zul | 66.5 | 66.4 | zul_Latn | 63.3 | 63.3 |
| **Mean** | **63.47** | **63.40** | | **62.07** | **62.19** |

Comparing against the [SeamlessM4T](./seamlessm4t_text2text.md) single sentence
translation, we find that we generally get slightly better results with an average
chrF2++ score of 63.5 compared to the sentence level comparision of 58.8.
The Seamless paper quotes an average of 59.2. For
[NLLB](.nllb_200_distilled_600M.md), we see an average chrF2++ score of 62.1
compared to the sentence level comparison of 55.7. It's worth noting that a few of
the results were significantly worse (tha and khm). These are a result of the
[sentencex](./sentencex.md) failing to split the text into sentences due to these
languages lacking any punctuation. As a result, though Seamless has a context window
large enough to process all the text it generates a stop token after the first
sentence or two causing the scores to crater.

In addition, the cmn_Hant also struggles a little bit. This is due to the language
detection struggling to identify the correct language for this particular language.

### Code
The code can be found in the [validate.py](../model_repository/translate/validate.py)
file.
