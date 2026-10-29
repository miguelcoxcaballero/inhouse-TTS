# Corpus de entrenamiento

Cada carpeta `corpora/<locale>/` describe los datos de una voz o un acento. Cada `*.json` es una especificación que `ihtts-data prepare` convierte en WAV de 24 kHz mono y 16 bits, con un `manifest.jsonl` (texto, fonemas en el acento del corpus, hablante, duración) y un `report.json` (horas, hablantes, energía por encima de 8 kHz, grabaciones rechazadas).

```bash
cargo build --profile train -p inhouse-tts-data
./target/train/ihtts-data prepare corpora/es-ES/*.json --cache data/cache --out data/corpora
```

El workflow «Training data» (`.github/workflows/data.yml`) prepara un corpus completo y lo publica como pre-release `data-<locale>`, que descarga el cuaderno de Colab (`train/colab.ipynb`).

## Reglas

- **Solo licencias permisivas**: dominio público, CC0, CC BY. CC BY-SA se marca en la especificación: obliga a compartir igual. Nada NC, ND ni GPL.
- **Banda completa**: se rechaza toda grabación de origen cuyo *rolloff* espectral del 99 % (mediana por tramas) quede por debajo de `min_rolloff_hz` (10 kHz). El audio de banda estrecha, de unos 6–7 kHz, es lo que suena «a llamada de teléfono». En LibriVox se usa siempre el MP3 original de 128 kbps y 44,1 kHz (*rolloff* ~14 kHz), nunca el derivado de 64 kbps y 22 kHz de archive.org (corte a ~10 kHz), que es el que distribuyen CML-TTS, MLS y CSS10.
- **Acento verificado**: cada hablante se asigna a un locale después de medir su acento. Una voz nunca aparece en dos idiomas ni en dos acentos.

## es-ES (castellano con distinción)

| Especificación | Hablante | Horas | Licencia | Verificación |
|---|---|---|---|---|
| `librivox-10246.json` | LibriVox 10246 (mujer) | hasta 30 de ~61 de banda completa | Dominio público (LibriVox; CC0 en archive.org). Índice: CML-TTS, CC BY 4.0 | distinción: R_E(θ/s) = −0,84; *rolloff* mediano 14,1 kHz |
| `librivox-3946.json` | LibriVox 3946 (hombre) | hasta 15 de ~28; solo sus grabaciones de 2013–2016 de banda completa | ídem | distinción: R_E = 0,01; sus grabaciones de 2010–2012 son de banda estrecha y quedan fuera |
| `ohf-dave.json` | *dave* (hombre) | 1,25 | CC0 (Nabu Casa / Open Home Foundation voice-datasets) | distinción: D = −0,09 ± 0,11; contenido hasta 20 kHz |

Los segmentos de LibriVox salen del índice de CML-TTS (Oliveira et al., «CML-TTS: A Multilingual Dataset for Speech Synthesis in Low-Resource Languages», 2023, CC BY 4.0): el fichero de segmentos de OpenSLR 146 (URL del capítulo, inicio, fin) y los textos del export parquet de Hugging Face (`freds0/cml_tts_dataset_spanish`). Solo se descarga la columna de texto, por rangos HTTP. Los tiempos coinciden muestra a muestra con el MP3 de 128 kbps: misma duración, desfase 0, correlación 0,9975. Para regenerar los índices:

```bash
ihtts-data cml-index --speakers 10246 --out corpora/es-ES/librivox-10246.tsv.gz
ihtts-data cml-index --speakers 3946 \
  --items historia_de_herodoto_3_1401_librivox,laestafetaromantica_1409_librivox,luchana_1312_librivox,maestrazgo_1404_librivox,naufragios_1308_librivox,onate_granja_librivox \
  --out corpora/es-ES/librivox-3946.tsv.gz
```

**Cómo se midió el acento.** Se usaron tramas de 20 ms y bandas de 250 Hz, con el espectro igualado por hablante. Una trama cuenta como fricativa sibilante cuando la banda de 3,8–11 kHz domina. Después se hace una regresión del número de tramas fricativas de cada frase sobre el número de letras de cada clase (s, z/ce/ci, ch, j…). Con distinción, la z no aporta energía sibilante (D ≈ 0); con seseo, suena como la s (D ≥ 1). El método se validó con hablantes conocidos de ambos tipos. Varios lectores etiquetados como «es_ES» en otros proyectos sesean y aquí **no** se usan como castellano: LibriVox 9972, 8882, 11797, 103 y 7433.

**Nota legal.** La Ley Orgánica 1/1982 (art. 7.6) protege el uso de la voz de una persona identificable con fines comerciales. Las voces publicadas no llevan el nombre de los lectores. Antes de ofrecer una voz que se parezca claramente a un lector concreto, conviene pedirle permiso (foro de LibriVox) o publicar voces mezcladas que no sean identificables.

## es-CO (prueba)

`google-female.json` y `google-male.json`: corpus colaborativo de Google para el español de Colombia (OpenSLR 72, CC BY-SA 4.0, 48 kHz, ~7,6 h). Por la cláusula *share-alike*, un modelo entrenado con estos datos debería publicarse también bajo CC BY-SA.
