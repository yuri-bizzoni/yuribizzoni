# Machine Learning for the Humanities

**Corso di laurea magistrale · Università di Roma Tor Vergata · novembre 2026**

Nove notebook Jupyter, uno per lezione (3 ore ciascuna), per studentesse e studenti di area umanistica con poca o nessuna
esperienza di programmazione. Le spiegazioni sono in italiano, gli esempi usano testi italiani di pubblico dominio e tutto
gira su CPU in pochi secondi, anche senza connessione.

**Il filo rosso è la felicità.** Ogni lezione torna sulla stessa famiglia di parole (*felicità, gioia, allegria, contentezza,
letizia, beatitudine*) e sulla stessa domanda: che cosa riesce a coglierne una macchina, e che cosa le sfugge?
La letizia sale davvero verso il Paradiso? Il capitolo più "felice" di Pinocchio è quello delle orecchie d'asino.
Un rilevatore di frasi felici raggiunge il 98% di accuratezza senza trovarne nemmeno una. Per i vettori di parole,
*gioia* e *dolore* sono quasi sinonimi.

## Programma

| Data | Tema e domanda guida | Metodo | Laboratorio e risultato | Notebook | Soluzioni |
|---|---|---|---|---|---|
| 4 nov. | **Python da zero.** Che cosa sa fare un primo programma? | Regressione lineare e logistica | Primo programma funzionante in Colab: la felicità nei 100 canti della Commedia | [01](lessons/01_python_da_zero_regressione.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yuri-bizzoni/yuribizzoni/blob/main/ml-humanities/lessons/01_python_da_zero_regressione.ipynb) | [01](solutions/01_python_da_zero_regressione_soluzioni.ipynb) |
| 5 nov. | **I dati e le parole.** Come trasformare testi in numeri? | Albero di decisione | Liste, cicli, frequenze, tabelle e un CSV: la contentezza di Pinocchio | [02](lessons/02_dati_e_parole_alberi.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yuri-bizzoni/yuribizzoni/blob/main/ml-humanities/lessons/02_dati_e_parole_alberi.ipynb) | [02](solutions/02_dati_e_parole_alberi_soluzioni.ipynb) |
| 9 nov. | **Imparare dagli esempi.** Una macchina può scoprire da sola le regole? | SVM | Addestrare un classificatore testuale con scikit-learn (cinque autori) | [03](lessons/03_imparare_dagli_esempi_svm.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yuri-bizzoni/yuribizzoni/blob/main/ml-humanities/lessons/03_imparare_dagli_esempi_svm.ipynb) | [03](solutions/03_imparare_dagli_esempi_svm_soluzioni.ipynb) |
| 11 nov. | **Misurare il risultato.** Un'accuratezza del 95% è davvero buona? | Non supervisionato (PCA) | Baseline, matrice di confusione, precision e recall: il rilevatore di frasi felici | [04](lessons/04_misurare_il_risultato.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yuri-bizzoni/yuribizzoni/blob/main/ml-humanities/lessons/04_misurare_il_risultato.ipynb) | [04](solutions/04_misurare_il_risultato_soluzioni.ipynb) |
| 12 nov. | **L'AI come assistente.** Come possiamo programmare meglio e risparmiare tempo? | K-means | Una piccola automazione Python con l'aiuto di ChatGPT: prompt, test, correzione | [05](lessons/05_ai_assistente_kmeans.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yuri-bizzoni/yuribizzoni/blob/main/ml-humanities/lessons/05_ai_assistente_kmeans.ipynb) | [05](solutions/05_ai_assistente_kmeans_soluzioni.ipynb) |
| 16 nov. | **Significato e similarità.** Le macchine capiscono davvero? | Clustering gerarchico (e ancora k-means) | Un motore di ricerca per passi simili; il dendrogramma dei *Canti* di Leopardi | [06](lessons/06_significato_similarita_clustering.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yuri-bizzoni/yuribizzoni/blob/main/ml-humanities/lessons/06_significato_similarita_clustering.ipynb) | [06](solutions/06_significato_similarita_clustering_soluzioni.ipynb) |
| 18 nov. | **Creatività, ironia e bias.** Che cosa non comprendono i modelli? | Semantica distribuzionale | Rappresentare testi come vettori, confrontare similarità, discutere gli embeddings | [07](lessons/07_semantica_distribuzionale.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yuri-bizzoni/yuribizzoni/blob/main/ml-humanities/lessons/07_semantica_distribuzionale.ipynb) | [07](solutions/07_semantica_distribuzionale_soluzioni.ipynb) |
| 19 nov. | **Il nostro esperimento.** Possiamo rispondere a una domanda vera con i dati? | Modelli di lingua (n-grammi) | Laboratorio per i piccoli progetti di gruppo | [08](lessons/08_modelli_di_lingua_progetto.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yuri-bizzoni/yuribizzoni/blob/main/ml-humanities/lessons/08_modelli_di_lingua_progetto.ipynb) | [08](solutions/08_modelli_di_lingua_progetto_soluzioni.ipynb) |
| 20 nov. | **Dimostrare e discutere.** Che cosa sappiamo costruire, e che cosa sappiamo criticare? | Tutti i metodi a confronto | Presentazioni, riflessione collettiva e valutazione finale | [09](lessons/09_dimostrare_e_discutere.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yuri-bizzoni/yuribizzoni/blob/main/ml-humanities/lessons/09_dimostrare_e_discutere.ipynb) | [09](solutions/09_dimostrare_e_discutere_soluzioni.ipynb) |

Ogni notebook contiene: **obiettivi di apprendimento**, brevi spiegazioni, esempi da eseguire, un riepilogo e **2–3 esercizi**.
Le soluzioni commentate sono nella cartella `solutions/`.

## Giochi in classe

Due piccoli giochi da aprire con il telefono, inquadrando il QR code. Girano tutti nel browser: niente account, niente server, niente dati inviati.

| Lezione | Gioco | Che cosa si fa | Slide con il QR |
|---|---|---|---|
| 5 nov. · alberi | [Sei tu l'albero di decisione](https://yuri-bizzoni.github.io/yuribizzoni/ml-humanities/gioco/) | Collodi o Manzoni? Costruisci una regola con una soglia; il capitolo più felice di Pinocchio | `slides/Slide_QR_gioco.pptx` |
| 9 nov. · SVM | [Sei tu la SVM](https://yuri-bizzoni.github.io/yuribizzoni/ml-humanities/gioco-svm/) | Traccia il confine tra Inferno e Paradiso; il peso delle parole; una SVM vera che attribuisce le tue frasi felici | `slides/Slide_QR_gioco_svm.pptx` |

## Dentro un modello di lingua

[L'Inferno del Transformer](https://yuri-bizzoni.github.io/yuribizzoni/ml-humanities/imbuto/) è una visualizzazione in 3D
da girare ed esplorare con il mouse o con le dita. Un modello di lingua è disegnato come un imbuto di nove cerchi:
tokenizzazione, embedding, attenzione, teste, rete feed-forward, residui, strati, unembedding, softmax. Ogni cerchio ha la sua
matematica essenziale. La frase «La felicità è…» scende fino al fondo e risale una parola alla volta, fino a «riveder le
stelle»; il Purgatorio spiega l'addestramento. Si collega alle lezioni 1, 2, 6, 7 e 8. Gira offline, perché three.js è nella
cartella `imbuto/vendor/`. Il QR code è in `imbuto/qr_imbuto.png`.

## Come usare i notebook

**Su Google Colab (consigliato in aula).** Cliccate sul bottone *Open in Colab* della lezione. La prima cella del notebook
(*cella di preparazione*) scarica automaticamente i dati del corso da questo repository. Poi eseguite le celle in ordine con
**Shift + Invio**.

**Sul vostro computer, anche offline.**

```bash
git clone https://github.com/yuri-bizzoni/yuribizzoni
cd yuribizzoni/ml-humanities
pip install -r requirements.txt
jupyter notebook
```

La cella di preparazione trova da sola la cartella `data/`, quindi non serve internet.

## Struttura

```
ml-humanities/
├── lessons/        9 notebook, uno per lezione
├── solutions/      le soluzioni degli esercizi
├── data/           i testi e le tabelle del corso (vedi data/README.md)
└── requirements.txt
```

Gli esercizi che salvano file (CSV, rapporti) li scrivono in una cartella `risultati/` accanto al notebook.

## I dati

Otto opere italiane di pubblico dominio da Project Gutenberg: *Pinocchio* (Collodi), *I promessi sposi* (Manzoni),
la *Divina Commedia* (Dante), i *Canti* (Leopardi), *Dopo il divorzio* (Deledda), *Un cavallo nella luna* e *L'umorismo*
(Pirandello), *Novelle napolitane* (Di Giacomo). Ci sono anche alcune tabelle derivate: canti, capitoli e brani d'autore.
Infine un piccolo **lessico della felicità** in più lingue e le forme italiane delle sei famiglie di parole felici.
Fonti, licenze e procedura di preparazione sono descritte in [`data/README.md`](data/README.md).

## Note per chi insegna

- Ogni notebook gira dall'alto in basso in meno di 15 secondi su un portatile, senza GPU e senza scaricare modelli.
- I notebook sono salvati **con gli output**, così si possono leggere anche senza eseguirli. In aula conviene
  *Runtime → Esegui tutto* oppure *Kernel → Restart & Run All*.
- Diversi risultati sono **negativi o deludenti di proposito**, e sono discussi come tali: il 100% della lezione 3 dipende da
  un libro per autore (lo smonta la lezione 9), il rilevatore di frasi felici della lezione 4 non batte la baseline,
  la felicità di Pinocchio non cresce verso il lieto fine (lezione 8).
- Il link Colab e la cella di preparazione puntano al ramo `main` di questo repository.
