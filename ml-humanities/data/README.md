# I dati del corso

Tutti i file sono in UTF-8 e sono stati scaricati una sola volta, così il corso funziona anche **offline**.

## Testi (`testi/`)

Opere di pubblico dominio da [Project Gutenberg](https://www.gutenberg.org), scaricate tramite il mirror GitHub del progetto
[GITenberg](https://github.com/GITenberg). Abbiamo rimosso l'intestazione e la licenza di Project Gutenberg, i frontespizi,
gli indici, le note dell'editore e le didascalie delle illustrazioni. Ogni file inizia con alcune righe di intestazione che
cominciano con `#` (titolo, autore, numero dell'eBook); i notebook le saltano.

| File | Opera | Autore | eBook # | Parole |
|---|---|---|---|---|
| `collodi_pinocchio.txt` | *Le avventure di Pinocchio* (ed. Bemporad 1902) | Carlo Collodi | 52484 | ~42.000 |
| `manzoni_promessi_sposi.txt` | *I promessi sposi* (solo il romanzo, senza saggio introduttivo) | Alessandro Manzoni | 45334 | ~214.000 |
| `dante_commedia.txt` | *La Divina Commedia* | Dante Alighieri | 1012 | ~98.000 |
| `leopardi_canti.txt` | *Canti* (40 componimenti) | Giacomo Leopardi | 55236 | ~24.000 |
| `deledda_dopo_il_divorzio.txt` | *Dopo il divorzio* (1902) | Grazia Deledda | 43226 | ~59.000 |
| `pirandello_cavallo_nella_luna.txt` | *Un cavallo nella luna* (novelle, 1918) | Luigi Pirandello | 56775 | ~36.000 |
| `pirandello_umorismo.txt` | *L'umorismo* (saggio, 2ª ed. 1920) | Luigi Pirandello | 56958 | ~50.000 |
| `digiacomo_novelle_napolitane.txt` | *Novelle napolitane* (senza la prefazione) | Salvatore Di Giacomo | 57040 | ~65.000 |

## Tabelle derivate

| File | Contenuto |
|---|---|
| `commedia_canti.csv` | 100 righe: `cantica`, `canto`, `posizione` (1–100), `testo` |
| `leopardi_canti.csv` | 40 righe: `numero`, `numero_romano`, `titolo`, `testo` |
| `pinocchio_capitoli.csv` | 36 righe: `capitolo`, `numero_romano`, `titolo`, `testo` |
| `promessi_sposi_capitoli.csv` | 38 righe: `capitolo`, `numero_romano`, `testo` (utile per i progetti) |
| `passaggi_autori.csv` | 600 brani di circa 200 parole (paragrafi interi): 120 ciascuno di Collodi, Manzoni, Deledda, Pirandello (novelle), Di Giacomo. Colonne: `autore`, `opera`, `id_passaggio`, `testo` |

## Il lessico della felicità

| File | Contenuto |
|---|---|
| `lessico_felicita.csv` | 25 parole della felicità (e dei suoi confini) in 14 lingue, ciascuna con la sua sfumatura: `lingua`, `parola`, `traslitterazione`, `sfumatura` |
| `famiglie_felicita_it.csv` | Le 35 forme italiane usate per contare le parole felici, divise nelle sei famiglie *felicità, gioia, allegria, contentezza, letizia, beatitudine* |
| `stopwords_it.txt` | Una breve lista di parole funzionali italiane, fatta a mano: modificatela! |

**Limiti da tenere a mente.** Le 35 forme non contano parole come *infelice*, *sorriso* o *festa*, né le negazioni
(*"non era felice"* conta come felice). Ogni autore è rappresentato da un solo libro, tranne Pirandello (due), e su questo
lavora la lezione 9. Le edizioni hanno grafie diverse: per esempio Manzoni usa le virgolette «» per i dialoghi, gli altri il trattino.
