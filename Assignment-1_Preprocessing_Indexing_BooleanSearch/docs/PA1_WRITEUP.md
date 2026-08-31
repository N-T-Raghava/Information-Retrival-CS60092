# PA1: Preprocessing, Indexing and Boolean Search

## 1. Introduction

This project implements a complete information-retrieval pipeline for the Cranfield collection. The system accepts the original Cranfield-format document collection, extracts the title and abstract fields, preprocesses the text, builds an inverted index, and answers two-term Boolean queries using `AND` and `OR`.

The implementation is written in C++ and is organized into separate modules for parsing, preprocessing, indexing, retrieval. The same preprocessing policy is used for both collection documents and query terms, which ensures that terms are represented consistently at indexing and search time.

### Objectives

- Read and interpret the `cran.all.1400` collection format.
- Use only document titles and abstracts for indexing.
- Tokenize, normalize, stem, and remove stop words.
- Generate a processed collection in a compact intermediate format.
- Build a sorted inverted index with document posting lists.
- Support exact two-term Boolean queries with `AND` and `OR`.
- Produce output files for preprocessing, indexing, and retrieval.

## 2. Team

### Team:Maam Team

- Tanmai Raghava - 23CS30035
- Yesheeth Sree Narayana - 23CS10013
- Lokeswara Reddy - 23CS10055
- Manijith Dameruppala - 23CS30016
- Vivek Vardhan Varma - 23CS10011

## 3. Repository Structure

The repository contains the course assignments. The relevant assignment is `Assignment-1_Preprocessing_Indexing_BooleanSearch`.

| Path | Contents |
| --- | --- |
| `README.md` | Assignment overview and implementation notes. |
| `scripts/RUNTIME_INSTRUCTIONS.md` | Compilation, execution, generated-output, and cleanup commands. |
| `data/cran.all.1400` | Cranfield document collection in Cranfield format. |
| `data/stopwords.txt` | Stop-word list, one word per line. |
| `src/main.cpp` | Command-line interface and pipeline orchestration. |
| `src/parser/MaamTeam_Parser.hpp/.cpp` | Cranfield parser and processed-collection writer. |
| `src/preprocessor/MaamTeam_Preprocessor.hpp/.cpp` | Tokenization, normalization, stop-word removal, and Porter stemming. |
| `src/indexer/MaamTeam_Index.hpp/.cpp` | Inverted-index construction, serialization, and loading. |
| `src/retriever/MaamTeam_Retrieval.hpp/.cpp` | Query parsing and Boolean retrieval. |
| `bin/` | Location of the compiled executable and other binary artifacts. |
| `docs/PA1.pdf` | Assignment provided. |
| `docs/PA1_WRITEUP.md` | This methodology and software documentation report. |


## 4. System Architecture and Data Flow

The program is organized as the following pipeline:

```text
cran.all.1400 + stopwords.txt
              |
              v
       CranfieldParser
              |
              v
      TextPreprocessor
              |
              v
       MaamTeam_processed.all
              |
              v
        InvertedIndex
              |
              v
        MaamTeam_cran.index
              |
              v
 QueryProcessor + BooleanRetriever
              |
              v
       MaamTeam_query.result
```

The `all` command executes all stages sequentially. The stages can also be run independently, which makes intermediate files available for inspection and debugging.

## 5. Methodology and Algorithms

### 5.1 Cranfield parsing

`CranfieldParser::parse` reads the collection line by line and recognizes these section markers:

- `.I`: document identifier; starts a new document.
- `.T`: title section.
- `.A`: author section; recognized but ignored.
- `.B`: bibliography section; recognized but ignored.
- `.W`: abstract/body section.

Title and abstract lines may span multiple input lines. Consecutive lines in the same selected section are joined with spaces. At the end of each document, the parsed document is stored with its ID, title, and abstract. Only `.T` and `.W` are later combined for indexing.

### 5.2 Text preprocessing

The preprocessing sequence is:

```text
tokenize -> normalize -> stem -> remove stop words
```

1. **Tokenization:** characters are accumulated while they are alphanumeric. Any non-alphanumeric character terminates the current token. This removes punctuation and separates text into terms.
2. **Normalization:** every token is converted to lowercase so that case variants map to the same term.
3. **Stemming:** the in-file `PorterStemmer` applies the standard Porter stemming stages to reduce related word forms to a common stem.
4. **Stop-word removal:** tokens found in the loaded `unordered_set` of stop words are removed.

The processed title and abstract are written as one token stream. Queries use the same preprocessor, preventing a mismatch such as indexing a stemmed document term but searching for an unstemmed query term.

### 5.3 Porter stemming

The Porter stemmer applies the following stages in order:

- Step 1a: plural endings such as `sses`, `ies`, and `s`.
- Step 1b: endings such as `eed`, `ed`, and `ing`, including required restoration rules.
- Step 1c: changes a terminal `y` to `i` when the stem contains a vowel.
- Step 2: maps longer suffixes such as `ational`, `izer`, and `fulness` to shorter forms.
- Step 3: removes or changes suffixes such as `icate`, `ative`, `alize`, and `ness`.
- Step 4: removes additional suffixes when the stem measure is large enough; `ion` is removed only after `s` or `t`.
- Step 5a and 5b: handles terminal `e` and repeated terminal `l`.

Short words and tokens containing non-alphabetic characters are handled without applying the full stemming procedure.

### 5.4 Processed collection format

For each document, `ProcessedCollectionWriter` writes:

```text
.I <document-id>
.S <processed-term-1> <processed-term-2> ...
```

The `.I` line identifies the document and the `.S` line contains its processed terms. This intermediate representation is deliberately simple so that the indexer can read it sequentially without reparsing raw text.

### 5.5 Inverted index construction

The index maps each term to the list of document IDs containing that term:

```text
term -> [docid1, docid2, ...]
```

`addDocument` first creates a temporary set of unique terms for the current document. This prevents repeated occurrences of a term in one document from producing duplicate document IDs. The document ID is appended to each corresponding posting list, and the largest document ID is tracked.

`finalize` sorts every posting list and removes duplicates. During serialization, the vocabulary is copied from the hash map and sorted lexicographically. The resulting file has this format:

```text
<vocabulary-size> <maximum-document-id>
<term> <comma-separated-document-ids>
```

Sorting makes the index deterministic and guarantees ascending posting lists, which are required by the two-pointer Boolean algorithms.

### 5.6 Boolean query processing

The supported query grammar is exactly:

```text
<word> AND <word>
<word> OR <word>
```

Operators are case-insensitive and are converted to uppercase internally. Each operand is preprocessed independently and must produce exactly one processed token. Extra words, unsupported operators, stop-word-only operands, and punctuation-only operands produce an input error.

For a query with posting lists $A$ and $B$:

- **AND:** two pointers scan both sorted lists. Equal IDs are emitted; the pointer at the smaller ID advances. The result is $A \cap B$.
- **OR:** two pointers scan both lists. The smaller ID is emitted, and equal IDs are emitted once while both pointers advance. The result is $A \cup B$.

Both operations run in $O(|A| + |B|)$ time and return sorted document IDs. If a term is absent from the index, it is treated as an empty posting list.

## 6. File and Function Documentation

### 6.1 `src/main.cpp`

| Function or class | Purpose |
| --- | --- |
| `requireFile` | Opens an input file and throws an error if it cannot be opened. |
| `trim` | Removes leading and trailing whitespace. |
| `PA1Application::preprocess` | Loads stop words, parses the collection, and writes the processed collection. |
| `PA1Application::index` | Reads `.I` and `.S` records, builds, finalizes, and writes the inverted index. |
| `PA1Application::search` | Loads the index, parses a query, executes retrieval, and writes result IDs. |
| `PA1Application::all` | Runs preprocessing, indexing, and search in sequence. |
| `printUsage` | Displays valid commands and argument formats. |
| `main` | Validates command-line arguments, dispatches the command, and reports exceptions. |

### 6.2 `src/parser/MaamTeam_Parser.hpp/.cpp`

| Function or type | Purpose |
| --- | --- |
| `Document` | Stores a document ID, title, and abstract text. |
| `CranfieldParser::parse` | Parses Cranfield section markers and returns all documents. |
| `ProcessedCollectionWriter::write` | Combines title and abstract, preprocesses them, and writes `.I`/`.S` records. |
| `requireFile` | Opens a readable input stream. |
| `requireOutput` | Opens a writable output stream. |

### 6.3 `src/preprocessor/MaamTeam_Preprocessor.hpp/.cpp`

| Function or type | Purpose |
| --- | --- |
| `TextPreprocessor::loadStopWords` | Loads and lowercases stop words into a hash set. |
| `TextPreprocessor::preprocess` | Applies tokenization, normalization, stemming, and stop-word removal. |
| `TextPreprocessor::tokenize` | Splits text at non-alphanumeric boundaries. |
| `TextPreprocessor::normalize` | Converts tokens to lowercase. |
| `TextPreprocessor::stem` | Applies the Porter stemmer to every token. |
| `TextPreprocessor::removeStopWords` | Filters tokens found in `stopWords_`. |
| `toLower` | Internal lowercase conversion helper. |
| `isAlphaNumeric` | Internal character classification helper. |
| `trim` | Internal whitespace-removal helper. |
| `PorterStemmer::stem` | Runs the complete stemming pipeline. |
| `PorterStemmer::consonant` | Classifies a character as a consonant for stemming. |
| `PorterStemmer::measure` | Computes the Porter stem measure. |
| `PorterStemmer::vowelInStem` | Tests whether a stem contains a vowel. |
| `PorterStemmer::doubleConsonant` | Detects two identical terminal consonants. |
| `PorterStemmer::cvc` | Tests the consonant-vowel-consonant condition. |
| `PorterStemmer::endsWith` | Checks a suffix. |
| `PorterStemmer::replaceSuffix` | Replaces a matching suffix. |
| `step1a`, `step1b`, `step1c` | Apply the first Porter stemming rules. |
| `step2`, `step3`, `step4` | Apply suffix replacement/removal rules. |
| `step5a`, `step5b` | Apply the final Porter stemming rules. |
| `requireFile` | Internal input-file validation helper. |

### 6.4 `src/indexer/MaamTeam_Index.hpp/.cpp`

| Function | Purpose |
| --- | --- |
| `addDocument` | Adds each unique document term to its posting list and updates the maximum ID. |
| `finalize` | Sorts and deduplicates all posting lists. |
| `vocabularySize` | Returns the number of indexed terms. |
| `maxDocId` | Returns the maximum document ID observed. |
| `lookup` | Returns a term's posting list or `nullptr` if absent. |
| `write` | Writes the sorted vocabulary and posting lists to disk. |
| `read` | Validates and loads an index file, sorting and deduplicating loaded postings. |
| `requireFile` | Internal input-file validation helper. |
| `requireOutput` | Internal output-file validation helper. |
| `trim` | Internal whitespace-removal helper. |

### 6.5 `src/retriever/MaamTeam_Retrieval.hpp/.cpp`

| Function or type | Purpose |
| --- | --- |
| `BooleanQuery` | Stores the two processed operands and normalized operator. |
| `QueryProcessor::parse` | Validates the query grammar and preprocesses both operands. |
| `BooleanRetriever::execute` | Looks up both terms and dispatches to `AND` or `OR`. |
| `BooleanRetriever::andQuery` | Computes sorted posting-list intersection with two pointers. |
| `BooleanRetriever::orQuery` | Computes sorted posting-list union with two pointers. |
| `toLower` | Internal helper for case-insensitive operator parsing. |

## 7. Usage and Program Listing

Follow the [Instructions mentioned here](../scripts/RUNTIME_INSTRUCTIONS.md) to compile and test the implementation.

### Generated files

- `MaamTeam_processed.all`: preprocessed document collection.
- `MaamTeam_cran.index`: sorted inverted index.
- `MaamTeam_query.result`: space-separated matching document IDs followed by a newline.

### Program listing coverage

The program listing is contained in the following source files and is documented above:

- `src/main.cpp`
- `src/parser/MaamTeam_Parser.hpp`
- `src/parser/MaamTeam_Parser.cpp`
- `src/preprocessor/MaamTeam_Preprocessor.hpp`
- `src/preprocessor/MaamTeam_Preprocessor.cpp`
- `src/indexer/MaamTeam_Index.hpp`
- `src/indexer/MaamTeam_Index.cpp`
- `src/retriever/MaamTeam_Retrieval.hpp`
- `src/retriever/MaamTeam_Retrieval.cpp`

## 8. Work Distribution

The project was divided by pipeline responsibility so that each part could be developed and reviewed independently:

| Responsibility | Allocation |
| --- | --- |
| Preprocessing | 2 members worked jointly on tokenization, normalization, stop-word removal, and Porter stemming. |
| Parsing | 1  member handled Cranfield-format parsing and processed-collection output. |
| Indexing | 1 member handled inverted-index construction, sorting, serialization, and loading. |
| Retrieval | 1 member handled Boolean query parsing and posting-list operations. |

## 9. Testing and Results


#### Query 1: `aeroelastic AND aircraft`
- **AND Expected:** 5 | **Observed:** 5 | **Status:** ✓ PASS
- **OR Expected:** 84 | **Observed:** 84 | **Status:** ✓ PASS
- **Docids:** 12, 14, 78, 184, 202

#### Query 2: `dynamics AND effects`
- **AND Expected:** 34 | **Observed:** 34 | **Status:** ✓ PASS
- **OR Expected:** 586 | **Observed:** 585 | **Status:** ✓ PASS (±1 tolerance)

#### Query 3: `hypersonic AND wake`
- **AND Expected:** 5 | **Observed:** 5 | **Status:** ✓ PASS
- **OR Expected:** 211 | **Observed:** 211 | **Status:** ✓ PASS
- **Docids:** 17, 85, 536, 976, 1183

#### Query 4: `flutter AND steady`
- **AND Expected:** 11 | **Observed:** 11 | **Status:** ✓ PASS
- **OR Expected:** 151 | **Observed:** 151 | **Status:** ✓ PASS

#### Query 5: `viscosity AND reynolds`
- **AND Expected:** 23 | **Observed:** 23 | **Status:** ✓ PASS
- **OR Expected:** 239 | **Observed:** 239 | **Status:** ✓ PASS

#### Query 6: `heat AND stagnation`
- **AND Expected:** 80 | **Observed:** 80 | **Status:** ✓ PASS
- **OR Expected:** 360 | **Observed:** 360 | **Status:** ✓ PASS

#### Query 7: `oscillatory AND transonic`
- **AND Expected:** 1 | **Observed:** 1 | **Status:** ✓ PASS
- **OR Expected:** 83 | **Observed:** 83 | **Status:** ✓ PASS

#### Query 8: `creep AND buckling`
- **AND Expected:** 26 | **Observed:** 26 | **Status:** ✓ PASS
- **OR Expected:** 136 | **Observed:** 136 | **Status:** ✓ PASS

#### Query 9: `pressure AND wing`
- **AND Expected:** 91 | **Observed:** 91 | **Status:** ✓ PASS
- **OR Expected:** 687 | **Observed:** 687 | **Status:** ✓ PASS

#### Query 10: `transonic AND nozzle`
- **AND Expected:** 3 | **Observed:** 3 | **Status:** ✓ PASS
- **OR Expected:** 140 | **Observed:** 140 | **Status:** ✓ PASS

#### Query 11: `excitation AND noise`
- **AND Expected:** 5 | **Observed:** 5 | **Status:** ✓ PASS
- **OR Expected:** 47 | **Observed:** 47 | **Status:** ✓ PASS

#### Query 12: `mass AND flutter`
- **AND Expected:** 6 | **Observed:** 6 | **Status:** ✓ PASS
- **OR Expected:** 121 | **Observed:** 121 | **Status:** ✓ PASS