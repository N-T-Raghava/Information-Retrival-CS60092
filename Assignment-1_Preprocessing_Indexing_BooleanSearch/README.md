# PA1 Information Retrieval Project

This is a C++ implementation of the PA1 preprocessing, indexing, and Boolean search pipeline.

See [scripts/README.md](../scripts/README.md) for build, run, and test instructions.

## Notes

- Only the `.T` (title) and `.W` (abstract) sections are processed.
- Tokenization, stemming, stop-word removal, and normalization are implemented as separate functions.
- The index file is sorted lexicographically by term and stores comma-separated postings lists.
- Boolean search supports exactly two query terms with `AND` or `OR` and searches through the index.
