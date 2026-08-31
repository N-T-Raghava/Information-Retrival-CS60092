# PA1 Information Retrieval Project

This is a C++ implementation of the PA1 preprocessing, indexing, and Boolean search pipeline.

Refer [scripts/RUNTIME_INSTRUCTIONS.md](scripts/RUNTIME_INSTRUCTIONS.md) for build, run, and test instructions; And [docs/PA1_WRITEUP.md](docs/PA1_WRITEUP.md) for the methodology, repository structure,
software documentation, work distribution, and testing.

## Notes

- Only the `.T` (title) and `.W` (abstract) sections are processed.
- Tokenization, stemming, stop-word removal, and normalization are implemented as separate functions.
- The index file is sorted lexicographically by term and stores comma-separated postings lists.
- Boolean search supports exactly two query terms with `AND` or `OR` and searches through the index.

## Testing with Sample Queries
Run the verification script to test all 12 sample queries:

- For more testing, Add the new queries directly to the [tests\sample_queries.md](tests\sample_queries.md), and test it using the below commands

```powershell
cd tests
.\verify_queries.ps1
```