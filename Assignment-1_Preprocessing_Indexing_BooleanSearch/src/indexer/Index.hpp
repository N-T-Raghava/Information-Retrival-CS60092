#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace pa1 {

/**
 * Inverted index for Boolean retrieval.
 * 
 * Maps terms to sorted, unique posting lists (document IDs).
 */
class InvertedIndex {
public:
    using PostingList = std::vector<int>;

    /**
     * Adds a document's terms to the index.
     * Automatically deduplicates terms within a document.
     * @param docId   Document ID
     * @param tokens  Preprocessed tokens from the document
     */
    void addDocument(int docId, const std::vector<std::string>& tokens);

    /**
     * Finalizes the index after all documents have been added.
     * Sorts and deduplicates posting lists.
     */
    void finalize();

    /**
     * Returns the number of unique terms in the vocabulary.
     */
    std::size_t vocabularySize() const { return postings_.size(); }

    /**
     * Returns the maximum document ID seen.
     */
    int maxDocId() const { return maxDocId_; }

    /**
     * Looks up a term in the index.
     * @param term  The term to look up
     * @return      Pointer to posting list, or nullptr if term not found
     */
    const PostingList* lookup(const std::string& term) const;

    /**
     * Writes the index to a file.
     * Format:
     *   <vocabulary_size> <max_docid>
     *   <term> <comma-separated docids>
     *   ...
     * Terms are in lexicographic order, docids in ascending order within each posting list.
     * @param path  Output file path
     */
    void write(const std::string& path) const;

    /**
     * Reads the index from a file.
     * @param path  Input file path
     * @throws      std::runtime_error on parse errors
     */
    void read(const std::string& path);

private:
    std::unordered_map<std::string, PostingList> postings_;
    int maxDocId_ = 0;
};

}  // namespace pa1
