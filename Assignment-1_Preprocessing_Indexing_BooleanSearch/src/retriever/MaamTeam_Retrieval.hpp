#pragma once

#include "../indexer/MaamTeam_Index.hpp"

#include <string>
#include <vector>

namespace pa1 {

class TextPreprocessor;  // Forward declaration

/**
 * Represents a parsed Boolean query with two operands.
 */
struct BooleanQuery {
    std::string left;   ///< Left operand (processed term)
    std::string op;     ///< Operator: "AND" or "OR"
    std::string right;  ///< Right operand (processed term)
};

/**
 * Parses raw Boolean queries and preprocesses query terms.
 * 
 * Supported format: "WORD AND WORD" or "WORD OR WORD"
 * Case-insensitive operators.
 * Query operands are preprocessed (tokenized, normalized, stemmed, stopwords removed).
 */
class QueryProcessor {
public:
    /**
     * Parses a raw Boolean query.
     * @param query           Raw query string, e.g., "wing AND flow"
     * @param preprocessor    TextPreprocessor instance
     * @return                Parsed query with processed terms
     * @throws                std::invalid_argument if format is invalid
     */
    BooleanQuery parse(const std::string& query, const TextPreprocessor& preprocessor) const;
};

/**
 * Executes Boolean AND/OR operations on posting lists.
 * 
 * AND: Two-pointer intersection (O(|A| + |B|))
 * OR:  Two-pointer union (O(|A| + |B|))
 */
class BooleanRetriever {
public:
    using PostingList = InvertedIndex::PostingList;

    /**
     * Executes a Boolean query against the index.
     * @param index      InvertedIndex instance
     * @param left       Left term (already processed)
     * @param op         Operator ("AND" or "OR")
     * @param right      Right term (already processed)
     * @return           Sorted result document IDs
     * @throws           std::invalid_argument if operator is invalid
     */
    std::vector<int> execute(const InvertedIndex& index,
                             const std::string& left,
                             const std::string& op,
                             const std::string& right) const;

private:
    /**
     * Computes intersection of two posting lists (AND operation).
     */
    std::vector<int> andQuery(const PostingList& a, const PostingList& b) const;

    /**
     * Computes union of two posting lists (OR operation).
     */
    std::vector<int> orQuery(const PostingList& a, const PostingList& b) const;
};

}  // namespace pa1
