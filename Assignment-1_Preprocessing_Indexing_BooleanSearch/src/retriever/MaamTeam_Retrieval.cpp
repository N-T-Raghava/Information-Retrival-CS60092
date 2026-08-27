#include "MaamTeam_Retrieval.hpp"
#include "../preprocessor/MaamTeam_Preprocessor.hpp"

#include <algorithm>
#include <stdexcept>
#include <sstream>

namespace pa1 {

namespace {

// Helper: convert to lowercase
static std::string toLower(std::string s) {
    for (char& ch : s) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return s;
}

}  // anonymous namespace

std::vector<int> BooleanRetriever::andQuery(const PostingList& a, const PostingList& b) const {
    std::vector<int> result;
    result.reserve(std::min(a.size(), b.size()));

    std::size_t i = 0;
    std::size_t j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] == b[j]) {
            result.push_back(a[i]);
            ++i;
            ++j;
        } else if (a[i] < b[j]) {
            ++i;
        } else {
            ++j;
        }
    }
    return result;
}

std::vector<int> BooleanRetriever::orQuery(const PostingList& a, const PostingList& b) const {
    std::vector<int> result;
    result.reserve(a.size() + b.size());

    std::size_t i = 0;
    std::size_t j = 0;
    while (i < a.size() || j < b.size()) {
        if (j == b.size() || (i < a.size() && a[i] < b[j])) {
            result.push_back(a[i++]);
        } else if (i == a.size() || b[j] < a[i]) {
            result.push_back(b[j++]);
        } else {
            result.push_back(a[i]);
            ++i;
            ++j;
        }
    }
    return result;
}

std::vector<int> BooleanRetriever::execute(const InvertedIndex& index,
                                           const std::string& left,
                                           const std::string& op,
                                           const std::string& right) const {
    const auto* leftPostings = index.lookup(left);
    const auto* rightPostings = index.lookup(right);

    static const PostingList empty;
    const PostingList& a = leftPostings ? *leftPostings : empty;
    const PostingList& b = rightPostings ? *rightPostings : empty;

    if (op == "AND") return andQuery(a, b);
    if (op == "OR") return orQuery(a, b);

    throw std::invalid_argument("Operator must be AND or OR.");
}

BooleanQuery QueryProcessor::parse(const std::string& query, const TextPreprocessor& preprocessor) const {
    std::istringstream iss(query);
    std::string rawLeft;
    std::string rawOp;
    std::string rawRight;
    std::string extra;

    if (!(iss >> rawLeft >> rawOp >> rawRight) || (iss >> extra)) {
        throw std::invalid_argument(
            "Query must contain exactly two words and one AND/OR operator, e.g. \"wing AND flow\"."
        );
    }

    rawOp = toLower(rawOp);
    std::string normalizedOp;
    if (rawOp == "and") normalizedOp = "AND";
    else if (rawOp == "or") normalizedOp = "OR";
    else throw std::invalid_argument("Only AND and OR are supported.");

    const auto leftTokens = preprocessor.preprocess(rawLeft);
    const auto rightTokens = preprocessor.preprocess(rawRight);

    if (leftTokens.size() != 1 || rightTokens.size() != 1) {
        throw std::invalid_argument(
            "Each query operand must produce exactly one processed token. Avoid stop words or punctuation-only operands."
        );
    }

    return {leftTokens.front(), normalizedOp, rightTokens.front()};
}

}  // namespace pa1
