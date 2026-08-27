#include "MaamTeam_Parser.hpp"
#include "../preprocessor/MaamTeam_Preprocessor.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace pa1 {

namespace {

// Helper: open file for reading
static void requireFile(const std::string& path, std::ifstream& in) {
    in.open(path);
    if (!in) {
        throw std::runtime_error("Cannot open input file: " + path);
    }
}

// Helper: open file for writing
static void requireOutput(const std::string& path, std::ofstream& out) {
    out.open(path);
    if (!out) {
        throw std::runtime_error("Cannot create output file: " + path);
    }
}

}  // anonymous namespace

std::vector<Document> CranfieldParser::parse(const std::string& path) const {
    std::ifstream in;
    requireFile(path, in);

    std::vector<Document> documents;
    Document current;
    char section = '\0';
    std::string line;

    auto finishDocument = [&]() {
        if (current.id != 0) {
            documents.push_back(std::move(current));
            current = Document{};
        }
    };

    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();

        if (line.rfind(".I", 0) == 0) {
            finishDocument();
            std::istringstream iss(line.substr(2));
            iss >> current.id;
            section = '\0';
        } else if (line == ".T") {
            section = 'T';
        } else if (line == ".A") {
            section = 'A';
        } else if (line == ".B") {
            section = 'B';
        } else if (line == ".W") {
            section = 'W';
        } else {
            if (section == 'T') {
                if (!current.title.empty()) current.title += ' ';
                current.title += line;
            } else if (section == 'W') {
                if (!current.abstractText.empty()) current.abstractText += ' ';
                current.abstractText += line;
            }
        }
    }

    finishDocument();
    return documents;
}

void ProcessedCollectionWriter::write(const std::string& path,
                                      const std::vector<Document>& documents,
                                      const TextPreprocessor& preprocessor) const {
    std::ofstream out;
    requireOutput(path, out);

    for (const auto& doc : documents) {
        const std::string combined = doc.title + " " + doc.abstractText;
        const auto tokens = preprocessor.preprocess(combined);

        out << ".I " << doc.id << '\n';
        out << ".S";
        for (const auto& token : tokens) {
            out << ' ' << token;
        }
        out << '\n';
    }
}

}  // namespace pa1
