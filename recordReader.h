#pragma once
#include <fstream>
#include <string>
#include <memory>

struct DataRecord { float v1, v2, v3; };

class RecordReader {
public:
    virtual bool next(DataRecord& record) = 0;
    virtual bool is_valid() const = 0;
    virtual ~RecordReader() = default;
};

class BinaryFloatTripleReader : public RecordReader {
    std::ifstream file;
public:
    BinaryFloatTripleReader(const std::string& path) {
        file.open(path, std::ios::binary);
    }
    bool is_valid() const override { return file.is_open(); }
    bool next(DataRecord& record) override {
        float buf[3];
        if (file.read(reinterpret_cast<char*>(buf), sizeof(float) * 3)) {
            record.v1 = buf[0];
            record.v2 = buf[1];
            record.v3 = buf[2];
            return true;
        }
        return false;
    }
};

inline std::unique_ptr<RecordReader> make_record_reader(const std::string& filename) {
    return std::make_unique<BinaryFloatTripleReader>(filename);
}