#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "recordReader.h"
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <sys/stat.h>
#include <fstream>
#include <algorithm>

struct FastRecord {
    int x, y;
    float inv_weight;
};

class MLDataGenerator {
private:
    size_t m_width{100};
    size_t m_height{100};
    std::string m_filename;
    std::vector<FastRecord> m_all_data;
    
public:
    MLDataGenerator(std::string filename) : m_filename(std::move(filename)) {}

    void load_all_to_ram() {
        std::cout << "[RAM Load] Scanning file and caching to memory...\n";
        auto reader = make_record_reader(m_filename);
        
        // Pass 1: Find bounds
        float min_val = 999999.0f, max_val = -999999.0f;
        DataRecord temp_rec;
        while (reader->next(temp_rec)) {
            if (temp_rec.v1 < min_val) min_val = temp_rec.v1;
            if (temp_rec.v1 > max_val) max_val = temp_rec.v1;
            if (temp_rec.v2 < min_val) min_val = temp_rec.v2;
            if (temp_rec.v2 > max_val) max_val = temp_rec.v2;
        }
        float range = std::max(std::abs(min_val), std::abs(max_val));
        if (range == 0.0f) range = 1.0f;

        // Pass 2: Cache to RAM
        reader = make_record_reader(m_filename);
        while (reader->next(temp_rec)) {
            float normalized_x = (temp_rec.v1 + range) / (2.0f * range);
            float normalized_y = (temp_rec.v2 + range) / (2.0f * range);
            int x = static_cast<int>(normalized_x * (m_width - 1));
            int y = static_cast<int>(normalized_y * (m_height - 1));
            
            if (x >= 0 && x < m_width && y >= 0 && y < m_height && temp_rec.v3 != 0.0f) {
                m_all_data.push_back({x, y, 1.0f / temp_rec.v3});
            }
        }
        std::cout << "[RAM Load] Successfully loaded " << m_all_data.size() << " valid paths into RAM.\n\n";
    }

    void generate_dataset(int num_samples, double percentage) {
        mkdir("ml_dataset", 0777);
        
        long long target_records = static_cast<long long>(m_all_data.size() * percentage);
        std::cout << "Generating " << num_samples << " samples of size " << target_records << " paths...\n";

        std::mt19937 rng(42);

        // --- 1. GENERATE GROUND TRUTH (100%) ---
        std::vector<float> ground_truth_grid(m_width * m_height, 0.0f);
        for (const auto& rec : m_all_data) {
            ground_truth_grid[rec.y * m_width + rec.x] += rec.inv_weight;
        }
        save_binary("ml_dataset/ground_truth.bin", ground_truth_grid);
        std::cout << "-> Saved Ground Truth (100%)\n";

        // --- 2. GENERATE RANDOM SUBSETS ---
        for (int i = 0; i < num_samples; ++i) {
            std::vector<float> sample_grid(m_width * m_height, 0.0f);
            
            // Fast random sampling from RAM
            std::vector<int> random_indices(target_records);
            std::uniform_int_distribution<int> dist(0, m_all_data.size() - 1);
            
            for(int j = 0; j < target_records; ++j) {
                const auto& rec = m_all_data[dist(rng)];
                sample_grid[rec.y * m_width + rec.x] += rec.inv_weight;
            }
            
            save_binary("ml_dataset/sample_" + std::to_string(i) + ".bin", sample_grid);
            
            if ((i + 1) % 200 == 0) {
                std::cout << "-> Generated " << (i + 1) << " / " << num_samples << " samples...\n";
            }
        }
        std::cout << "\n=== ML Dataset Generation Complete! ===\n";
    }

private:
    void save_binary(const std::string& filename, const std::vector<float>& grid) {
        std::ofstream out(filename, std::ios::binary);
        out.write(reinterpret_cast<const char*>(grid.data()), grid.size() * sizeof(float));
        out.close();
    }
};

int main() {
    MLDataGenerator generator("circuit_900012.mldata");
    
    // Load all data into RAM (takes ~2 seconds)
    generator.load_all_to_ram();
    
    // Generate 1000 samples at 0.001% data
    generator.generate_dataset(1000, 0.00001); 
    
    return 0;
}