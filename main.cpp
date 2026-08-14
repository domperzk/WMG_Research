#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "recordReader.h"
#include <utility>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <complex>
#include <random>
#include <sys/stat.h>

struct ImageDatabase {
    std::vector<std::vector<std::vector<float>>> imageData;
};

class ImageClass {
private:
    ImageDatabase m_data{};
    size_t m_width{0};
    size_t m_height{0};
    std::string m_filename{};
    long long m_samples_accumulated{0};
    
    // Cached stats so we only read the 37M file for boundaries once!
    float m_range{1.0f};
    long long m_total_records_in_file{0};
    bool m_stats_initialized{false};

public:
    ImageClass(size_t width, size_t height, std::string filename) {
        m_width = width;
        m_height = height;
        m_filename = std::move(filename);
    }

    void init_file_stats() {
        std::cout << "[init_stats] Scanning file for boundaries and total count... (This only happens once)\n";
        auto reader = make_record_reader(m_filename);
        if (!reader->is_valid()) {
            std::cerr << "Error: Could not open file " << m_filename << std::endl;
            return;
        }

        float min_val = 999999.0f, max_val = -999999.0f;
        m_total_records_in_file = 0;
        
        DataRecord temp_rec;
        while (reader->next(temp_rec)) {
            m_total_records_in_file++;
            if (temp_rec.v1 < min_val) min_val = temp_rec.v1;
            if (temp_rec.v1 > max_val) max_val = temp_rec.v1;
            if (temp_rec.v2 < min_val) min_val = temp_rec.v2;
            if (temp_rec.v2 > max_val) max_val = temp_rec.v2;
        }
        
        m_range = std::max(std::abs(min_val), std::abs(max_val));
        if (m_range == 0.0f) m_range = 1.0f;
        m_stats_initialized = true;
        
        std::cout << "[init_stats] Auto-detected amplitude range: +/- " << m_range << std::endl;
        std::cout << "[init_stats] Total paths found: " << m_total_records_in_file << "\n\n";
    }

    void load_data(double percentage, bool random_sample) {
        if (!m_stats_initialized) init_file_stats();
        
        // Reset image grid for the new run
        m_data.imageData.assign(m_height, std::vector<std::vector<float>>(m_width));
        m_samples_accumulated = 0;

        long long target_records = static_cast<long long>(m_total_records_in_file * percentage);
        
        std::cout << "[load_data] Mode: " << (random_sample ? "RANDOM" : "SEQUENTIAL") 
                  << " | " << (percentage * 100) << "% data.\n";
        std::cout << "[load_data] Target paths to read: " << target_records << "\n";

        auto reader = make_record_reader(m_filename); 
        int total_loaded = 0, skipped_out_of_bounds = 0, skipped_zero_weight = 0;

        // If asking for 100%, random and sequential are literally the exact same thing (all of it).
        if (!random_sample || percentage >= 1.0) {
            // --- SEQUENTIAL LOGIC ---
            DataRecord record;
            while (reader->next(record)) {
                if (m_samples_accumulated >= target_records) break;
                m_samples_accumulated++;
                process_record(record, total_loaded, skipped_out_of_bounds, skipped_zero_weight);
            }
        } else {
            // --- RANDOM LOGIC (Knuth's Algorithm S) ---
            // This mathematical trick picks a perfectly unbiased random sample 
            // of any size in a single fast sweep, avoiding memory crashes!
            std::mt19937_64 rng(42); 
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            
            long long records_left_to_pick = target_records;
            long long records_left_in_file = m_total_records_in_file;
            
            DataRecord record;
            while (reader->next(record) && records_left_to_pick > 0) {
                // Dynamically adjust the probability of picking the current record
                double prob = static_cast<double>(records_left_to_pick) / records_left_in_file;
                
                if (dist(rng) < prob) {
                    m_samples_accumulated++;
                    process_record(record, total_loaded, skipped_out_of_bounds, skipped_zero_weight);
                    records_left_to_pick--;
                }
                records_left_in_file--;
            }
        }

        std::cout << "[load_data] Kept: " << total_loaded << " points.\n";
    }

private:
    // Helper function to keep load_data clean
    inline void process_record(const DataRecord& record, int& total_loaded, int& skipped_out_of_bounds, int& skipped_zero_weight) {
        float normalized_x = (record.v1 + m_range) / (2.0f * m_range);
        float normalized_y = (record.v2 + m_range) / (2.0f * m_range);

        int x = static_cast<int>(normalized_x * (m_width - 1));
        int y = static_cast<int>(normalized_y * (m_height - 1));
        float weight = record.v3;

        if (x < 0 || x >= static_cast<int>(m_width) || y < 0 || y >= static_cast<int>(m_height)) {
            skipped_out_of_bounds++;
            return;
        }

        if (weight == 0.0f) {
            skipped_zero_weight++;
            return;
        }

        m_data.imageData[y][x].push_back(1.0f / weight);
        total_loaded++;
    }

public:
    std::complex<float> compute_expected_value() const {
        std::complex<float> pixel_sum(0.0f, 0.0f);
        for (size_t y = 0; y < m_height; ++y) {
            for (size_t x = 0; x < m_width; ++x) {
                if (!m_data.imageData[y][x].empty())
                    pixel_sum += compute_pixel_value(x, y);
            }
        }
        if (m_samples_accumulated == 0) return {0.0f, 0.0f};
        return pixel_sum / static_cast<float>(m_samples_accumulated);
    }

    std::complex<float> compute_pixel_value(size_t x, size_t y) const {
        float intensity = 0.0f;
        for (float contribution : m_data.imageData[y][x])
            intensity += contribution;
        std::complex<float> cell_centre(static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f);
        return cell_centre * intensity;
    }

    void transform_to_image(const std::string& output_png_filename) {
        std::vector<unsigned char> pixels;
        pixels.resize(m_width * m_height * 3, 0); 

        int unique_pixels_found = 0;

        for (size_t y = 0; y < m_height; ++y) {
            for (size_t x = 0; x < m_width; ++x) {
                if (!m_data.imageData[y][x].empty()) {
                    unique_pixels_found++;

                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            int nx = static_cast<int>(x) + dx;
                            int ny = static_cast<int>(y) + dy;
                            
                            if (nx >= 0 && nx < m_width && ny >= 0 && ny < m_height) {
                                int index = (ny * m_width + nx) * 3;
                                pixels[index] = 255;
                                pixels[index + 1] = 255;
                                pixels[index + 2] = 255;
                            }
                        }
                    }
                }
            }
        }

        if (!stbi_write_png(output_png_filename.c_str(), static_cast<int>(m_width), static_cast<int>(m_height), 3, pixels.data(), static_cast<int>(m_width * 3))) {
            std::cerr << "Failed to write PNG.\n";
            return;
        }
        std::cout << "[transform_to_image] Found " << unique_pixels_found << " unique specks. Saved to " << output_png_filename << "\n";
    }
};

int main() {
    std::cout << "=== Starting Sequential vs Random Batch ===\n\n";
    
    // Create output directories
    mkdir("sequential", 0777);
    mkdir("random", 0777);

    size_t grid_width = 100;
    size_t grid_height = 100;
    std::string mldata_path = "circuit_900012.mldata"; 

    ImageClass processor(grid_width, grid_height, mldata_path);
    
    // The FULL test suite: 100% all the way down to 0.0001%
    std::vector<std::pair<double, std::string>> tests = {
        {1.0, "100_pct"},
        {0.50, "50_pct"},
        {0.25, "25_pct"},
        {0.10, "10_pct"},
        {0.01, "1_pct"},
        {0.001, "0.1_pct"},
        {0.0001, "0.01_pct"},
        {0.00001, "0.001_pct"},
        {0.000001, "0.0001_pct"} 
    };

    for (const auto& test : tests) {
        double p = test.first;
        std::string name_base = test.second;
        
        std::cout << "======================================\n";
        
        // --- RUN 1: SEQUENTIAL ---
        processor.load_data(p, false);
        std::complex<float> ev_seq = processor.compute_expected_value();
        std::cout << "Sequential Expected Value: " << ev_seq.real() << " + " << ev_seq.imag() << "i\n";
        processor.transform_to_image("sequential/output_" + name_base + ".png");
        
        std::cout << "\n";

        // --- RUN 2: RANDOM ---
        processor.load_data(p, true);
        std::complex<float> ev_rand = processor.compute_expected_value();
        std::cout << "Random Expected Value:     " << ev_rand.real() << " + " << ev_rand.imag() << "i\n";
        processor.transform_to_image("random/output_" + name_base + "_rand.png");
        
        std::cout << "======================================\n\n";
    }

    std::cout << "=== All Sequential and Random Images Generated! ===\n";
    return 0;
}