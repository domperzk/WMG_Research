#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "recordReader.h"
#include <utility>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <complex>

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

public:
    ImageClass(size_t width, size_t height, std::string filename) {
        m_width = width;
        m_height = height;
        m_filename = std::move(filename);
    }

    void load_data() {
        m_data.imageData.resize(m_height, std::vector<std::vector<float>>(m_width));
        
        auto reader = make_record_reader(m_filename);
        if (!reader->is_valid()) {
            std::cerr << "Error: Could not open file " << m_filename << std::endl;
            return;
        }

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
        
        std::cout << "[load_data] Auto-detected amplitude range: +/- " << range << std::endl;

        reader = make_record_reader(m_filename); 
        int total_loaded = 0, skipped_out_of_bounds = 0, skipped_zero_weight = 0;
        m_samples_accumulated = 0;

        DataRecord record;
        while (reader->next(record)) {
            m_samples_accumulated++;

            float normalized_x = (record.v1 + range) / (2.0f * range);
            float normalized_y = (record.v2 + range) / (2.0f * range);

            // FIX: Multiply by (m_width - 1) so maximum values sit safely at 99 instead of falling off at 100
            int x = static_cast<int>(normalized_x * (m_width - 1));
            int y = static_cast<int>(normalized_y * (m_height - 1));
            float weight = record.v3;

            if (x < 0 || x >= static_cast<int>(m_width) || y < 0 || y >= static_cast<int>(m_height)) {
                skipped_out_of_bounds++;
                continue;
            }

            // FIX: Only skip if literally zero. Quantum probabilities are microscopic!
            if (weight == 0.0f) {
                skipped_zero_weight++;
                continue;
            }

            m_data.imageData[y][x].push_back(1.0f / weight);
            total_loaded++;
        }

        std::cout << "[load_data] Processed dataset. Kept: " << total_loaded
                  << " points, Skipped (Out of bounds): " << skipped_out_of_bounds
                  << ", Skipped (zero weight): " << skipped_zero_weight << std::endl;
    }

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
        pixels.resize(m_width * m_height * 3, 0); // Pre-fill the whole image with pure black

        int unique_pixels_found = 0;

        std::cout << "\n--- SCANNING GRID FOR SPECKS ---\n";

        for (size_t y = 0; y < m_height; ++y) {
            for (size_t x = 0; x < m_width; ++x) {
                if (!m_data.imageData[y][x].empty()) {
                    unique_pixels_found++;
                    std::cout << "  -> Speck found at coordinate (x: " << x << ", y: " << y << ")!\n";

                    // DRAW A 3x3 WHITE SQUARE SO IT'S ACTUALLY VISIBLE
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            int nx = static_cast<int>(x) + dx;
                            int ny = static_cast<int>(y) + dy;
                            
                            // Keep the box inside the image bounds
                            if (nx >= 0 && nx < m_width && ny >= 0 && ny < m_height) {
                                int index = (ny * m_width + nx) * 3;
                                pixels[index] = 255;     // Red
                                pixels[index + 1] = 255; // Green
                                pixels[index + 2] = 255; // Blue
                            }
                        }
                    }
                }
            }
        }

        std::cout << "--------------------------------------\n";
        std::cout << "[transform_to_image] Total unique specks drawn: " << unique_pixels_found << "\n";

        if (!stbi_write_png(output_png_filename.c_str(), static_cast<int>(m_width), static_cast<int>(m_height), 3, pixels.data(), static_cast<int>(m_width * 3))) {
            std::cerr << "Failed to write PNG.\n";
            return;
        }
        std::cout << "[transform_to_image] Image written to " << output_png_filename << '\n';
    }
};

int main() {
    std::cout << "=== Starting Pipeline ===" << std::endl;
    size_t grid_width = 100;
    size_t grid_height = 100;
    std::string mldata_path = "circuit_900012.mldata"; 

    ImageClass processor(grid_width, grid_height, mldata_path);
    processor.load_data();
    
    std::complex<float> expected_value = processor.compute_expected_value();
    std::cout << "Expected Value: " << expected_value.real() << " + " << expected_value.imag() << "i" << std::endl;
    
    processor.transform_to_image("output_grid.png");
    
    std::cout << "=== Done! ===" << std::endl;
    return 0;
}
