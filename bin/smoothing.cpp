#include <iostream>
#include <fstream>
#include <cstdio>

#include <Halide.h>
#include <halide_image_io.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <include/stb_image_write.h>

#include <guided_image_filter.h>

namespace {
    bool save_png(const std::string& img_path, const Halide::Runtime::Buffer<uint8_t> &img) {
        const int stride_in_bytes = img.width() * img.channels();
        if (!stbi_write_png(img_path.c_str(), img.width(), img.height(), img.channels(), img.data(), stride_in_bytes)) {
            std::cerr << "Unable to write output image '" << img_path << "'" << std::endl;
            return false;
        }
        return true;
    }
}


int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " input_img output_img" << std::endl;
        return 1;
    }

    const std::string input_img_path(argv[1]);
    const std::string output_img_path(argv[2]);

    Halide::Runtime::Buffer<uint8_t> input_img = Halide::Tools::load_image(input_img_path);
    Halide::Runtime::Buffer<uint8_t> output = Halide::Runtime::Buffer<uint8_t>::make_with_shape_of(input_img);

    guided_image_filter(input_img, input_img, 5, 255.f, output);

    if (!save_png(output_img_path, output)) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}