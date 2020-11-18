#include <Halide.h>

#include <utility>

using namespace Halide;
using namespace Halide::ConciseCasts;

namespace {
    class GuidedPipeline : public Halide::Generator<GuidedPipeline> {
    public:
        Input<Halide::Buffer<uint8_t>> input{"input", 2};
        Input<Halide::Buffer<uint8_t>> guidance{"guidance", 2};
        Input<int> rad{"radius"};
        Input<float> epsilon{"epsilon"};
        Output<Halide::Buffer<uint8_t>> output{"output", 2};

        void generate() {
            const int k_size = 5;
            input_bounded = Halide::BoundaryConditions::repeat_edge(input);
            guidance_bounded = Halide::BoundaryConditions::repeat_edge(guidance);

            mean_input(x, y) = box_blur(input_bounded, k_size, "mean_input")(x, y);
            mean_guidance(x, y) = box_blur(guidance_bounded, k_size, "mean_guidance")(x, y);
            mean_ii(x, y) = box_blur<int, int>(lambda(x, y, u16(input_bounded(x, y)) * input_bounded(x, y)), k_size, "mean_ii")(x, y);
            mean_ig(x, y) = box_blur<int, int>(lambda(x, y, u16(input_bounded(x, y)) * guidance_bounded(x, y)), k_size, "mean_ig")(x, y);

            var_i(x, y) = i32(mean_ii(x, y)) - i32(mean_input(x, y)) * mean_input(x, y);
            cov_ip(x, y) = i32(mean_ig(x, y)) - i32(mean_input(x, y)) * mean_guidance(x, y);
            a(x, y) = cov_ip(x, y) / (f32(var_i(x, y)) + epsilon);
            b(x, y) = mean_guidance(x, y) - a(x, y) * mean_input(x, y);

            mean_a(x, y) = box_blur<float, float>(a, k_size, "mean_a")(x, y);
            mean_b(x, y) = box_blur<float, float>(b, k_size, "mean_a")(x, y);

            output(x, y) = u8_sat(mean_a(x, y) * input_bounded(x, y) + mean_b(x, y));
        }

        void schedule() {
            if (auto_schedule) {

            } else {
                const auto tile_w = 64; // manually tuned on i7-7600U
                const auto tile_h = 16;
                Var xo("xo"), yo("yo"), xi("xi"), yi("yi"), tile("tile");

                mean_input.compute_root();
                mean_guidance.compute_root();
                mean_ig.compute_root();
                mean_ii.compute_root();
                cov_ip.compute_root();
                var_i.compute_root();
                a.compute_root();
                b.compute_root();
                mean_a.compute_root();
                mean_b.compute_root();
            }
        }
    private:
        template<class TResult=unsigned char, class TSum=signed short>
        Func box_blur(const Halide::Func& in, const int k_size = 5, const std::string& name = "box_blur") const {

            // TODO: implement O(1) filtering
            assert(k_size > 0 && k_size % 2 == 1);
            Halide::RDom r(-(k_size / 2), k_size, -(k_size / 2), k_size);

            Func filtered(name);
            const Halide::Expr box_sum = sum(Halide::cast<TSum>(in(x + r.x, y + r.y)), "blur_sum");
            filtered(x, y) = Halide::saturating_cast<TResult>(fast_integer_divide(box_sum, k_size * k_size));
            return filtered;
        }

    private:
        Func input_bounded = Func{"input_bounded"};
        Func guidance_bounded = Func{"guidance_bounded"};
        Func mean_input = Func{"mean_input"};
        Func mean_guidance = Func{"mean_guidance"};
        Func mean_ig = Func{"mean_ig"};
        Func mean_ii = Func{"mean_ii"};
        Func cov_ip = Func{"cov_ip"};
        Func var_i = Func{"var_i"};
        Func a = Func{"a"};
        Func b = Func{"b"};
        Func mean_a = Func{"mean_a"};
        Func mean_b = Func{"mean_b"};
        Var x = {"x"};
        Var y = {"y"};
        Var c = {"c"};
    };

}  // namespace

HALIDE_REGISTER_GENERATOR(GuidedPipeline, guided_image_filter)
