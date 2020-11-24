#include <Halide.h>

#include <utility>

using namespace Halide;
using namespace Halide::ConciseCasts;

namespace {
    //template<int InputChannelCount>
    class ColorGuidedPipeline : public Halide::Generator<ColorGuidedPipeline> {
    public:
        Input<Halide::Buffer<uint8_t>> input{"input", 2};  // input image single channel
        Input<Halide::Buffer<uint8_t>> guidance{"guidance", 3};  // 3 channels
        Input<int> rad{"radius"};
        Input<float> epsilon{"epsilon"};
        Output<Halide::Buffer<uint8_t>> output{"output", 2};

        void generate() {
            assert(guidance.channels() == 3);

            const int k_size = 20;

            input_bounded(x, y) = Halide::BoundaryConditions::repeat_edge(input)(x, y);
            guidance_bounded(x, y, c) = Halide::BoundaryConditions::repeat_edge(guidance)(x, y, c);

            mean_input(x, y) = box_blur<float, float>(input_bounded, k_size)(x, y);

            mean_guidance(x, y, c) = box_blur<float, float>(guidance_bounded, k_size)(x, y, c);

            mean_ig(x, y, c) = box_blur<float, float>(lambda(x, y, c, f32(input_bounded(x, y)) * guidance_bounded(x, y, c)), k_size)(x, y, c);

            var_i(x, y) = Halide::Tuple(
                box_blur<float, float>(lambda(x, y, f32(guidance_bounded(x, y, 0)) * guidance_bounded(x, y, 0)), k_size)(x, y) - f32(mean_guidance(x, y, 0)) * mean_guidance(x, y, 0) + epsilon,
                box_blur<float, float>(lambda(x, y, f32(guidance_bounded(x, y, 0)) * guidance_bounded(x, y, 1)), k_size)(x, y) - f32(mean_guidance(x, y, 0)) * mean_guidance(x, y, 1),
                box_blur<float, float>(lambda(x, y, f32(guidance_bounded(x, y, 0)) * guidance_bounded(x, y, 2)), k_size)(x, y) - f32(mean_guidance(x, y, 0)) * mean_guidance(x, y, 2),
                box_blur<float, float>(lambda(x, y, f32(guidance_bounded(x, y, 1)) * guidance_bounded(x, y, 1)), k_size)(x, y) - f32(mean_guidance(x, y, 1)) * mean_guidance(x, y, 1) + epsilon,
                box_blur<float, float>(lambda(x, y, f32(guidance_bounded(x, y, 1)) * guidance_bounded(x, y, 2)), k_size)(x, y) - f32(mean_guidance(x, y, 1)) * mean_guidance(x, y, 2),
                box_blur<float, float>(lambda(x, y, f32(guidance_bounded(x, y, 2)) * guidance_bounded(x, y, 2)), k_size)(x, y) - f32(mean_guidance(x, y, 2)) * mean_guidance(x, y, 2) + epsilon
            );

            inv_var_i(x, y) = sym_mat_inv_3x3(var_i)(x, y);

            cov_ip(x, y, c) = f32(mean_ig(x, y, c)) - f32(mean_input(x, y)) * mean_guidance(x, y, c);

            a(x, y, c) = sym_mat_mul(inv_var_i, cov_ip)(x, y, c);
            b(x, y) = f32(mean_input(x, y)) - a(x, y, 0) * mean_guidance(x, y, 0) - a(x, y, 1) * mean_guidance(x, y, 1) - a(x, y, 2) * mean_guidance(x, y, 2);

            mean_a(x, y, c) = box_blur<float, float>(a, k_size)(x, y, c);
            mean_b(x, y) = box_blur<float, float>(b, k_size)(x, y);

            output(x, y) = u8_sat(
                mean_a(x, y, 0) * guidance_bounded(x, y, 0) +
                mean_a(x, y, 1) * guidance_bounded(x, y, 1) +
                mean_a(x, y, 2) * guidance_bounded(x, y, 2) +
                mean_b(x, y)
            );
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
                inv_var_i.compute_root();
                a.compute_root();
                b.compute_root();
                mean_a.compute_root();
                mean_b.compute_root();
            }
        }
    private:
        Func sym_mat_mul(const Halide::Func& m, const Halide::Func& v) {
            Expr ma = m(x, y)[0];
            Expr mb = m(x, y)[1];
            Expr mc = m(x, y)[2];
            Expr md = m(x, y)[3];
            Expr me = m(x, y)[4];
            Expr mf = m(x, y)[5];

            Expr vx = v(x, y, 0);
            Expr vy = v(x, y, 1);
            Expr vz = v(x, y, 2);

            Func result(m.name() + "_" + v.name() + "_sym_mat_mul");
            result(x, y, c) = select(
                c == 0, ma * vx + mb * vy + mc * vz,
                c == 1, mb * vx + md * vy + me * vz,
                c == 2, mc * vx + me * vy + mf * vz,
                0
            );
            return result;
        }

        Func sym_mat_inv_3x3(const Halide::Func& m) {
            /* Derivation:
             * >>> from sympy import *
             * >>> a, b, c, d, e, f = symbols('a b c d e f')
             * >>> m = Matrix([[a, b, c], [b, d, e], [c, e, f]])
             * >>> print(m.det())
             * >>> print(m.inv())
             */

            Expr ma = f32(m(x, y)[0]);
            Expr mb = f32(m(x, y)[1]);
            Expr mc = f32(m(x, y)[2]);
            Expr md = f32(m(x, y)[3]);
            Expr me = f32(m(x, y)[4]);
            Expr mf = f32(m(x, y)[5]);

            Func det(m.name() + "_det");
            det(x, y) = ma * md * mf - ma * me * me - mb * mb * mf + 2 * mb * mc * me - mc * mc * md;
            det.compute_root();

            Func result(m.name() + "_inv");
            result(x, y) = Tuple(
                 (md * mf - me * me) / det(x, y),
                 -(mb * mf - mc * me) / det(x, y),
                 (mb * me - mc * md) / det(x, y),
                 (ma * mf - mc * mc) / det(x, y),
                 -(ma * me - mb * mc) / det(x, y),
                 (ma * md - mb * mb) / det(x, y)
            );
            return result;
        }

        template<class TResult=unsigned char, class TSum=signed short>
        Func box_blur(const Halide::Func& in, const int k_radius = 5) const {
            // TODO: implement O(1) filtering
            assert(k_size > 0 && k_size % 2 == 1);
            const auto normalization = 2 * k_radius + 1;
            Halide::RDom rx(-k_radius, normalization);
            Halide::RDom ry(-k_radius, normalization);

            Func blur_x(in.name() + "_blur_x");
            Func filtered(in.name() + "_box_blur");

            const auto sum_cast = [&](const Expr& e, const std::string& sfx = ""){
                return sum(Halide::cast<TSum>(e), filtered.name() + "_sum" + sfx) / normalization;
            };

            if (in.dimensions() == 2) {
                blur_x(x, y) = sum_cast(in(x + rx.x, y), "_x");
                filtered(x, y) = Halide::cast<TResult>(sum_cast(blur_x(x, y + ry.x)));
            } else if (in.dimensions() == 3) {
                blur_x(x, y, c) = sum_cast(in(x + rx.x, y, c), "_x");
                filtered(x, y, c) = Halide::cast<TResult>(sum_cast(blur_x(x, y + ry.x, c)));
            } else {
                 throw std::runtime_error("Wrong number of dimensions in box_blur");
            }
            blur_x.compute_root(); //at(filtered, x);
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
        Func inv_var_i = Func{"inv_var_i"};
        Func a = Func{"a"};
        Func b = Func{"b"};
        Func mean_a = Func{"mean_a"};
        Func mean_b = Func{"mean_b"};
        Var x = {"x"};
        Var y = {"y"};
        Var c = {"c"};
        Var c2 = {"c2"};
    };

}  // namespace

HALIDE_REGISTER_GENERATOR(ColorGuidedPipeline, color_guided_image_filter)
