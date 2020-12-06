#include <Halide.h>

#include <type_traits>
#include <utility>

using namespace Halide;
using namespace Halide::ConciseCasts;

namespace {

    class ColorGuidedPipeline : public Halide::Generator<ColorGuidedPipeline> {

        Func& calc_channel_corr(Func& dest, const Func& f1, const Func& f2, const int channels) {
            assert(f1.dimensions() == f2.dimensions());
            static Buffer<int> i = []() {
                Buffer<int> buf(6, "channel_lookup_i");
                buf(0) = 0; buf(1) = 0; buf(2) = 0; buf(3) = 1; buf(4) = 1; buf(5) = 2;
                return buf;
            }();
            static Buffer<int> j = [](){
                Buffer<int> buf(6, "channel_lookup_j");
                buf(0) = 0; buf(1) = 1; buf(2) = 2; buf(3) = 1; buf(4) = 2; buf(5) = 2;
                return buf;
            }();
            dest(x, y, c) = f1(x, y, clamp(i(c), 0, 2)) * f2(x, y, clamp(j(c), 0, 2));
            dest.compute_root().bound(c, 0, 6);
            return dest;
        }

    public:
        Input<Halide::Buffer<uint8_t>> input_p{"input_p", 2};  // input_p image single channel
        Input<Halide::Buffer<uint8_t>> guidance_i{"guidance_i", 3};  // 3 channels

        Input<int> rad{"radius"};
        Input<float> epsilon{"epsilon"};
        Output<Halide::Buffer<uint8_t>> output{"output", 2};

        const bool FastGF = true;

        void generate() {
            assert(guidance_i.channels() == 3);

            const int k_size = 20;

            if (FastGF) {
                downsample_2x(p_bounded, Halide::BoundaryConditions::repeat_edge(input_p));
                downsample_2x(i_bounded, Halide::BoundaryConditions::repeat_edge(guidance_i));
            } else {
                p_bounded(x, y) = f32(Halide::BoundaryConditions::repeat_edge(input_p)(x, y));
                i_bounded(x, y, c) = f32(Halide::BoundaryConditions::repeat_edge(guidance_i)(x, y, c));
            }
            // Algorithm 1 Guided Filter [1]
            // Step 1
            box_blur<float, float>(mean_p, p_bounded, k_size);
            box_blur<float, float>(mean_i, i_bounded, k_size);

            calc_channel_corr(channel_corr_i, i_bounded, i_bounded, 3);
            box_blur<float, float>(corr_i, channel_corr_i, k_size);

            box_blur<float, float>(corr_ip, lambda(x, y, c, f32(p_bounded(x, y)) * i_bounded(x, y, c)), k_size);

            // Step 2
            calc_channel_corr(channel_corr_mean_i, mean_i, mean_i, 3);
            var_i(x, y, c) = corr_i(x, y, c) - channel_corr_mean_i(x, y, c);

            cov_ip(x, y, c) = f32(corr_ip(x, y, c)) - f32(mean_p(x, y)) * mean_i(x, y, c);

            // Step 3
            // a
            sym_mat_inv_3x3(inv_var_i, var_i, epsilon);
            sym_mat_mul(a, inv_var_i, cov_ip);
            // b
            b(x, y) = f32(mean_p(x, y)) - a(x, y, 0) * mean_i(x, y, 0) - a(x, y, 1) * mean_i(x, y, 1) - a(x, y, 2) * mean_i(x, y, 2);

            // Step 4
            box_blur<float, float>(mean_a, a, k_size);
            box_blur<float, float>(mean_b, b, k_size);

            if (FastGF) {
                upsample_2x(umean_a, mean_a);
                upsample_2x(umean_b, mean_b);
            } else {
                umean_a = mean_a;
                umean_b = mean_b;
            }

            // Step 5
            output(x, y) = u8_sat(
                umean_a(x, y, 0) * f32(guidance_i(x, y, 0)) +
                umean_a(x, y, 1) * f32(guidance_i(x, y, 1)) +
                umean_a(x, y, 2) * f32(guidance_i(x, y, 2)) +
                umean_b(x, y)
            );
        }

        void schedule() {
            if (auto_schedule) {

            } else {
                const auto tile_w = 64; // manually tuned on i7-7600U
                const auto tile_h = 16;
                Var xo("xo"), yo("yo"), xi("xi"), yi("yi"), tile("tile");

                p_bounded.compute_root();
                i_bounded.compute_root();

                mean_p.compute_root();
                mean_i.compute_root().bound(c, 0, 3);
                corr_ip.compute_root();
                mean_ii.compute_root();
                corr_i.compute_root();
                cov_ip.compute_root();
                var_i.compute_root();
                inv_var_i.compute_root();

                a.compute_root().tile(x, y, xo, yo, xi, yi, tile_w, tile_h);
                b.compute_root().tile(x, y, xo, yo, xi, yi, tile_w, tile_h);
                //store_at(output, tile).compute_at(output, tile).
                mean_a.compute_root().vectorize(x, tile_w);
                mean_b.compute_root().vectorize(x, tile_w);
                if (FastGF) {
                    umean_a.compute_root().vectorize(x, tile_w);
                    umean_b.compute_root().vectorize(x, tile_w);
                }
                output.tile(x, y, xo, yo, xi, yi, tile_w, tile_h).fuse(xo, yo, tile).vectorize(xi, tile_w).unroll(yi);

                const bool make_dumps = false;
                if (make_dumps) {
                    const auto functions_to_dump = {&channel_corr_i, &corr_i, &channel_corr_mean_i, &p_bounded, &i_bounded,
                                                    &mean_p, &mean_i, &corr_ip, &mean_ii, &cov_ip, &var_i,
                                                    &inv_var_i, &a, &b, &mean_a, &mean_b};
                    for (Func *func : functions_to_dump) {
                        const std::string dumpFileName = "func_dumps/" + func->name();
                        if (func->outputs() > 1) {
                            std::cerr << "Can't dump " << func->name() << std::endl;
                        } else {
                            std::cerr << "Making function " << func->name() << " dump: " << dumpFileName << std::endl;
                            func->debug_to_file(dumpFileName);
                        }
                    }
                }
            }
        }
    private:
        void sym_mat_mul(const Halide::Func& dest, const Halide::Func& m, const Halide::Func& v) {
            Expr ma = f32(m(x, y, 0));
            Expr mb = f32(m(x, y, 1));
            Expr mc = f32(m(x, y, 2));
            Expr md = f32(m(x, y, 3));
            Expr me = f32(m(x, y, 4));
            Expr mf = f32(m(x, y, 5));

            Expr vx = v(x, y, 0);
            Expr vy = v(x, y, 1);
            Expr vz = v(x, y, 2);

            dest(x, y, c) = select(
                c == 0, ma * vx + mb * vy + mc * vz,
                c == 1, mb * vx + md * vy + me * vz,
                c == 2, mc * vx + me * vy + mf * vz,
                0
            );
        }

        void sym_mat_inv_3x3(Halide::Func& dest, const Halide::Func& m, Expr epsilon) {
            /* Derivation:
             * >>> from sympy import *
             * >>> a, b, c, d, e, f = symbols('a b c d e f')
             * >>> m = Matrix([[a, b, c], [b, d, e], [c, e, f]])
             * >>> print(m.det())
             * >>> print(m.inv())
             */

            Expr ma = f32(m(x, y, 0)) + epsilon;
            Expr mb = f32(m(x, y, 1));
            Expr mc = f32(m(x, y, 2));
            Expr md = f32(m(x, y, 3)) + epsilon;
            Expr me = f32(m(x, y, 4));
            Expr mf = f32(m(x, y, 5)) + epsilon;

            Func det(m.name() + "_det");
            det(x, y) = (ma * md * mf - ma * me * me - mb * mb * mf + 2 * mb * mc * me - mc * mc * md);

            dest(x, y, c) = select(
                 c == 0, (md * mf - me * me) / det(x, y),
                 c == 1, -(mb * mf - mc * me) / det(x, y),
                 c == 2,  (mb * me - mc * md) / det(x, y),
                 c == 3, (ma * mf - mc * mc) / det(x, y),
                 c == 4, -(ma * me - mb * mc) / det(x, y),
                 c == 5, (ma * md - mb * mb) / det(x, y),
                 0
            );
        }

        template<class TResult=unsigned char, class TSum=signed short>
        void box_blur(Halide::Func& dest, const Halide::Func& in, const int k_radius = 5) const {
            // TODO: implement O(1) filtering
            assert(k_size > 0 && k_size % 2 == 1);
            const auto normalization = 2 * k_radius + 1;
            Halide::RDom rx(-k_radius, normalization);
            Halide::RDom ry(-k_radius, normalization);

            Func blur_x(in.name() + "_blur_x");

            const auto sum_cast = [&](const Expr& e, const std::string& sfx = "") {
                if constexpr (std::is_integral_v<TSum>) {
                    return fast_integer_divide(sum(Halide::saturating_cast<TSum>(e), dest.name() + "_sum" + sfx), normalization);
                } else {
                    return sum(Halide::cast<TSum>(e), dest.name() + "_sum" + sfx) / normalization;
                }
            };

            if (in.dimensions() == 2) {
                blur_x(x, y) = sum_cast(in(x + rx.x, y), "_x");
                dest(x, y) = Halide::saturating_cast<TResult>(sum_cast(blur_x(x, y + ry.x)));
            } else if (in.dimensions() == 3) {
                blur_x(x, y, c) = sum_cast(in(x + rx.x, y, c), "_x");
                dest(x, y, c) = Halide::saturating_cast<TResult>(sum_cast(blur_x(x, y + ry.x, c)));
            } else {
                 throw std::runtime_error("Wrong number of dimensions in box_blur");
            }
            //blur_x.compute_root(); //at(dest, x); //at(filtered, x);
        }


        // Downsample with a 1 3 3 1 filter
        void downsample_2x(Func& dest, const Func& in) {
            Func downx(in.name() + "_downx");
            if (in.dimensions() == 2) {
                downx(x, y) = f32(f32(in(2 * x - 1, y)) + 3.0f * (f32(in(2 * x, y)) + in(2 * x + 1, y)) + f32(in(2 * x + 2, y))) / 8.0f;
                dest(x, y) = (downx(x, 2 * y - 1) + 3.0f * (downx(x, 2 * y) + downx(x, 2 * y + 1)) + downx(x, 2 * y + 2)) / 8.0f;
            } else if (in.dimensions() == 3) {
                downx(x, y, c) = f32(f32(in(2 * x - 1, y, c)) + 3.0f * (f32(in(2 * x, y, c)) + in(2 * x + 1, y, c)) + f32(in(2 * x + 2, y, c))) / 8.0f;
                dest(x, y, c) = (downx(x, 2 * y - 1, c) + 3.0f * (downx(x, 2 * y, c) + downx(x, 2 * y + 1, c)) + downx(x, 2 * y + 2, c)) / 8.0f;
            }
        }

        // Upsample using bilinear interpolation
        void upsample_2x(Func& dest, const Func& f) {
            using Halide::_;
            Func upx(f.name() + "_upx");
            if (f.dimensions() == 2) {
                upx(x, y) = 0.25f * f((x / 2) - 1 + 2 * (x % 2), y) + 0.75f * f(x / 2, y);
                dest(x, y) = 0.25f * upx(x, (y / 2) - 1 + 2 * (y % 2)) + 0.75f * upx(x, y / 2);
            } else if (f.dimensions() == 3) {
                upx(x, y, c) = 0.25f * f((x / 2) - 1 + 2 * (x % 2), y, c) + 0.75f * f(x / 2, y, c);
                dest(x, y, c) = 0.25f * upx(x, (y / 2) - 1 + 2 * (y % 2), c) + 0.75f * upx(x, y / 2, c);
            }
        }

    private:
        Func p_bounded = Func{"p_bounded"};
        Func i_bounded = Func{"i_bounded"};
        Func mean_p = Func{"mean_p"};
        Func mean_i = Func{"mean_i"};
        Func corr_ip = Func{"corr_ip"};
        Func mean_ii = Func{"mean_ii"};
        Func channel_corr_i = Func{"channel_corr_i"};
        Func corr_i = Func{"corr_i"};
        Func channel_corr_mean_i = Func{"channel_corr_mean_i"};

        Func cov_ip = Func{"cov_ip"};
        Func var_i = Func{"var_i"};
        Func inv_var_i = Func{"inv_var_i"};
        Func a = Func{"a"};
        Func b = Func{"b"};
        Func mean_a = Func{"mean_a"};
        Func mean_b = Func{"mean_b"};
        Func umean_a = Func{"umean_a"};;
        Func umean_b = Func{"umean_b"};;
        Var x = {"x"};
        Var y = {"y"};
        Var c = {"c"};
    };

}  // namespace

HALIDE_REGISTER_GENERATOR(ColorGuidedPipeline, color_guided_image_filter)
