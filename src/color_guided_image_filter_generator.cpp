#include <Halide.h>

#include <type_traits>
#include <utility>

using namespace Halide;
using namespace Halide::ConciseCasts;

#ifdef USE_COLOR_GUIDED_FILTER_AUTOSCHEDULE
#include "color_guided_image_filter.schedule.h"
namespace {
    const bool auto_schedule_available = true;
}
#else
namespace {
    const bool auto_schedule_available = false;
    inline void apply_schedule_color_guided_image_filter(Halide::Pipeline pipeline, Halide::Target target) {}
}
#endif

namespace {

    struct SeparableFunc {
        explicit SeparableFunc(const std::string& name)
            : result_x(name + "_x")
            , result(name)
        {}

        template<typename... Args>
        typename std::enable_if<Halide::Internal::all_are_convertible<Expr, Args...>::value, FuncRef>::type
        operator()(Args &&... args) const {
            std::vector<Expr> collected_args{std::forward<Args>(args)...};
            return result(collected_args);
        }

        operator const Func&() const { return result; }

        [[nodiscard]] const std::string& name() const { return result.name(); }

        Func& compute_root(const Var& x, const Var& y, const int tile_w, const int tile_h, const int vector_width) {
            result_x.compute_at(result, yi).store_at(result, yo).vectorize(x, vector_width);;
            return result.compute_root().tile(x, y, xo, yo, xi, yi, tile_w, tile_h).vectorize(xi, vector_width);
        }

        Func result_x;
        Func result;
        Var xo = Var("xo");
        Var yo = Var("yo");
        Var xi = Var("xi");
        Var yi = Var("yi");
    };

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
            dest(x, y, c) = i64(f1(x, y, unsafe_promise_clamped(i(c), 0, 2))) * f2(x, y, unsafe_promise_clamped(j(c), 0, 2));
            return dest;
        }

    public:
        Input<Halide::Buffer<uint8_t>> input_p{"input_p", 2};  // input_p image single channel
        Input<Halide::Buffer<uint8_t>> guidance_i{"guidance_i", 3};  // 3 channels

        Input<int> rad{"radius"};
        Input<int> epsilon{"epsilon"};
        Output<Halide::Buffer<uint8_t>> output{"output", 2};

        const bool FastGF = true;

        void generate() {
            const int k_size = 20;

            if (FastGF) {
                downsample_2x(p_bounded, Halide::BoundaryConditions::repeat_edge(input_p));
                downsample_2x(i_bounded, Halide::BoundaryConditions::repeat_edge(guidance_i));
            } else {
                p_bounded(x, y) = Halide::BoundaryConditions::repeat_edge(input_p)(x, y);
                i_bounded(x, y, c) = Halide::BoundaryConditions::repeat_edge(guidance_i)(x, y, c);
            }
            // Algorithm 1 Guided Filter [1]
            // Step 1
            const auto blur_normalization = (2 * k_size + 1) * (2 * k_size + 1);
            box_blur_int(mean_p, p_bounded, k_size);
            box_blur_int(mean_i, i_bounded, k_size);

            calc_channel_corr(channel_corr_i, i_bounded, i_bounded, 3);
            box_blur_int(corr_i, channel_corr_i, k_size);

            box_blur_int(corr_ip, lambda(x, y, c, u32(p_bounded(x, y)) * i_bounded(x, y, c)), k_size);

            // Step 2
            calc_channel_corr(channel_corr_mean_i, mean_i, mean_i, 3);
            var_i(x, y, c) = (i64(corr_i(x, y, c)) - i64(channel_corr_mean_i(x, y, c)) / blur_normalization) / blur_normalization;
            cov_ip(x, y, c) = (i64(corr_ip(x, y, c)) - i64(mean_p(x, y)) * mean_i(x, y, c) / blur_normalization) / blur_normalization;

            // Step 3
            // a
            Expr a_multiplier = 65536;
            sym_mat_inv_3x3(inv_var_i, var_i, epsilon, a_multiplier);
            sym_mat_mul(a, inv_var_i, cov_ip);

            // b
            Expr a_mul_mean_i = (a(x, y, 0) * mean_i(x, y, 0) + a(x, y, 1) * mean_i(x, y, 1) + a(x, y, 2) * mean_i(x, y, 2)) / a_multiplier;
            b(x, y) = (mean_p(x, y) - a_mul_mean_i) / blur_normalization;

            // Step 4
            box_blur_int(mean_a, a, k_size, true);
            box_blur_int(mean_b, b, k_size, true);

            if (FastGF) {
                upsample_2x(umean_a, mean_a);
                upsample_2x(umean_b, mean_b);
            } else {
                umean_a = mean_a;
                umean_b = mean_b;
            }

            // Step 5
            output(x, y) = u8_sat(
                (umean_a(x, y, 0) * guidance_i(x, y, 0) +
                umean_a(x, y, 1) * guidance_i(x, y, 1) +
                umean_a(x, y, 2) * guidance_i(x, y, 2)) / a_multiplier +
                umean_b(x, y)
            );
        }

        void schedule() {
            if (auto_schedule) {
                input_p.set_estimates({{0, 2048}, {0, 1024}});
                guidance_i.set_estimates({{0, 2048}, {0, 1024}, {0, 3}});
                rad.set_estimate(30);
                epsilon.set_estimate(0.01f);
                output.set_estimates({{0, 2048}, {0, 1024}});
            } else if (auto_schedule_available) {
                std::cerr << "Use automatically generated schedule." << std::endl;
                apply_schedule_color_guided_image_filter(get_pipeline(), get_target());
            } else {
                std::cerr << "Use manually written schedule." << std::endl;
                const auto tile_w = 64; // manually tuned on i7-7600U
                const auto tile_h = 64;
                const auto vector_width = 32;
                Var xo("xo"), yo("yo"), xi("xi"), yi("yi"), tile("tile");

                if (FastGF) {
                    p_bounded.compute_root(x, y, tile_w, tile_h, vector_width);
                    i_bounded.compute_root(x, y, tile_w, tile_h, vector_width);
                }

                mean_p.compute_root(x, y, tile_w, tile_h, vector_width);
                mean_i.compute_root(x, y, tile_w, tile_h, vector_width);
                corr_ip.compute_root(x, y, tile_w, tile_h, vector_width);
                corr_i.compute_root(x, y, tile_w, tile_h, vector_width);

                var_i.compute_at(a, xi).vectorize(x, vector_width);
                inv_var_i.compute_at(a, xi).vectorize(x, vector_width);
                cov_ip.compute_at(a, xi).vectorize(x, vector_width);
                a.compute_root().tile(x, y, xo, yo, xi, yi, tile_w, tile_h).vectorize(xi, vector_width);

                b.store_at(mean_b, mean_b.yo).compute_at(mean_b, mean_b.yi).vectorize(x, vector_width);

                mean_a.compute_root(x, y, tile_w, tile_h, vector_width);
                mean_b.compute_root(x, y, tile_w, tile_h, vector_width);

                if (FastGF) {
                    umean_a.compute_root(x, y, tile_w, tile_h, vector_width);
                    umean_b.compute_root(x, y, tile_w, tile_h, vector_width);
                }
                output.tile(x, y, xo, yo, xi, yi, tile_w, tile_h).fuse(xo, yo, tile).vectorize(xi, vector_width); //.unroll(yi);

                const bool make_dumps = true;
                if (make_dumps) {
                    const auto functions_to_dump = {&channel_corr_i, &corr_i.result, &channel_corr_mean_i,
                                                    &p_bounded.result, &i_bounded.result, &mean_p.result,
                                                    &mean_i.result, &corr_ip.result, &cov_ip, &var_i,
                                                    &inv_var_i, &a, &b, &mean_a.result, &mean_b.result};
                    for (Func *func : functions_to_dump) {
                        func->compute_root().store_root();
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
            Expr ma = m(x, y, 0);
            Expr mb = m(x, y, 1);
            Expr mc = m(x, y, 2);
            Expr md = m(x, y, 3);
            Expr me = m(x, y, 4);
            Expr mf = m(x, y, 5);

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

        void sym_mat_inv_3x3(Halide::Func& dest, const Halide::Func& m, Expr epsilon, Expr a_multiplier) {
            /* Derivation:
             * >>> from sympy import *
             * >>> a, b, c, d, e, f = symbols('a b c d e f')
             * >>> m = Matrix([[a, b, c], [b, d, e], [c, e, f]])
             * >>> print(m.det())
             * >>> print(m.inv())
             */

            Expr ma = m(x, y, 0) + epsilon;
            Expr mb = m(x, y, 1);
            Expr mc = m(x, y, 2);
            Expr md = m(x, y, 3) + epsilon;
            Expr me = m(x, y, 4);
            Expr mf = m(x, y, 5) + epsilon;

            Func det(m.name() + "_det");
            det(x, y) = (ma * md * mf - ma * me * me - mb * mb * mf + 2 * mb * mc * me - mc * mc * md);

            Expr multiplier = 65535;
            dest(x, y, c) = i64(select(
                 c == 0, multiplier *  (md * mf - me * me) / det(x, y),
                 c == 1, multiplier * -(mb * mf - mc * me) / det(x, y),
                 c == 2, multiplier *  (mb * me - mc * md) / det(x, y),
                 c == 3, multiplier *  (ma * mf - mc * mc) / det(x, y),
                 c == 4, multiplier * -(ma * me - mb * mc) / det(x, y),
                 c == 5, multiplier *  (ma * md - mb * mb) / det(x, y),
                 0
            ));
        }


        void box_blur_int(SeparableFunc& dest, const Halide::Func& in, const int k_radius = 5, bool normalize = false) const {
            const auto k_size = 2 * k_radius + 1;
            Halide::RDom rx(-k_radius, k_size);
            Halide::RDom ry(-k_radius, k_size);

            Func& blur_x = dest.result_x;

            const auto normalization = (normalize ? (k_size * k_size) : 1);
            if (in.dimensions() == 2) {
                blur_x(x, y) = sum(i32(in(x + rx.x, y)));
                dest(x, y) = sum(i32(blur_x(x, y + ry.x))) / normalization;
            } else if (in.dimensions() == 3) {
                blur_x(x, y, c) = sum(i32(in(x + rx.x, y, c)));
                dest(x, y, c) = sum(i32(blur_x(x, y + ry.x, c))) / normalization;
            } else {
                throw std::runtime_error("Wrong number of dimensions in box_blur");
            }
        }


        template<class TResult=unsigned char, class TSum=signed short>
        void box_blur(SeparableFunc& dest, const Halide::Func& in, const int k_radius = 5) const {
            const int normalization = 2 * k_radius + 1;
            Halide::RDom rx(-k_radius, normalization);
            Halide::RDom ry(-k_radius, normalization);

            Func& blur_x = dest.result_x;

            const auto sum_cast = [&](const Expr& e, const std::string& sfx = "") -> Expr {
                if (std::numeric_limits<TSum>::is_integer) {
                    std::cerr << dest.name() << " uses integer sums." << std::endl;
                    return fast_integer_divide(sum(Halide::cast<TSum>(e)), Halide::cast<TSum>(normalization));
                } else {
                    return sum(Halide::cast<TSum>(e), dest.name() + "_sum" + sfx) / normalization;
                }
            };

            if (in.dimensions() == 2) {
                blur_x(x, y) = Halide::cast<TResult>(sum_cast(in(x + rx.x, y), "_x"));
                dest(x, y) = Halide::cast<TResult>(sum_cast(blur_x(x, y + ry.x)));
            } else if (in.dimensions() == 3) {
                blur_x(x, y, c) = Halide::cast<TResult>(sum_cast(in(x + rx.x, y, c), "_x"));
                dest(x, y, c) = Halide::cast<TResult>(sum_cast(blur_x(x, y + ry.x, c)));
            } else {
                 throw std::runtime_error("Wrong number of dimensions in box_blur");
            }
        }


        // Downsample with a 1 3 3 1 filter
        void downsample_2x(SeparableFunc& dest, const Func& in) {
            Func& downx = dest.result_x;
            const auto sum_1331 = [](const Expr& v0, const Expr& v1, const Expr& v2, const Expr& v3) {
                return fast_integer_divide(u16(v0) + 3 * (u16(v1) + v2) + v3, 8);
            };
            if (in.dimensions() == 2) {
                downx(x, y) = sum_1331(in(2 * x - 1, y), in(2 * x, y), in(2 * x + 1, y), in(2 * x + 2, y));
                dest(x, y) = sum_1331(downx(x, 2 * y - 1), downx(x, 2 * y), downx(x, 2 * y + 1), downx(x, 2 * y + 2));
            } else if (in.dimensions() == 3) {
                downx(x, y, c) = sum_1331(in(2 * x - 1, y, c), in(2 * x, y, c), in(2 * x + 1, y, c), in(2 * x + 2, y, c));
                dest(x, y, c) = sum_1331(downx(x, 2 * y - 1, c), downx(x, 2 * y, c), downx(x, 2 * y + 1, c), downx(x, 2 * y + 2, c));
            }
        }

        // Upsample using bilinear interpolation
        void upsample_2x(SeparableFunc& dest, const Func& f) {
            Func& upx = dest.result_x;
            if (f.dimensions() == 2) {
                upx(x, y) = f((x / 2) - 1 + 2 * (x % 2), y) + 3 * f(x / 2, y);
                dest(x, y) = (upx(x, (y / 2) - 1 + 2 * (y % 2)) + 3 * upx(x, y / 2)) / 16.f;
            } else if (f.dimensions() == 3) {
                upx(x, y, c) = f((x / 2) - 1 + 2 * (x % 2), y, c) +  3 * f(x / 2, y, c);
                dest(x, y, c) = (upx(x, (y / 2) - 1 + 2 * (y % 2), c) + 3 * upx(x, y / 2, c)) / 16.f;
            }
        }

    private:
        SeparableFunc p_bounded = SeparableFunc{"p_bounded"};
        SeparableFunc i_bounded = SeparableFunc{"i_bounded"};

        SeparableFunc mean_p = SeparableFunc{"mean_p"};
        SeparableFunc mean_i = SeparableFunc{"mean_i"};
        SeparableFunc corr_ip = SeparableFunc{"corr_ip"};
        Func channel_corr_i = Func{"channel_corr_i"};
        SeparableFunc corr_i = SeparableFunc{"corr_i"};
        Func channel_corr_mean_i = Func{"channel_corr_mean_i"};

        Func cov_ip = Func{"cov_ip"};
        Func var_i = Func{"var_i"};
        Func inv_var_i = Func{"inv_var_i"};
        Func a = Func{"a"};
        Func b = Func{"b"};
        SeparableFunc mean_a = SeparableFunc{"mean_a"};
        SeparableFunc mean_b = SeparableFunc{"mean_b"};
        SeparableFunc umean_a = SeparableFunc{"umean_a"};;
        SeparableFunc umean_b = SeparableFunc{"umean_b"};;
        Var x = {"x"};
        Var y = {"y"};
        Var c = {"c"};
    };

}  // namespace

HALIDE_REGISTER_GENERATOR(ColorGuidedPipeline, color_guided_image_filter)
