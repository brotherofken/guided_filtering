#ifndef color_guided_image_filter_SCHEDULE_H
#define color_guided_image_filter_SCHEDULE_H

// MACHINE GENERATED -- DO NOT EDIT
// This schedule was automatically generated by Adams2019
// for target=x86-64-linux-avx-avx2-f16c-fma-sse41  // NOLINT
// with machine_params=16,16777216,40

#include "Halide.h"


inline void apply_schedule_color_guided_image_filter(
    ::Halide::Pipeline pipeline,
    ::Halide::Target target
) {
    using ::Halide::Func;
    using ::Halide::MemoryType;
    using ::Halide::RVar;
    using ::Halide::TailStrategy;
    using ::Halide::Var;
    Func output = pipeline.get_func(47);
    Func umean_b = pipeline.get_func(46);
    Func umean_b_x = pipeline.get_func(45);
    Func mean_b = pipeline.get_func(44);
    Func mean_b_sum = pipeline.get_func(43);
    Func mean_b_x = pipeline.get_func(42);
    Func mean_b_sum_x = pipeline.get_func(41);
    Func b = pipeline.get_func(40);
    Func umean_a = pipeline.get_func(39);
    Func umean_a_x = pipeline.get_func(38);
    Func mean_a = pipeline.get_func(37);
    Func mean_a_sum = pipeline.get_func(36);
    Func mean_a_x = pipeline.get_func(35);
    Func mean_a_sum_x = pipeline.get_func(34);
    Func a = pipeline.get_func(33);
    Func inv_var_i = pipeline.get_func(32);
    Func var_i_det = pipeline.get_func(31);
    Func var_i = pipeline.get_func(30);
    Func corr_i = pipeline.get_func(29);
    Func corr_i_sum = pipeline.get_func(28);
    Func corr_i_x = pipeline.get_func(27);
    Func corr_i_sum_x = pipeline.get_func(26);
    Func channel_corr_i = pipeline.get_func(25);
    Func channel_corr_mean_i = pipeline.get_func(24);
    Func cov_ip = pipeline.get_func(23);
    Func mean_p = pipeline.get_func(22);
    Func mean_p_sum = pipeline.get_func(21);
    Func mean_p_x = pipeline.get_func(20);
    Func mean_p_sum_x = pipeline.get_func(19);
    Func mean_i = pipeline.get_func(18);
    Func mean_i_sum = pipeline.get_func(17);
    Func mean_i_x = pipeline.get_func(16);
    Func mean_i_sum_x = pipeline.get_func(15);
    Func corr_ip = pipeline.get_func(14);
    Func corr_ip_sum = pipeline.get_func(13);
    Func corr_ip_x = pipeline.get_func(12);
    Func corr_ip_sum_x = pipeline.get_func(11);
    Func lambda_2 = pipeline.get_func(10);
    Func p_bounded = pipeline.get_func(9);
    Func p_bounded_x = pipeline.get_func(8);
    Func repeat_edge = pipeline.get_func(7);
    Func lambda_0 = pipeline.get_func(6);
    Func i_bounded = pipeline.get_func(4);
    Func i_bounded_x = pipeline.get_func(3);
    Func repeat_edge_1 = pipeline.get_func(2);
    Func lambda_1 = pipeline.get_func(1);
    Var _0(repeat_edge.get_schedule().dims()[0].var);
    Var _1(repeat_edge.get_schedule().dims()[1].var);
    Var _1i("_1i");
    Var _2(repeat_edge_1.get_schedule().dims()[2].var);
    Var c(umean_a.get_schedule().dims()[2].var);
    Var ci("ci");
    Var x(output.get_schedule().dims()[0].var);
    Var xi("xi");
    Var xii("xii");
    Var y(output.get_schedule().dims()[1].var);
    Var yi("yi");
    Var yii("yii");
    Var yiii("yiii");
    RVar r114_x(corr_i_sum_x.update(0).get_schedule().dims()[0].var);
    RVar r119_x(corr_i_sum.update(0).get_schedule().dims()[0].var);
    RVar r161_x(corr_ip_sum_x.update(0).get_schedule().dims()[0].var);
    RVar r166_x(corr_ip_sum.update(0).get_schedule().dims()[0].var);
    RVar r224_x(mean_a_sum_x.update(0).get_schedule().dims()[0].var);
    RVar r229_x(mean_a_sum.update(0).get_schedule().dims()[0].var);
    RVar r268_x(mean_b_sum_x.update(0).get_schedule().dims()[0].var);
    RVar r273_x(mean_b_sum.update(0).get_schedule().dims()[0].var);
    RVar r29_x(mean_p_sum_x.update(0).get_schedule().dims()[0].var);
    RVar r34_x(mean_p_sum.update(0).get_schedule().dims()[0].var);
    RVar r34_xi("r34$xi");
    RVar r67_x(mean_i_sum_x.update(0).get_schedule().dims()[0].var);
    RVar r72_x(mean_i_sum.update(0).get_schedule().dims()[0].var);
    output
        .split(x, x, xi, 128, TailStrategy::ShiftInwards)
        .split(xi, xi, xii, 32, TailStrategy::ShiftInwards)
        .split(y, y, yi, 256, TailStrategy::ShiftInwards)
        .split(yi, yi, yii, 128, TailStrategy::ShiftInwards)
        .split(yii, yii, yiii, 64, TailStrategy::ShiftInwards)
        .vectorize(xii)
        .compute_root()
        .reorder({xii, yiii, yii, yi, xi, y, x})
        .parallel(x);
    umean_b
        .store_in(MemoryType::Stack)
        .split(x, x, xi, 8, TailStrategy::RoundUp)
        .unroll(x)
        .vectorize(xi)
        .compute_at(output, yiii)
        .reorder({xi, x, y});
    umean_b_x
        .store_in(MemoryType::Stack)
        .split(x, x, xi, 8, TailStrategy::RoundUp)
        .vectorize(xi)
        .compute_at(output, yi)
        .reorder({xi, x, y});
    mean_b_sum
        .store_in(MemoryType::Stack)
        .split(y, y, yi, 8, TailStrategy::RoundUp)
        .vectorize(yi)
        .compute_at(output, xi)
        .reorder({yi, y, x})
        .reorder_storage(y, x);
    mean_b_sum.update(0)
        .split(y, y, yi, 8, TailStrategy::RoundUp)
        .vectorize(yi)
        .reorder({yi, r273_x, y, x});
    mean_b_x
        .split(x, x, xi, 65, TailStrategy::RoundUp)
        .split(y, y, yi, 32, TailStrategy::RoundUp)
        .split(xi, xi, xii, 3, TailStrategy::RoundUp)
        .split(yi, yi, yii, 8, TailStrategy::RoundUp)
        .unroll(yi)
        .unroll(xii)
        .vectorize(yii)
        .compute_root()
        .reorder({yii, yi, xii, xi, y, x})
        .parallel(x)
        .reorder_storage(y, x);
    mean_b_sum_x
        .store_in(MemoryType::Stack)
        .split(y, y, yi, 8, TailStrategy::RoundUp)
        .unroll(y)
        .unroll(x)
        .vectorize(yi)
        .compute_at(mean_b_x, xi)
        .reorder({yi, y, x})
        .reorder_storage(y, x);
    mean_b_sum_x.update(0)
        .split(y, y, yi, 8, TailStrategy::RoundUp)
        .unroll(y)
        .unroll(x)
        .vectorize(yi)
        .reorder({yi, y, x, r268_x});
    b
        .store_in(MemoryType::Stack)
        .split(y, y, yi, 8, TailStrategy::RoundUp)
        .vectorize(yi)
        .compute_at(mean_b_x, y)
        .reorder({yi, y, x})
        .reorder_storage(y, x);
    umean_a
        .store_in(MemoryType::Stack)
        .split(x, x, xi, 8, TailStrategy::RoundUp)
        .unroll(x)
        .unroll(c)
        .vectorize(xi)
        .compute_at(output, yiii)
        .reorder({xi, x, y, c});
    umean_a_x
        .store_in(MemoryType::Stack)
        .split(x, x, xi, 8, TailStrategy::RoundUp)
        .vectorize(xi)
        .compute_at(output, yii)
        .reorder({xi, x, y, c});
    mean_a_sum
        .store_in(MemoryType::Stack)
        .split(y, y, yi, 8, TailStrategy::RoundUp)
        .vectorize(yi)
        .compute_at(output, xi)
        .reorder({yi, y, x, c})
        .reorder_storage(y, x, c);
    mean_a_sum.update(0)
        .split(x, x, xi, 6, TailStrategy::RoundUp)
        .split(y, y, yi, 72, TailStrategy::RoundUp)
        .split(xi, xi, xii, 2, TailStrategy::GuardWithIf)
        .split(yi, yi, yii, 8, TailStrategy::GuardWithIf)
        .unroll(yi)
        .vectorize(yii)
        .reorder({yii, yi, r229_x, xii, y, xi, x, c});
    mean_a_x
        .store_in(MemoryType::Stack)
        .split(y, y, yi, 8, TailStrategy::RoundUp)
        .vectorize(yi)
        .compute_at(mean_a_sum, xii)
        .store_at(mean_a_sum, y)
        .reorder({yi, y, x, c})
        .reorder_storage(y, x, c);
    mean_a_sum_x
        .store_in(MemoryType::Stack)
        .split(y, y, yi, 8, TailStrategy::RoundUp)
        .vectorize(yi)
        .compute_at(mean_a_sum, x)
        .reorder({yi, y, x, c})
        .reorder_storage(y, x, c);
    mean_a_sum_x.update(0)
        .split(y, y, yi, 8, TailStrategy::RoundUp)
        .vectorize(yi)
        .reorder({yi, r224_x, y, x, c});
    a
        .split(x, x, xi, 67, TailStrategy::ShiftInwards)
        .split(y, y, yi, 8, TailStrategy::ShiftInwards)
        .vectorize(yi)
        .compute_root()
        .reorder({yi, y, c, xi, x})
        .parallel(x)
        .reorder_storage(y, x, c);
    inv_var_i
        .store_in(MemoryType::Stack)
        .split(y, y, yi, 32, TailStrategy::ShiftInwards)
        .split(yi, yi, yii, 8, TailStrategy::ShiftInwards)
        .unroll(yi)
        .vectorize(yii)
        .compute_at(a, xi)
        .reorder({yii, yi, c, y, x})
        .reorder_storage(y, x, c);
    var_i_det
        .store_in(MemoryType::Stack)
        .split(y, y, yi, 8, TailStrategy::RoundUp)
        .unroll(y)
        .vectorize(yi)
        .compute_at(inv_var_i, c)
        .store_at(inv_var_i, y)
        .reorder({yi, y, x})
        .reorder_storage(y, x);
    var_i
        .store_in(MemoryType::Stack)
        .split(y, y, yi, 16, TailStrategy::ShiftInwards)
        .split(yi, yi, yii, 8, TailStrategy::ShiftInwards)
        .unroll(yi)
        .unroll(c)
        .vectorize(yii)
        .compute_at(inv_var_i, y)
        .reorder({yii, yi, c, y, x})
        .reorder_storage(y, x, c);
    corr_i
        .store_in(MemoryType::Stack)
        .split(c, c, ci, 2, TailStrategy::RoundUp)
        .split(y, y, yi, 8, TailStrategy::RoundUp)
        .unroll(y)
        .unroll(ci)
        .vectorize(yi)
        .compute_at(var_i, y)
        .reorder({yi, y, ci, x, c})
        .reorder_storage(y, x, c);
    corr_i_sum
        .store_in(MemoryType::Stack)
        .split(y, y, yi, 8, TailStrategy::RoundUp)
        .unroll(y)
        .unroll(c)
        .vectorize(yi)
        .compute_at(corr_i, x)
        .reorder({yi, y, x, c})
        .reorder_storage(y, x, c);
    corr_i_sum.update(0)
        .split(y, y, yi, 8, TailStrategy::RoundUp)
        .unroll(y)
        .unroll(c)
        .vectorize(yi)
        .reorder({yi, y, x, c, r119_x});
    corr_i_x
        .store_in(MemoryType::Stack)
        .split(y, y, yi, 8, TailStrategy::RoundUp)
        .split(c, c, ci, 3, TailStrategy::RoundUp)
        .unroll(ci)
        .vectorize(yi)
        .compute_at(a, xi)
        .reorder({yi, ci, y, x, c})
        .reorder_storage(y, x, c);
    corr_i_sum_x
        .store_in(MemoryType::Stack)
        .split(y, y, yi, 8, TailStrategy::RoundUp)
        .unroll(c)
        .vectorize(yi)
        .compute_at(corr_i_x, y)
        .reorder({yi, y, x, c})
        .reorder_storage(y, x, c);
    corr_i_sum_x.update(0)
        .split(y, y, yi, 8, TailStrategy::RoundUp)
        .unroll(c)
        .vectorize(yi)
        .reorder({yi, y, x, c, r114_x});
    channel_corr_i
        .split(y, y, yi, 8, TailStrategy::ShiftInwards)
        .vectorize(yi)
        .compute_at(a, xi)
        .store_at(a, x)
        .reorder({yi, y, x, c})
        .reorder_storage(y, x, c);
    cov_ip
        .store_in(MemoryType::Stack)
        .split(y, y, yi, 192, TailStrategy::RoundUp)
        .split(yi, yi, yii, 64, TailStrategy::RoundUp)
        .split(yii, yii, yiii, 8, TailStrategy::RoundUp)
        .unroll(yii)
        .vectorize(yiii)
        .compute_at(a, xi)
        .reorder({yiii, yii, yi, y, x, c})
        .reorder_storage(y, x, c);
    mean_p
        .split(x, x, xi, 67, TailStrategy::RoundUp)
        .split(xi, xi, xii, 17, TailStrategy::RoundUp)
        .split(y, y, yi, 64, TailStrategy::RoundUp)
        .split(yi, yi, yii, 8, TailStrategy::RoundUp)
        .unroll(yi)
        .vectorize(yii)
        .compute_root()
        .reorder({yii, yi, y, xii, xi, x})
        .parallel(x)
        .reorder_storage(y, x);
    mean_p_sum
        .store_in(MemoryType::Stack)
        .split(y, y, yi, 8, TailStrategy::RoundUp)
        .unroll(y)
        .vectorize(yi)
        .compute_at(mean_p, y)
        .reorder({yi, y, x})
        .reorder_storage(y, x);
    mean_p_sum.update(0)
        .split(r34_x, r34_x, r34_xi, 21, TailStrategy::GuardWithIf)
        .split(y, y, yi, 8, TailStrategy::RoundUp)
        .unroll(y)
        .vectorize(yi)
        .reorder({yi, y, r34_xi, r34_x, x});
    mean_p_x
        .store_in(MemoryType::Stack)
        .split(y, y, yi, 8, TailStrategy::RoundUp)
        .vectorize(yi)
        .compute_at(mean_p_sum, r34_x)
        .reorder({yi, y, x})
        .reorder_storage(y, x);
    mean_p_sum_x
        .store_in(MemoryType::Stack)
        .split(y, y, yi, 8, TailStrategy::RoundUp)
        .vectorize(yi)
        .compute_at(mean_p, xi)
        .reorder({yi, y, x})
        .reorder_storage(y, x);
    mean_p_sum_x.update(0)
        .split(y, y, yi, 8, TailStrategy::RoundUp)
        .vectorize(yi)
        .reorder({yi, r29_x, y, x});
    mean_i
        .split(x, x, xi, 67, TailStrategy::RoundUp)
        .split(y, y, yi, 280, TailStrategy::RoundUp)
        .split(yi, yi, yii, 32, TailStrategy::RoundUp)
        .split(yii, yii, yiii, 8, TailStrategy::RoundUp)
        .unroll(yii)
        .unroll(c)
        .vectorize(yiii)
        .compute_root()
        .reorder({yiii, yii, c, yi, y, xi, x})
        .parallel(x)
        .reorder_storage(y, x, c);
    mean_i_sum
        .store_in(MemoryType::Stack)
        .split(y, y, yi, 8, TailStrategy::RoundUp)
        .unroll(y)
        .unroll(c)
        .vectorize(yi)
        .compute_at(mean_i, yi)
        .reorder({yi, y, x, c})
        .reorder_storage(y, x, c);
    mean_i_sum.update(0)
        .split(y, y, yi, 8, TailStrategy::RoundUp)
        .unroll(y)
        .unroll(c)
        .vectorize(yi)
        .reorder({yi, y, x, c, r72_x});
    mean_i_x
        .store_in(MemoryType::Stack)
        .split(y, y, yi, 24, TailStrategy::RoundUp)
        .split(yi, yi, yii, 8, TailStrategy::RoundUp)
        .unroll(yi)
        .vectorize(yii)
        .compute_at(mean_i, y)
        .reorder({yii, yi, y, x, c})
        .reorder_storage(y, x, c);
    mean_i_sum_x
        .store_in(MemoryType::Stack)
        .split(y, y, yi, 8, TailStrategy::RoundUp)
        .unroll(y)
        .vectorize(yi)
        .compute_at(mean_i_x, y)
        .reorder({yi, y, x, c})
        .reorder_storage(y, x, c);
    mean_i_sum_x.update(0)
        .split(y, y, yi, 8, TailStrategy::RoundUp)
        .unroll(y)
        .vectorize(yi)
        .reorder({yi, y, x, c, r67_x});
    corr_ip
        .store_in(MemoryType::Stack)
        .split(y, y, yi, 8, TailStrategy::RoundUp)
        .unroll(y)
        .vectorize(yi)
        .compute_at(cov_ip, yi)
        .reorder({yi, y, x, c})
        .reorder_storage(y, x, c);
    corr_ip_sum
        .store_in(MemoryType::Stack)
        .split(y, y, yi, 8, TailStrategy::RoundUp)
        .unroll(y)
        .vectorize(yi)
        .compute_at(cov_ip, yi)
        .reorder({yi, y, x, c})
        .reorder_storage(y, x, c);
    corr_ip_sum.update(0)
        .split(y, y, yi, 8, TailStrategy::RoundUp)
        .unroll(y)
        .vectorize(yi)
        .reorder({yi, y, x, c, r166_x});
    corr_ip_x
        .store_in(MemoryType::Stack)
        .split(y, y, yi, 16, TailStrategy::RoundUp)
        .split(yi, yi, yii, 8, TailStrategy::RoundUp)
        .unroll(yi)
        .vectorize(yii)
        .compute_at(cov_ip, y)
        .reorder({yii, yi, y, x, c})
        .reorder_storage(y, x, c);
    corr_ip_sum_x
        .store_in(MemoryType::Stack)
        .split(y, y, yi, 8, TailStrategy::RoundUp)
        .unroll(y)
        .vectorize(yi)
        .compute_at(corr_ip_x, y)
        .reorder({yi, y, x, c})
        .reorder_storage(y, x, c);
    corr_ip_sum_x.update(0)
        .split(y, y, yi, 8, TailStrategy::RoundUp)
        .unroll(y)
        .vectorize(yi)
        .reorder({yi, y, x, c, r161_x});
    lambda_2
        .split(y, y, yi, 8, TailStrategy::RoundUp)
        .vectorize(yi)
        .compute_at(a, xi)
        .store_at(a, x)
        .reorder({yi, y, x, c})
        .reorder_storage(y, x, c);
    p_bounded
        .split(x, x, xi, 70, TailStrategy::ShiftInwards)
        .split(y, y, yi, 8, TailStrategy::ShiftInwards)
        .vectorize(yi)
        .compute_root()
        .reorder({yi, y, xi, x})
        .parallel(x)
        .reorder_storage(y, x);
    repeat_edge
        .split(_1, _1, _1i, 32, TailStrategy::ShiftInwards)
        .vectorize(_1i)
        .compute_at(p_bounded, x)
        .reorder({_1i, _1, _0})
        .reorder_storage(_1, _0);
    i_bounded
        .split(x, x, xi, 70, TailStrategy::ShiftInwards)
        .split(y, y, yi, 8, TailStrategy::ShiftInwards)
        .vectorize(yi)
        .compute_root()
        .reorder({yi, y, xi, x, c})
        .fuse(x, c, x)
        .parallel(x)
        .reorder_storage(y, x, c);
    i_bounded_x
        .split(x, x, xi, 14, TailStrategy::ShiftInwards)
        .split(y, y, yi, 32, TailStrategy::ShiftInwards)
        .vectorize(yi)
        .compute_at(i_bounded, x)
        .reorder({yi, y, xi, x, c})
        .reorder_storage(y, x, c);
    repeat_edge_1
        .store_in(MemoryType::Stack)
        .split(_1, _1, _1i, 32, TailStrategy::ShiftInwards)
        .vectorize(_1i)
        .compute_at(i_bounded_x, xi)
        .store_at(i_bounded_x, x)
        .reorder({_1i, _1, _0, _2})
        .reorder_storage(_1, _0, _2);

}

#endif  // color_guided_image_filter_SCHEDULE_H
