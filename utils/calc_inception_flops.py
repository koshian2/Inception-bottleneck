def calc_single_conv_layer(kernel, width, input_ch, output_ch):
    return kernel**2 * width ** 2 * input_ch * output_ch

def calc_single_normal_layer(kernel, width, input_ch):
    return kernel**2 * width ** 2 * input_ch

def calc_single_module(kernel, width, input_ch, output_ch):
    x = calc_single_conv_layer(kernel, width, input_ch, output_ch) # Conv
    x += calc_single_normal_layer(1, width, input_ch) # BatchNorm（簡易計算）
    x += calc_single_normal_layer(1, width, input_ch) # ReLU
    return x

def calc_single_module_with_bottleneck(mode, kernel, width, input_ch, output_ch, alpha):
    if mode == 1:
        return calc_single_module(kernel, width, input_ch, output_ch)
    elif mode == 2:
        # Bottleneck -> Conv
        x = calc_single_module(1, width, input_ch, output_ch//alpha)
        x += calc_single_module(kernel, width, output_ch//alpha, output_ch)
        return x
    elif mode == 3:
        # Bottlneck -> Conv -> Bottleneck
        x = calc_single_module(1, width, input_ch, output_ch//alpha)
        x += calc_single_module(kernel, width, output_ch//alpha, output_ch//alpha)
        x += calc_single_module(1, width, output_ch//alpha, output_ch)
        return x

def calc_inception_module(width, input_ch, output_ch, alpha, mode):
    assert output_ch % (alpha * 8) == 0
    assert mode >= 1 and mode <= 3 and type(mode) is int
    # conv1
    x = calc_single_module(1, width, input_ch, output_ch//4)
    # conv3, 5
    x += calc_single_module_with_bottleneck(mode, 3, width, input_ch, output_ch//2, alpha)
    x += calc_single_module_with_bottleneck(mode, 5, width, input_ch, output_ch//8, alpha)
    # pool
    x += calc_single_normal_layer(3, width, input_ch)
    x += calc_single_module(1, width, input_ch, output_ch//8)
    return x

def calc_model_flops(mode, alpha):
    x = calc_inception_module(32, 3, 96, alpha, mode)
    # pool 32->16
    x += calc_single_normal_layer(2, 32, 96)
    x += calc_inception_module(16, 96, 256, alpha, mode)
    # pool 16 -> 8
    x += calc_single_normal_layer(2, 16, 256)
    x += calc_inception_module(8, 256, 384, alpha, mode)
    x += calc_inception_module(8, 384, 384, alpha, mode)
    x += calc_inception_module(8, 384, 256, alpha, mode)
    return x

m1 = calc_model_flops(1, 2)
m2 = calc_model_flops(2, 2)
m3 = calc_model_flops(2, 4)
m4 = calc_model_flops(3, 2)
m5 = calc_model_flops(3, 4)

print(f"{m1:,d}\t{m2:,d}\t{m3:,d}\t{m4:,d}\t{m5:,d}")
