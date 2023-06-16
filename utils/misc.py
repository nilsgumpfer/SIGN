def pretty_method_name(m, wo_params=False):
    mapping_with_params = {
        'gradient_x_sign': r'Gradient $\times$ SIGN $(\mu = 0)$',
        'gradient_x_sign_mu_0': r'Gradient $\times$ SIGN $(\mu = 0)$',
        'gradient_x_sign_mu_0_5': r'Gradient $\times$ SIGN $(\mu = 0.5)$',
        'gradient_x_sign_mu_neg_0_5': r'Gradient $\times$ SIGN $(\mu = -0.5)$',

        'smoothgrad_x_sign': r'SmoothGrad $\times$ SIGN $(\mu = 0)$',
        'smoothgrad_x_sign_mu_0': r'SmoothGrad $\times$ SIGN $(\mu = 0)$',
        'smoothgrad_x_sign_mu_0_5': r'SmoothGrad $\times$ SIGN $(\mu = 0.5)$',
        'smoothgrad_x_sign_mu_neg_0_5': r'SmoothGrad $\times$ SIGN $(\mu = -0.5)$',

        'deconvnet_x_sign': r'DeconvNet $\times$ SIGN $(\mu = 0)$',
        'deconvnet_x_sign_mu_0': r'DeconvNet $\times$ SIGN $(\mu = 0)$',
        'deconvnet_x_sign_mu_0_5': r'DeconvNet $\times$ SIGN $(\mu = 0.5)$',
        'deconvnet_x_sign_mu_neg_0_5': r'DeconvNet $\times$ SIGN $(\mu = -0.5)$',

        'guided_backprop_x_sign': r'Guided Backpropagation $\times$ SIGN $(\mu = 0)$',
        'guided_backprop_x_sign_mu_0': r'Guided Backpropagation $\times$ SIGN $(\mu = 0)$',
        'guided_backprop_x_sign_mu_0_5': r'Guided Backpropagation $\times$ SIGN $(\mu = 0.5)$',
        'guided_backprop_x_sign_mu_neg_0_5': r'Guided Backpropagation $\times$ SIGN $(\mu = -0.5)$',

        'lrp_epsilon_0_01': r'LRP-$\epsilon~(\epsilon = 0.01)$',
        'zblrp_epsilon_0_01_VGG16ILSVRC': r'LRP-$\epsilon~(\epsilon = 0.01)$ / LRP-$z^\mathcal{B}$',
        'w2lrp_epsilon_0_01': r'LRP-$\epsilon~(\epsilon = 0.01)$ / LRP-$w^2$',
        'flatlrp_epsilon_0_01': r'LRP-$\epsilon~(\epsilon = 0.01)$ / LRP-flat',
        'lrpz_epsilon_0_01': r'LRP-$\epsilon~(\epsilon = 0.01)$ / LRP-$z$',
        'lrpsign_epsilon_0_01': r'LRP-$\epsilon~(\epsilon = 0.01)$ / LRP-SIGN $(\mu = 0)$',

        'lrp_epsilon_0_1': r'LRP-$\epsilon~(\epsilon = 0.1)$',
        'zblrp_epsilon_0_1_VGG16ILSVRC': r'LRP-$\epsilon~(\epsilon = 0.1)$ / LRP-$z^\mathcal{B}$',
        'w2lrp_epsilon_0_1': r'LRP-$\epsilon~(\epsilon = 0.1)$ / LRP-$w^2$',
        'flatlrp_epsilon_0_1': r'LRP-$\epsilon~(\epsilon = 0.1)$ / LRP-flat',
        'lrpz_epsilon_0_1': r'LRP-$\epsilon~(\epsilon = 0.1)$ / LRP-$z$',
        'lrpsign_epsilon_0_1': r'LRP-$\epsilon~(\epsilon = 0.1)$ / LRP-SIGN $(\mu = 0)$',

        'lrp_epsilon_1': r'LRP-$\epsilon~(\epsilon = 1)$',
        'zblrp_epsilon_1_VGG16ILSVRC': r'LRP-$\epsilon~(\epsilon = 1)$ / LRP-$z^\mathcal{B}$',
        'w2lrp_epsilon_1': r'LRP-$\epsilon~(\epsilon = 1)$ / LRP-$w^2$',
        'flatlrp_epsilon_1': r'LRP-$\epsilon~(\epsilon = 1)$ / LRP-flat',
        'lrpz_epsilon_1': r'LRP-$\epsilon~(\epsilon = 1)$ / LRP-$z$',
        'lrpsign_epsilon_1': r'LRP-$\epsilon~(\epsilon = 1)$ / LRP-SIGN $(\mu = 0)$',

        'lrp_epsilon_10': r'LRP-$\epsilon~(\epsilon = 10)$',
        'zblrp_epsilon_10_VGG16ILSVRC': r'LRP-$\epsilon~(\epsilon = 10)$ / LRP-$z^\mathcal{B}$',
        'w2lrp_epsilon_10': r'LRP-$\epsilon~(\epsilon = 10)$ / LRP-$w^2$',
        'flatlrp_epsilon_10': r'LRP-$\epsilon~(\epsilon = 10)$ / LRP-flat',
        'lrpz_epsilon_10': r'LRP-$\epsilon~(\epsilon = 10)$ / LRP-$z$',
        'lrpsign_epsilon_10': r'LRP-$\epsilon~(\epsilon = 10)$ / LRP-SIGN $(\mu = 0)$',

        'lrp_epsilon_100': r'LRP-$\epsilon~(\epsilon = 100)$',
        'zblrp_epsilon_100_VGG16ILSVRC': r'LRP-$\epsilon~(\epsilon = 100)$ / LRP-$z^\mathcal{B}$',
        'w2lrp_epsilon_100': r'LRP-$\epsilon~(\epsilon = 100)$ / LRP-$w^2$',
        'flatlrp_epsilon_100': r'LRP-$\epsilon~(\epsilon = 100)$ / LRP-flat',
        'lrpz_epsilon_100': r'LRP-$\epsilon~(\epsilon = 100)$ / LRP-$z$',
        'lrpsign_epsilon_100': r'LRP-$\epsilon~(\epsilon = 100)$ / LRP-SIGN $(\mu = 0)$',
        'flrpsign_epsilon_100': r'LRP-$\epsilon~(\epsilon = 100)$ / LRP-FSIGN',
        'lrpsign_epsilon_100_mu_0': r'LRP-$\epsilon~(\epsilon = 100)$ / LRP-SIGN $(\mu = 0)$',
        'lrpsign_epsilon_100_mu_0_5': r'LRP-$\epsilon~(\epsilon = 100)$ / LRP-SIGN $(\mu = 0.5)$',
        'lrpsign_epsilon_100_mu_neg_0_5': r'LRP-$\epsilon~(\epsilon = 100)$ / LRP-SIGN $(\mu = -0.5)$',
        'lrpsign_epsilon_100_mu_repl': r'LRP-$\epsilon~(\epsilon = 100)$ / LRP-SIGN$^*$',

        'lrp_epsilon_0_1_std_x': r'LRP-$\epsilon~(\epsilon = 0.1 \cdot \sigma(x))$',
        'zblrp_epsilon_0_1_std_x_VGG16ILSVRC': r'LRP-$\epsilon~(\epsilon = 0.1 \cdot \sigma(x))$ / LRP-$z^\mathcal{B}$',
        'w2lrp_epsilon_0_1_std_x': r'LRP-$\epsilon~(\epsilon = 0.1 \cdot \sigma(x))$ / LRP-$w^2$',
        'flatlrp_epsilon_0_1_std_x': r'LRP-$\epsilon~(\epsilon = 0.1 \cdot \sigma(x))$ / LRP-flat',
        'lrpz_epsilon_0_1_std_x': r'LRP-$\epsilon~(\epsilon = 0.1 \cdot \sigma(x))$ / LRP-$z$',
        'lrpsign_epsilon_0_1_std_x': r'LRP-$\epsilon~(\epsilon = 0.1 \cdot \sigma(x))$ / LRP-SIGN $(\mu = 0)$',

        'lrp_epsilon_0_25_std_x': r'LRP-$\epsilon~(\epsilon = 0.25 \cdot \sigma(x))$',
        'zblrp_epsilon_0_25_std_x_VGG16ILSVRC': r'LRP-$\epsilon~(\epsilon = 0.25 \cdot \sigma(x))$ / LRP-$z^\mathcal{B}$',
        'w2lrp_epsilon_0_25_std_x': r'LRP-$\epsilon~(\epsilon = 0.25 \cdot \sigma(x))$ / LRP-$w^2$',
        'flatlrp_epsilon_0_25_std_x': r'LRP-$\epsilon~(\epsilon = 0.25 \cdot \sigma(x))$ / LRP-flat',
        'lrpz_epsilon_0_25_std_x': r'LRP-$\epsilon~(\epsilon = 0.25 \cdot \sigma(x))$ / LRP-$z$',
        'lrpsign_epsilon_0_25_std_x': r'LRP-$\epsilon~(\epsilon = 0.25 \cdot \sigma(x))$ / LRP-SIGN $(\mu = 0)$',
        'lrpsign_epsilon_0_25_std_x_mu_0': r'LRP-$\epsilon~(\epsilon = 0.25 \cdot \sigma(x))$ / LRP-SIGN $(\mu = 0)$',
        'lrpsign_epsilon_0_25_std_x_mu_0_5': r'LRP-$\epsilon~(\epsilon = 0.25 \cdot \sigma(x))$ / LRP-SIGN $(\mu = 0.5)$',
        'lrpsign_epsilon_0_25_std_x_mu_neg_0_5': r'LRP-$\epsilon~(\epsilon = 0.25 \cdot \sigma(x))$ / LRP-SIGN $(\mu = -0.5)$',

        'lrp_epsilon_0_5_std_x': r'LRP-$\epsilon~(\epsilon = 0.5 \cdot \sigma(x))$',
        'zblrp_epsilon_0_5_std_x_VGG16ILSVRC': r'LRP-$\epsilon~(\epsilon = 0.5 \cdot \sigma(x))$ / LRP-$z^\mathcal{B}$',
        'w2lrp_epsilon_0_5_std_x': r'LRP-$\epsilon~(\epsilon = 0.5 \cdot \sigma(x))$ / LRP-$w^2$',
        'flatlrp_epsilon_0_5_std_x': r'LRP-$\epsilon~(\epsilon = 0.5 \cdot \sigma(x))$ / LRP-flat',
        'lrpz_epsilon_0_5_std_x': r'LRP-$\epsilon~(\epsilon = 0.5 \cdot \sigma(x))$ / LRP-$z$',
        'lrpsign_epsilon_0_5_std_x': r'LRP-$\epsilon~(\epsilon = 0.5 \cdot \sigma(x))$ / LRP-SIGN $(\mu = 0)$',

        'lrpsign_z': r'LRP-$z$ / LRP-SIGN $(\mu = 0)$',
        'lrpsign_alpha_1_beta_0': r'LRP-$\alpha_1 \beta_0$ / LRP-SIGN $(\mu = 0)$',
        'lrpsign_sequential_composite_a': r'LRP composite A / LRP-SIGN $(\mu = 0)$',
        'lrpsign_sequential_composite_b': r'LRP composite B / LRP-SIGN $(\mu = 0)$'

    }

    mapping_wo_params = {'random_uniform': 'Baseline (random uniform)',
                         'gradient': r'Gradient',
                         'gradient_x_input': r'Gradient $\times$ Input',

                         'gradient_x_sign': r'Gradient $\times$ SIGN',
                         'gradient_x_sign_mu_0': r'Gradient $\times$ SIGN',
                         'gradient_x_sign_mu_0_5': r'Gradient $\times$ SIGN',
                         'gradient_x_sign_mu_neg_0_5': r'Gradient $\times$ SIGN',
                         'gradient_x_sign_mu_repl': r'Gradient $\times$ SIGN$^*$',
                         'gradient_x_sign_mu_mean_01': r'Gradient $\times$ SIGN',
                         'gradient_x_sign_mu_mean_01_inv': r'Gradient $\times$ SIGN',
                         'gradient_x_sign_mu_mean_10': r'Gradient $\times$ SIGN',
                         'gradient_x_sign_mu_mean_10_inv': r'Gradient $\times$ SIGN',
                         'gradient_x_sign_mu_mean_11': r'Gradient $\times$ SIGN',
                         'gradient_x_sign_mu_mean_11_inv': r'Gradient $\times$ SIGN',

                         'smoothgrad_x_sign': r'SmoothGrad $\times$ SIGN',
                         'smoothgrad_x_sign_mu_0': r'SmoothGrad $\times$ SIGN',
                         'smoothgrad_x_sign_mu_0_5': r'SmoothGrad $\times$ SIGN',
                         'smoothgrad_x_sign_mu_neg_0_5': r'SmoothGrad $\times$ SIGN',
                         'smoothgrad_x_sign_mu_repl': r'SmoothGrad $\times$ SIGN$^*$',
                         'smoothgrad_x_sign_mu_mean_01': r'SmoothGrad $\times$ SIGN',
                         'smoothgrad_x_sign_mu_mean_01_inv': r'SmoothGrad $\times$ SIGN',
                         'smoothgrad_x_sign_mu_mean_10': r'SmoothGrad $\times$ SIGN',
                         'smoothgrad_x_sign_mu_mean_10_inv': r'SmoothGrad $\times$ SIGN',
                         'smoothgrad_x_sign_mu_mean_11': r'SmoothGrad $\times$ SIGN',
                         'smoothgrad_x_sign_mu_mean_11_inv': r'SmoothGrad $\times$ SIGN',

                         'deconvnet_x_sign': r'DeconvNet $\times$ SIGN',
                         'deconvnet_x_sign_mu_0': r'DeconvNet $\times$ SIGN',
                         'deconvnet_x_sign_mu_0_5': r'DeconvNet $\times$ SIGN',
                         'deconvnet_x_sign_mu_neg_0_5': r'DeconvNet $\times$ SIGN',
                         'deconvnet_x_sign_mu_repl': r'DeconvNet $\times$ SIGN$^*$',
                         'deconvnet_x_sign_mu_mean_01': r'DeconvNet $\times$ SIGN',
                         'deconvnet_x_sign_mu_mean_01_inv': r'DeconvNet $\times$ SIGN',
                         'deconvnet_x_sign_mu_mean_10': r'DeconvNet $\times$ SIGN',
                         'deconvnet_x_sign_mu_mean_10_inv': r'DeconvNet $\times$ SIGN',
                         'deconvnet_x_sign_mu_mean_11': r'DeconvNet $\times$ SIGN',
                         'deconvnet_x_sign_mu_mean_11_inv': r'DeconvNet $\times$ SIGN',

                         'guided_backprop_x_sign': r'Guided Backpropagation $\times$ SIGN',
                         'guided_backprop_x_sign_mu_0': r'Guided Backpropagation $\times$ SIGN',
                         'guided_backprop_x_sign_mu_0_5': r'Guided Backpropagation $\times$ SIGN',
                         'guided_backprop_x_sign_mu_neg_0_5': r'Guided Backpropagation $\times$ SIGN',
                         'guided_backprop_x_sign_mu_repl': r'Guided Backpropagation $\times$ SIGN$^*$',
                         'guided_backprop_x_sign_mu_mean_01': r'Guided Backpropagation $\times$ SIGN',
                         'guided_backprop_x_sign_mu_mean_01_inv': r'Guided Backpropagation $\times$ SIGN',
                         'guided_backprop_x_sign_mu_mean_10': r'Guided Backpropagation $\times$ SIGN',
                         'guided_backprop_x_sign_mu_mean_10_inv': r'Guided Backpropagation $\times$ SIGN',
                         'guided_backprop_x_sign_mu_mean_11': r'Guided Backpropagation $\times$ SIGN',
                         'guided_backprop_x_sign_mu_mean_11_inv': r'Guided Backpropagation $\times$ SIGN',

                         'integrated_gradients': r'Integrated Gradients',
                         'smoothgrad': r'SmoothGrad',
                         'vargrad': r'VarGrad',
                         'deconvnet': r'DeconvNet',
                         'guided_backprop': r'Guided Backpropagation',

                         'grad_cam_MNISTCNN': r'Grad-CAM',
                         'grad_cam_VGG16ILSVRC': r'Grad-CAM',
                         'grad_cam_VGG16MITPL365': r'Grad-CAM',

                         'guided_grad_cam_MNISTCNN': r'Guided Grad-CAM',
                         'guided_grad_cam_VGG16ILSVRC': r'Guided Grad-CAM',
                         'guided_grad_cam_VGG16MITPL365': r'Guided Grad-CAM',

                         'lrp_z': r'LRP-$z$',
                         'zblrp_z_VGG16ILSVRC': r'LRP-$z$ / LRP-$z^\mathcal{B}$',
                         'w2lrp_z': r'LRP-$z$ / LRP-$w^2$',
                         'flatlrp_z': r'LRP-$z$ / LRP-flat',
                         'lrpsign_z': r'LRP-$z$ / LRP-SIGN',

                         'lrp_epsilon_0_01': r'LRP-$\epsilon$',
                         'zblrp_epsilon_0_01_VGG16ILSVRC': r'LRP-$\epsilon$ / LRP-$z^\mathcal{B}$',
                         'w2lrp_epsilon_0_01': r'LRP-$\epsilon$ / LRP-$w^2$',
                         'flatlrp_epsilon_0_01': r'LRP-$\epsilon$ / LRP-flat',
                         'lrpz_epsilon_0_01': r'LRP-$\epsilon$ / LRP-$z$',
                         'lrpsign_epsilon_0_01': r'LRP-$\epsilon$ / LRP-SIGN',

                         'lrp_epsilon_0_1': r'LRP-$\epsilon$',
                         'zblrp_epsilon_0_1_VGG16ILSVRC': r'LRP-$\epsilon$ / LRP-$z^\mathcal{B}$',
                         'w2lrp_epsilon_0_1': r'LRP-$\epsilon$ / LRP-$w^2$',
                         'flatlrp_epsilon_0_1': r'LRP-$\epsilon$ / LRP-flat',
                         'lrpz_epsilon_0_1': r'LRP-$\epsilon$ / LRP-$z$',
                         'lrpsign_epsilon_0_1': r'LRP-$\epsilon$ / LRP-SIGN',

                         'lrp_epsilon_1': r'LRP-$\epsilon$',
                         'zblrp_epsilon_1_VGG16ILSVRC': r'LRP-$\epsilon$ / LRP-$z^\mathcal{B}$',
                         'w2lrp_epsilon_1': r'LRP-$\epsilon$ / LRP-$w^2$',
                         'flatlrp_epsilon_1': r'LRP-$\epsilon$ / LRP-flat',
                         'lrpz_epsilon_1': r'LRP-$\epsilon$ / LRP-$z$',
                         'lrpsign_epsilon_1': r'LRP-$\epsilon$ / LRP-SIGN',

                         'lrp_epsilon_10': r'LRP-$\epsilon$',
                         'zblrp_epsilon_10_VGG16ILSVRC': r'LRP-$\epsilon$ / LRP-$z^\mathcal{B}$',
                         'w2lrp_epsilon_10': r'LRP-$\epsilon$ / LRP-$w^2$',
                         'flatlrp_epsilon_10': r'LRP-$\epsilon$ / LRP-flat',
                         'lrpz_epsilon_10': r'LRP-$\epsilon$ / LRP-$z$',
                         'lrpsign_epsilon_10': r'LRP-$\epsilon$ / LRP-SIGN',

                         'lrp_epsilon_100': r'LRP-$\epsilon$',
                         'zblrp_epsilon_100_VGG16ILSVRC': r'LRP-$\epsilon$ / LRP-$z^\mathcal{B}$',
                         'w2lrp_epsilon_100': r'LRP-$\epsilon$ / LRP-$w^2$',
                         'flatlrp_epsilon_100': r'LRP-$\epsilon$ / LRP-flat',
                         'lrpz_epsilon_100': r'LRP-$\epsilon$ / LRP-$z$',
                         'lrpsign_epsilon_100': r'LRP-$\epsilon$ / LRP-SIGN',
                         'flrpsign_epsilon_100': r'LRP-$\epsilon$ / LRP-FSIGN',
                         'lrpsign_epsilon_100_mu_0': r'LRP-$\epsilon$ / LRP-SIGN',
                         'lrpsign_epsilon_100_mu_0_5': r'LRP-$\epsilon$ / LRP-SIGN',
                         'lrpsign_epsilon_100_mu_neg_0_5': r'LRP-$\epsilon$ / LRP-SIGN',
                         'lrpsign_epsilon_100_mu_repl': r'LRP-$\epsilon$ / LRP-SIGN$^*$',
                         'lrpsign_epsilon_100_mu_mean_01': r'LRP-$\epsilon$ / LRP-SIGN',
                         'lrpsign_epsilon_100_mu_mean_01_inv': r'LRP-$\epsilon$ / LRP-SIGN',
                         'lrpsign_epsilon_100_mu_mean_10': r'LRP-$\epsilon$ / LRP-SIGN',
                         'lrpsign_epsilon_100_mu_mean_10_inv': r'LRP-$\epsilon$ / LRP-SIGN',
                         'lrpsign_epsilon_100_mu_mean_11': r'LRP-$\epsilon$ / LRP-SIGN',
                         'lrpsign_epsilon_100_mu_mean_11_inv': r'LRP-$\epsilon$ / LRP-SIGN',

                         'lrp_epsilon_0_1_std_x': r'LRP-$\epsilon$',
                         'zblrp_epsilon_0_1_std_x_VGG16ILSVRC': r'LRP-$\epsilon$ / LRP-$z^\mathcal{B}$',
                         'w2lrp_epsilon_0_1_std_x': r'LRP-$\epsilon$ / LRP-$w^2$',
                         'flatlrp_epsilon_0_1_std_x': r'LRP-$\epsilon$ / LRP-flat',
                         'lrpz_epsilon_0_1_std_x': r'LRP-$\epsilon$ / LRP-$z$',
                         'lrpsign_epsilon_0_1_std_x': r'LRP-$\epsilon$ / LRP-SIGN',

                         'lrp_epsilon_0_25_std_x': r'LRP-$\epsilon$',
                         'zblrp_epsilon_0_25_std_x_VGG16ILSVRC': r'LRP-$\epsilon$ / LRP-$z^\mathcal{B}$',
                         'w2lrp_epsilon_0_25_std_x': r'LRP-$\epsilon$ / LRP-$w^2$',
                         'flatlrp_epsilon_0_25_std_x': r'LRP-$\epsilon$ / LRP-flat',
                         'lrpz_epsilon_0_25_std_x': r'LRP-$\epsilon$ / LRP-$z$',
                         'lrpsign_epsilon_0_25_std_x': r'LRP-$\epsilon$ / LRP-SIGN',
                         'lrpsign_epsilon_0_25_std_x_mu_0': r'LRP-$\epsilon$ / LRP-SIGN',
                         'lrpsign_epsilon_0_25_std_x_mu_0_5': r'LRP-$\epsilon$ / LRP-SIGN',
                         'lrpsign_epsilon_0_25_std_x_mu_neg_0_5': r'LRP-$\epsilon$ / LRP-SIGN',

                         'lrp_epsilon_0_5_std_x': r'LRP-$\epsilon$',
                         'zblrp_epsilon_0_5_std_x_VGG16ILSVRC': r'LRP-$\epsilon$ / LRP-$z^\mathcal{B}$',
                         'w2lrp_epsilon_0_5_std_x': r'LRP-$\epsilon$ / LRP-$w^2$',
                         'flatlrp_epsilon_0_5_std_x': r'LRP-$\epsilon$ / LRP-flat',
                         'lrpz_epsilon_0_5_std_x': r'LRP-$\epsilon$ / LRP-$z$',
                         'lrpsign_epsilon_0_5_std_x': r'LRP-$\epsilon$ / LRP-SIGN',

                         'lrp_alpha_1_beta_0': r'LRP-$\alpha_1 \beta_0$',
                         'zblrp_alpha_1_beta_0_VGG16ILSVRC': r'LRP-$\alpha_1 \beta_0$ / LRP-$z^\mathcal{B}$',
                         'w2lrp_alpha_1_beta_0': r'LRP-$\alpha_1 \beta_0$ / LRP-$w^2$',
                         'flatlrp_alpha_1_beta_0': r'LRP-$\alpha_1 \beta_0$ / LRP-flat',
                         'lrpz_alpha_1_beta_0': r'LRP-$\alpha_1 \beta_0$ / LRP-$z$',
                         'lrpsign_alpha_1_beta_0': r'LRP-$\alpha_1 \beta_0$ / LRP-SIGN',

                         'lrp_sequential_composite_a': r'LRP composite A',
                         'zblrp_sequential_composite_a_VGG16ILSVRC': r'LRP composite A / LRP-$z^\mathcal{B}$',
                         'w2lrp_sequential_composite_a': r'LRP composite A / LRP-$w^2$',
                         'flatlrp_sequential_composite_a': r'LRP composite A / LRP-flat',
                         'lrpz_sequential_composite_a': r'LRP composite A / LRP-$z$',
                         'lrpsign_sequential_composite_a': r'LRP composite A / LRP-SIGN',

                         'lrp_sequential_composite_b': r'LRP composite B',
                         'zblrp_sequential_composite_b_VGG16ILSVRC': r'LRP composite B / LRP-$z^\mathcal{B}$',
                         'w2lrp_sequential_composite_b': r'LRP composite B / LRP-$w^2$',
                         'flatlrp_sequential_composite_b': r'LRP composite B / LRP-flat',
                         'lrpz_sequential_composite_b': r'LRP composite B / LRP-$z$',
                         'lrpsign_sequential_composite_b': r'LRP composite B / LRP-SIGN'
                         }

    if wo_params:
        try:
            return mapping_wo_params[m]
        except KeyError:
            return m.replace('_', r'\_')
    else:
        try:
            return mapping_with_params[m]
        except KeyError:
            try:
                return mapping_wo_params[m]
            except KeyError:
                return m.replace('_', r'\_')


def method_color(m):
    mapping = {'gradient': 'lightgreen',

               'gradient_x_input': 'deepskyblue',
               'gradient_x_sign': 'royalblue',
               'gradient_x_sign_mu_0': 'royalblue',
               'gradient_x_sign_mu_0_5': 'royalblue',
               'gradient_x_sign_mu_neg_0_5': 'royalblue',

               'integrated_gradients': 'magenta',

               'smoothgrad': 'lightseagreen',
               'smoothgrad_x_sign': 'mediumseagreen',
               'smoothgrad_x_sign_mu_0': 'mediumseagreen',
               'smoothgrad_x_sign_mu_0_5': 'mediumseagreen',
               'smoothgrad_x_sign_mu_neg_0_5': 'mediumseagreen',

               'vargrad': 'darkorange',

               'deconvnet': 'teal',
               'deconvnet_x_sign': 'darkviolet',
               'deconvnet_x_sign_mu_0': 'darkviolet',
               'deconvnet_x_sign_mu_0_5': 'darkviolet',
               'deconvnet_x_sign_mu_neg_0_5': 'darkviolet',

               'guided_backprop': 'red',
               'guided_backprop_x_sign': 'lawngreen',
               'guided_backprop_x_sign_mu_0': 'lawngreen',
               'guided_backprop_x_sign_mu_0_5': 'lawngreen',
               'guided_backprop_x_sign_mu_neg_0_5': 'lawngreen',

               'grad_cam_MNISTCNN': 'gold',
               'grad_cam_VGG16ILSVRC': 'gold',
               'grad_cam_VGG16MITPL365': 'gold',

               'guided_grad_cam_MNISTCNN': 'darkgoldenrod',
               'guided_grad_cam_VGG16ILSVRC': 'darkgoldenrod',
               'guided_grad_cam_VGG16MITPL365': 'darkgoldenrod',

               'lrpz_epsilon_10': 'lightcoral',

               'lrpz_epsilon_100': 'lightcoral',
               'lrpsign_epsilon_100_mu_0': 'maroon',
               'lrpsign_epsilon_100_mu_0_5': 'maroon',
               'lrpsign_epsilon_100_mu_neg_0_5': 'maroon',

               'lrpz_epsilon_0_1_std_x': 'lightcoral',
               'lrpsign_epsilon_0_1_std_x': 'maroon',

               'lrpz_epsilon_0_25_std_x': 'lightcoral',
               'lrpsign_epsilon_0_25_std_x': 'maroon',

               'lrpz_alpha_1_beta_0': 'yellowgreen',
               'lrpsign_alpha_1_beta_0': 'forestgreen'
               }

    try:
        return mapping[m]
    except KeyError:
        print('WARNING: No default color set for "{}". Using color "gray".'.format(m))
        return 'gray'


def pretty_header_name(h):
    mapping = {'01': r'[0, 1]',
               '11': r'[-1, 1]',
               '10': r'[-1, 0]',
               'INV_True': r'inverted',
               'INV_False': r'normal'
               }

    try:
        return mapping[h]
    except KeyError:
        return h


def multiplication_by_x(m):
    mapping = {'gradient_x_input': 1,
               'gradient_x_sign': 2,
               'gradient_x_sign_mu_0': 2,
               'gradient_x_sign_mu_0_5': 2,
               'gradient_x_sign_mu_neg_0_5': 2,
               'gradient_x_sign_mu_repl': 2,

               'smoothgrad_x_sign': 2,
               'smoothgrad_x_sign_mu_0': 2,
               'smoothgrad_x_sign_mu_0_5': 2,
               'smoothgrad_x_sign_mu_neg_0_5': 2,

               'deconvnet_x_sign': 2,
               'deconvnet_x_sign_mu_0': 2,
               'deconvnet_x_sign_mu_0_5': 2,
               'deconvnet_x_sign_mu_neg_0_5': 2,

               'guided_backprop_x_sign': 2,
               'guided_backprop_x_sign_mu_0': 2,
               'guided_backprop_x_sign_mu_0_5': 2,
               'guided_backprop_x_sign_mu_neg_0_5': 2,

               'integrated_gradients': 1,

               'lrp_z': 1,
               'lrpsign_z': 2,

               'lrp_epsilon_0_01': 1,
               'lrpz_epsilon_0_01': 1,
               'lrpsign_epsilon_0_01': 2,

               'lrp_epsilon_0_1': 1,
               'lrpz_epsilon_0_1': 1,
               'lrpsign_epsilon_0_1': 2,

               'lrp_epsilon_1': 1,
               'lrpz_epsilon_1': 1,
               'lrpsign_epsilon_1': 2,

               'lrp_epsilon_10': 1,
               'lrpz_epsilon_10': 1,
               'lrpsign_epsilon_10': 2,

               'lrp_epsilon_100': 1,
               'lrpz_epsilon_100': 1,
               'lrpsign_epsilon_100': 2,
               'lrpsign_epsilon_100_mu_0': 2,
               'lrpsign_epsilon_100_mu_0_5': 2,
               'lrpsign_epsilon_100_mu_neg_0_5': 2,
               'lrpsign_epsilon_100_mu_repl': 2,

               'lrp_epsilon_0_1_std_x': 1,
               'lrpz_epsilon_0_1_std_x': 1,
               'lrpsign_epsilon_0_1_std_x': 2,

               'lrp_epsilon_0_25_std_x': 1,
               'lrpz_epsilon_0_25_std_x': 1,
               'lrpsign_epsilon_0_25_std_x': 2,
               'lrpsign_epsilon_0_25_std_x_mu_0': 2,
               'lrpsign_epsilon_0_25_std_x_mu_0_5': 2,
               'lrpsign_epsilon_0_25_std_x_mu_neg_0_5': 2,

               'lrp_epsilon_0_5_std_x': 1,
               'lrpz_epsilon_0_5_std_x': 1,
               'lrpsign_epsilon_0_5_std_x': 2,

               'lrp_alpha_1_beta_0': 1,
               'lrpz_alpha_1_beta_0': 1,
               'lrpsign_alpha_1_beta_0': 2,

               'lrp_sequential_composite_a': 1,
               'lrpz_sequential_composite_a': 1,
               'lrpsign_sequential_composite_a': 2,

               'lrpz_sequential_composite_b': 1,
               'lrpsign_sequential_composite_b': 2
               }

    try:
        return mapping[m]
    except KeyError:
        return 0
