# Sampling
1. p_sample_loop
   1. image = pred_img = noise ~ N(0, 1), cond=None
   2. p_sample
      1. noise ~ N(0, 1)
      2. mu_theta, var, logvar, pred_img = p_mean_variance(image, t)
         1. e_theta, pred_img = self.model_prediction(image, cond)
            1. e_theta = self.model(image)
            2. pred_img = image / sqrt(a_cumprod) - e_theta * sqrt(1 / a_cumprod - 1)
         2. posterior_mean, posterior_var, posterior_log_var = q_posterior(pred_img, img, t)
            1. posterior_mean = pred_mu_t = pred_img * coef1 + image * coef2 
            2. posterior_var = pred_beta_bar_t = (1-a_cumprod_{t-1})/(1-a_cumprod_t) * beta_t
            3. posterior_log = "similar"
      3. img = mu_theta + noise * e^(sqrt(log_var))
      4. cond = pred_img