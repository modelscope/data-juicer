# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps


def IDDPM(
    timestep_respacing,
    noise_schedule="linear",
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    pred_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000,
    snr=False,
    return_startx=False,
):
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(gd.ModelMeanType.START_X
                         if predict_xstart else gd.ModelMeanType.EPSILON),
        model_var_type=((gd.ModelVarType.LEARNED_RANGE if learn_sigma else
                         (gd.ModelVarType.FIXED_LARGE
                          if not sigma_small else gd.ModelVarType.FIXED_SMALL))
                        if pred_sigma else None),
        loss_type=loss_type,
        snr=snr,
        return_startx=return_startx,
        # rescale_timesteps=rescale_timesteps,
    )
