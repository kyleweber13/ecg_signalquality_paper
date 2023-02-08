import pingouin as pg

# One-way ANOVA: SNR grouped by posture_use
anova = pg.anova(data=data.df_epoch_med[~data.df_epoch_med['posture_use'].isna()], dv='snr', between='posture_use', detailed=True)

# One-way RM ANOVA: SNR grouped by posture_use
anova_rm = pg.rm_anova(data=data.df_epoch_med[~data.df_epoch_med['posture_use'].isna()], dv='snr', within='posture_use', subject='full_id', detailed=True)

# posthoc for anova_rm --> not functional yet; requires multiple participants
ttests = pg.pairwise_tests(data=data.df_epoch_med, dv='snr', within='posture_use', subject='full_id', parametric=True, padjust='none', effsize='cohen')

lr = pg.linear_regression(X=data.df_epoch_med[['wrist_avm']], y=data.df_epoch_med['snr'])