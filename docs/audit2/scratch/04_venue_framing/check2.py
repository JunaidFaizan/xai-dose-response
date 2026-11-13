import pandas as pd
R=r"results\dose_response\\"
rc=pd.read_csv(R+"responsiveness_corrected.csv")
so=rc[rc.comparison=="shortcut vs full object"]
print(so[['base','shortcut','metric','rho_rc','p_bh','fpr','pref_low','median_gap_low','switch_iso_rc','switch_status']].to_string())
sig=so[(so.p_bh<0.05)&(so.rho_rc>0)]; print("sig",len(sig),sig.rho_rc.min(),sig.rho_rc.max())
ro=rc[rc.comparison=="random vs full object"]; print(ro[['base','shortcut','metric','pref_low']].to_string())
bs=pd.read_csv(R+"behaviour_summary.csv")
b=bs[(bs.comparison=="shortcut vs full object")&(bs.label=="follows_shortcut_causal")]
print(b[['base','shortcut','metric','auroc_pooled_strat','median_model_auroc_strat','calibration_gap','observed_switch_rc']].to_string())
