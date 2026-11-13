import pandas as pd, numpy as np
R=r"results\dose_response\\"
p=pd.read_csv(R+"preference.csv")
p=p[(p.comparison=="shortcut vs full object")&(p.reliance_corrected<=50)]
rows=[]
for (b,s),g in p.groupby(['base','shortcut']):
    for m in ['deletion_zero','deletion_blur','deletion_inpaint','insertion_blur']:
        x=g[m].values; n=len(x)
        # binomial-ish 95% band for 100 images: 0.5 +- 1.96*sqrt(.25/100)=0.598
        rows.append(dict(setting=f"{b}/{s}",metric=m,n=n,fpr_gt50=(x>0.5).mean(),fpr_gt60=(x>0.6).mean(),fpr_gt598=(x>0.598).mean(),median_pref=np.median(x)))
df=pd.DataFrame(rows); print(df.round(3).to_string()); df.to_csv(r"docs\audit2\scratch\04_venue_framing\overdetection_band.csv",index=False)
