from pathlib import Path
import re,json,hashlib
import numpy as np
import pandas as pd
import argparse
parser=argparse.ArgumentParser(description='Audit recorded BT4012 submission predictions')
parser.add_argument('--workspace',type=Path,required=True,help='Original competition workspace with data, reports and submissions')
parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
args=parser.parse_args()
ROOT=args.workspace.resolve()
OUT=args.output.resolve()
OUT.mkdir(parents=True,exist_ok=True)
ledger=pd.read_csv(ROOT/'reports/kaggle_submissions.csv')
sample=pd.read_csv(ROOT/'data/sample_submission.csv')
test=pd.read_csv(ROOT/'data/test.csv',usecols=['index','time_step']).set_index('index').loc[sample['index']]
preds={}; audit=[]
for _,r in ledger.iterrows():
 name=re.sub(r'\s+','',r.submission_name)
 field=r.file
 match=re.search(r'\(= (.*?)\)',field)
 local=ROOT/(match.group(1) if match else (field if field.startswith('submissions/') else 'submissions/'+field))
 archive_matches=[p for p in (ROOT.parent/'Submissions to Kaggle').glob('*.csv') if ''.join(p.stem.split())==name]
 archived=archive_matches[0] if len(archive_matches)==1 else ROOT.parent/'Submissions to Kaggle'/f'{name}.csv'
 path=archived if archived.exists() else local
 assert path.exists(),(name,path)
 d=pd.read_csv(path)
 assert list(d.columns)==['index','target'] and d['index'].is_unique and len(d)==len(sample)
 assert set(d['index'])==set(sample['index'])
 p=d.set_index('index').loc[sample['index'],'target'].to_numpy()
 assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all() and np.std(p)>0
 sha=hashlib.sha256(path.read_bytes()).hexdigest()
 noted=re.search(r'\b[0-9a-f]{64}\b',str(r.notes))
 assert not noted or noted.group()==sha,(name,'hash mismatch')
 equivalent=None
 if local.exists():
  q=pd.read_csv(local).set_index('index').loc[sample['index'],'target'].to_numpy()
  equivalent=bool(np.array_equal(p,q));assert equivalent,(name,'local archive differs')
 preds[name]=p
 audit.append(dict(submission=name,public_auc=r.public_roc_auc,model=r.model_description,file=str(path.relative_to(ROOT.parent)),sha256=sha,rows=len(p),local_matches_archive=equivalent,source=r.source))
P=pd.DataFrame(preds)
pearson=P.corr();spearman=P.corr(method='spearman')
a=pd.DataFrame(audit).set_index('submission')
a['pearson_vs_1019_6']=pearson['1019-6'];a['spearman_vs_1019_6']=spearman['1019-6']
# Remove between-period shifts by independently ranking scores within each test period.
ranks=P.groupby(test.time_step.to_numpy()).rank(pct=True)
a['within_period_rank_corr_vs_1019_6']=ranks.corr()['1019-6']
a['public_gap_to_best']=a.public_auc-a.loc['1019-6','public_auc']
a.sort_values('public_auc',ascending=False).to_csv(OUT/'all_submissions_audit.csv')
pearson.to_csv(OUT/'pearson_all.csv');spearman.to_csv(OUT/'spearman_all.csv')
print(a.sort_values('public_auc',ascending=False)[['public_auc','pearson_vs_1019_6','spearman_vs_1019_6','within_period_rank_corr_vs_1019_6']].to_string())
(OUT/'methodology.json').write_text(json.dumps(dict(rows=len(P),submissions=len(a),periods=sorted(map(int,test.time_step.unique())),alignment='sample_submission index; unique identical ID sets',public_scores='User-reported; user confirms ledger complete; private scores unknown',correlation='Pearson probabilities; Spearman average ranks; within-period percentile-rank Pearson pooled over rows',limitation='Test labels unavailable: prediction similarity is not error correlation or private-score evidence'),indent=2))
