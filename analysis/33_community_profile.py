"""Profile of one panel community (COMMUNITY_ID - internal VK id without the minus sign): posting rhythm and text
duplicates. Used for a community excluded by the auto-posting rule. Descriptive only."""
import os
import pandas as pd, numpy as np
from common import *
CID = int(os.environ['COMMUNITY_ID'])
p = pd.read_parquet(D+'posts.parquet', columns=KEY+['community_vk_id','published_at','origin','text','text_norm_hash','attachments','is_repost','marked_as_ads','is_pinned'])
q = pd.read_parquet(O+'pools.parquet')[KEY+['modality','views','likes','reposts','comments','drop_basic','pool_ratio']]
g = p[p.community_vk_id==CID].merge(q, on=KEY)
msk = g.published_at.dt.tz_convert('Europe/Moscow')
day = g.groupby(msk.dt.date).size()
hour = g.groupby(msk.dt.hour).size()
print('posts', len(g), 'by origin', g.origin.value_counts().to_dict(), 'modality', g.modality.value_counts().to_dict())
print('period', g.published_at.min(), g.published_at.max(), ' days with posts', len(day))
print('per day (MSK): describe', day.describe().round(1).to_dict())
print('per hour of day (MSK)', hour.to_dict())
gap = g.sort_values('published_at').published_at.diff().dt.total_seconds()/60
print('interval between posts, min: quantiles', gap.quantile([.05,.25,.5,.75,.95]).round(1).to_dict(), ' share exactly on :00/:30?', (msk.dt.second==0).mean().round(3))
print('minute-of-hour top5', msk.dt.minute.value_counts().head(5).to_dict())
th = g.text_norm_hash
print('text dup: unique hashes', th.nunique(), ' share posts whose text hash repeats', th.duplicated(keep=False).mean().round(3), ' share = repeat of earlier', th.duplicated().mean().round(3),
      ' empty text share', (g.text.fillna('').str.strip()=='').mean().round(3))
print('top hash counts', th.value_counts().head(5).tolist())
print('text length median', g.text.fillna('').str.len().median())
print('counters: L>0', (g.likes>0).mean().round(3), ' R>0', (g.reposts>0).mean().round(3), ' C>0', (g.comments>0).mean().round(3), ' V null', g.views.isna().mean())
fin = q[q.pool_ratio].merge(p[KEY+['community_vk_id']], on=KEY)
day.rename('posts').to_csv(O+'step3_community_profile_per_day.csv'); hour.rename('posts').to_csv(O+'step3_community_profile_per_hour_msk.csv')
