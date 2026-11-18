"""Shared loading for the main analysis of the VK panel (reads the export only)."""
import json
import numpy as np, pandas as pd
import os
HERE = os.path.dirname(os.path.abspath(__file__))
# Paths come from environment variables (see config.example.env in the repository root):
#   EXPORT_DIR  - panel export (communities/posts/snapshots.parquet, manifest.json);
#   OUT_DIR     - results directory (default analysis/out);
#   TECH_GAP    - CSV of posts affected by the collection gap (optional; step 2 then uses an empty list);
#   VIDEO_VIEWS - CSV of main-video views of clips (optional; without it clips have no exposure).
if 'EXPORT_DIR' not in os.environ:
    raise SystemExit('Set EXPORT_DIR (panel export directory), see config.example.env')
D = os.path.join(os.environ['EXPORT_DIR'], '')
O = os.path.join(os.environ.get('OUT_DIR', os.path.join(HERE, 'out')), '')
TECH_GAP = os.environ.get('TECH_GAP', '')
VIDEO_VIEWS = os.environ.get('VIDEO_VIEWS', '')
os.makedirs(O, exist_ok=True)

def panel_start(s):
    return s.measured_at.min()  # panel start = first measurement

def export_time():
    import json as _j
    return pd.Timestamp(_j.load(open(D + 'manifest.json'))['exported_at'])
SIZES = ['S', 'M', 'L', 'XL']; MODS = ['text', 'image', 'video', 'clip']
KEY = ['vk_owner_id', 'vk_post_id']

def modality(att_json):
    """Modality from the post's OWN attachments (from_repost excluded): clip > video > image > text.
    clip = a clip/short_video attachment or a video with video_type='short_video'."""
    has_clip = has_video = has_photo = False
    for a in json.loads(att_json) if att_json else []:
        if a.get('from_repost'):
            continue
        t = a.get('type')
        if t in ('clip', 'short_video') or (t == 'video' and a.get('video_type') == 'short_video'):
            has_clip = True
        elif t == 'video':
            has_video = True
        elif t == 'photo':
            has_photo = True
    return 'clip' if has_clip else 'video' if has_video else 'image' if has_photo else 'text'

def load():
    c = pd.read_parquet(D + 'communities.parquet')
    p = pd.read_parquet(D + 'posts.parquet', columns=[
        'vk_owner_id', 'vk_post_id', 'community_vk_id', 'published_at', 'first_seen_at', 'origin',
        'is_pinned', 'marked_as_ads', 'is_repost', 'repost_source_owner_id', 'repost_source_post_id',
        'attachments', 'n_photo', 'n_video', 'next_target_index', 'missed_targets', 'deleted_at'])
    p['modality'] = p.attachments.map(modality)
    p = p.drop(columns='attachments')
    p = p.merge(c[['community_vk_id', 'stratum', 'size_bucket', 'members_at_selection']],
                on='community_vk_id', how='left')
    p['part'] = np.where(p.origin == 'live', 'live', 'retro')
    p['eligible'] = ~(p.is_repost | p.is_pinned | p.marked_as_ads | p.deleted_at.notna())
    s = pd.read_parquet(D + 'snapshots.parquet')
    return c, p, s

def last_snap(s):
    """Last measurement with non-null counters (1,340 getbyid measurements have all counters null)."""
    s = s[s.likes.notna()]
    return s.sort_values('measured_at').groupby(KEY, as_index=False).tail(1)
