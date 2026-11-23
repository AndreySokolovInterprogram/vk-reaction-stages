# YouTube cross-sectional comparison (Secs. IV-D and VI-F)

`run_analysis.py` reproduces the YouTube numbers of the paper from an export of the collector database:

| Paper | Value | Line of `key_numbers_youtube.txt` |
|---|---|---|
| Sec. IV-D: channel funnel | 282 selected → 185 with any video → 184 with a backfill video → 180 in the sample | `channel funnel`, `channels …` |
| Sec. IV-D: sample | 12,418 videos (6,626 Shorts), 418 with comments disabled | `analytical sample`, `comments disabled` |
| Sec. VI-F: elasticity w.r.t. views | likes 1.01, comments 0.87 | `elasticity …` |
| Sec. VI-F: overdispersion | α = 0.33 (likes), 0.82 (comments) | `NB2 overdispersion …` |
| Sec. VI-F: Shorts, comments per view | −44% (IRR 0.563), video duration not included | `Shorts vs standard videos …` |
| Sec. VI-F: within-channel correlation | 0.29 | `within-channel Spearman …` |

`key_numbers_youtube.txt` in this folder is the output of the script on the export of 2026-11-14. That run used the
backfill table only, so it prints the last two funnel steps (184 → 180); the first two (282 → 185) need the two
optional tables below and were taken from the original run of 2026-11-14. `nb2_coefficients.csv` (written to
`YT_OUT`) reproduces the coefficients of that run to within 1e-8.

## Data

The data are not public (channel and video identifiers, YouTube API terms). The script needs read-only exports
of the collector tables `yt_channels` and `yt_posts`; no database connection or credentials are used by the code.

1. **Backfill table** (`YT_BACKFILL_FILE`, parquet or csv; required), one row per video:

   ```sql
   SELECT p.post_id, p.channel_id, c.domain, c.size_tier, p.is_short, p.published_at, p.crawled_at,
          p.age_hours, p.views, p.likes, p.comments, p.duration_s
   FROM yt_posts p JOIN yt_channels c ON p.channel_id = c.channel_id
   WHERE c.is_selected AND p.is_backfill AND p.published_at < '2026-11-12 00:00:00+00';
   ```

   `comments` is NULL when comments are disabled. `domain` holds the topical domain (the Russian labels of the
   collector are mapped to the stratum codes of the paper inside the script); `size_tier` is S/M/L/XL.
2. **Selected channels** (`YT_CHANNELS_FILE`, optional): `SELECT channel_id FROM yt_channels WHERE is_selected;`
3. **All collected videos** (`YT_POSTS_FILE`, optional): `SELECT channel_id, is_backfill, published_at, crawled_at FROM yt_posts;`
   Videos crawled after `YT_SNAPSHOT_UTC` (default `2026-11-14T14:30:00Z`, the time of the reported run) are ignored,
   so that later live monitoring does not change the funnel.

Filters of the analytical sample: selected channels, `is_backfill`, published before 2026-11-12 (90-day window),
`age_hours >= 168`, `views >= 100`; comment models additionally drop videos with comments disabled.

## Run

```bash
pip install -r requirements.txt
YT_BACKFILL_FILE=/path/to/backfill.parquet \
YT_CHANNELS_FILE=/path/to/channels.csv YT_POSTS_FILE=/path/to/posts.csv \
YT_OUT=youtube/out python youtube/run_analysis.py
```

Models (NB2, `statsmodels`, standard errors clustered by channel): 1A/2A — likes/comments with log views as an
offset and topical domain, format (Shorts), size tier, log age and log duration; 1B/2B — the same with a free
coefficient on log views; 1A_nodur/2A_nodur — offset models without log duration (the Shorts comparison in the
paper). The audit of the 97 inactive channels (public RSS feeds) is described in the paper and is not part of the
script, because it requires channel identifiers.
