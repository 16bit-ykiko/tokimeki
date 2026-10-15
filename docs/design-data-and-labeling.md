# Data, names and voice labeling: a design proposal

> **Status: a proposal for review, nothing is built yet.** Written 2026-10-16 against `main` 5e63df1 and the layout applied on `refactor/layout` (58a6546). The questions for you are marked **Q1 … Q12** and collected in [§8](#8-questions-for-you). Mockup: [labeling page](design/labeling-episode.svg).

## Summary

Three changes that fit together:

1. **Your work is kept in small text files, and the database is a cache.** Names, aliases, voice labels, face fixes, MAD plans and song recipes live in `.tokimeki/work/` of each franchise, a local git repository you back up. `library.db`, frames, stems and video proxies are rebuilt from the media plus those files. Today the same work sits in SQLite rows keyed by ids that a redo deletes.
2. **One character registry per franchise** (`work/cast.toml`). Each character has a stable id (`momo`), the name you see (梦梦), names per language (モモ・ベリア・デビルーク), aliases per source (茉茉 in the subtitles, `momo_velia_deviluke` in WD14), her voice actor and the series she is in. Face clusters, speaker clusters, labels, voice banks and plans store the id; any name or alias works as input.
3. **A labeling page in the web UI.** The episode plays with a timeline under it; every line comes pre-filled with who the AI thinks says it; keys label as it plays, and the video stops only where the AI is unsure. Your label always wins. Where the AI disagrees with you the line is flagged, never changed. Other views list the lines that need you, one character's lines across the franchise, and the unnamed speaker clusters, and a board tracks progress.

The data is organised as **data root → franchise → series → episode**. There is one library per franchise: To LOVE-Ru's TV series, Motto, Darkness, Darkness 2nd, the OVAs and the OADs share one cast and one Momo voice bank. Episodes are addressed by keys like `darkness-2nd/03`, never by row ids or absolute paths.

---

## 1. Where things stand

Measured on the Darkness library (S1 01–05 through speakers, schema v6; the ingest of 06 and 07 was running meanwhile), opened read-only:

| Finding | Where | Consequence |
|---|---|---|
| Characters are free strings matched by equality | `clusters.name`, `speakers.name`, the plan's `"character": "梦梦"`, `named_faces(conn, episode, name)` | 茉茉, モモ and `momo_velia_deviluke` have nowhere to go (only the README mentions them). Renaming 梦梦 means editing two tables and every plan. |
| Human work is keyed by row ids that a redo deletes | `speaker_fixes`, `transcripts` and `voice_styles` (by `line_id`) cascade when `--redo lines` re-inserts the lines; `cast_fixes` (by `shot_id`) cascade when `--redo shots` does | The docs already say fixes "last until the episode's shots are redone" |
| An episode is identified by its relative path, and its id by registration order | `episodes.path`; episode id 4 is `[05]` and id 5 is `[04]` | Moving or renaming a file makes it a new episode, and its whole analysis is lost |
| Outputs hold absolute paths | the plan's `series`, the song's `source.path` and `inputs.lyrics` | Moving the media breaks plans and song analyses |
| Precious and regenerable data are mixed | `library.db` is 73 MB of embeddings around 13 cluster names, 7 speaker names and 3 face fixes. Backups are hand copies (`library.db.bak-v4`, `-v5`). The 170 hand-checked ep01 speakers are in `scratch/labels_ep01.txt`, keyed by line id. | Nothing says what to back up, and the most valuable file sits in `scratch/` |
| There is one library per directory | Darkness S1, S2 and the OADs share a library; Motto would get its own | Momo would be named twice and her voice bank split in two (`voice/paths.py` puts banks under the series) |
| Media is scattered | Songs are in both `~/anime/music/` and `to-love-ru-darkness/music/` (rakuen-project is in both), subtitle packs in `_subs/`, unpacked tracks in `.tokimeki/subs/`, fonts copied per series (31 MB), analyses in `~/.local/share/tokimeki/songs` | |
| The episodes cannot play in a browser as they are | HEVC Main10 in MKV with FLAC audio; `voice serve` (`SimpleHTTPRequestHandler`) has no HTTP Range support | A labeling page needs a browser-playable proxy and a server that answers Range requests |

What labeling has to cover, per episode: 265–320 dialogue lines. Of these, 186–284 have a voice in kept shots; the rest fall in dropped shots (about one shot in six is dropped) or the OP/ED. Then:

- 115–208 voices have a named speaker;
- 37–117 have an unnamed one (up to 4.5 min on ep05);
- 4–18 have no speaker at all;
- 7–13 are overlap, short or mixed.

The unnamed clusters are most of the work, not the 14% the ep01 check suggests.

## 2. Organising the data

### 2.1 Levels

| Level | What | Example | Id |
|---|---|---|---|
| Data root | all your media and tokimeki's data | `~/anime` | `$TOKIMEKI_DATA` |
| Franchise | one cast: one library, one registry, one set of voice banks | To LOVE-Ru | `to-love-ru` |
| Series | one season, or a set of OVAs, in one directory | Darkness 2nd | `darkness-2nd` |
| Episode | one video file | Darkness 2nd episode 3, its SP | `darkness-2nd/03`, `darkness-2nd/sp` |

The franchise is the unit because the cast is. Momo is the same character in Motto, Darkness and Darkness 2nd: her face clusters and her voice should be named once, and her voice bank should draw from all of them (filtered by series where her voice changed over the years). A series has its own OP/ED, numbering and art style. An episode is the unit of work and of progress.

### 2.2 Four kinds of data

| Kind | Examples | Size | Rebuilt from | Back up | Git |
|---|---|---|---|---|---|
| Media (yours) | episodes, subtitle tracks, CD rips | 100s of GB | nothing: you have it | your own way | never |
| **Work** (yours, made here) | registry, voice labels, face bindings and fixes, plans, song recipes, progress | KB to MB | nothing: precious | always | its own private repository (not the code's) |
| Voice data | banks, voice models, samples | GBs | banks from work + media; models take hours of training | locally (the models) | never; never leaves the machine |
| Cache and outputs | `library.db`, frames, stems, proxies, peaks, clips, renders, reports | GBs | media + work (the library takes the whole ingest, about 20 min of GPU per episode) | no; `library.db` is snapshotted before migrations | never |

The rule: **nothing in `cache/` or `out/` is the only copy of a decision you made.** A redo, a schema migration or a deleted cache never loses work.

### 2.3 On disk

```
~/anime/                                    $TOKIMEKI_DATA: your media and tokimeki's data
  to-love-ru/                               a franchise
    tv/ ova/ motto/ darkness/ darkness-2nd/ darkness-oad/     your episodes, one directory per series
      [VCB-Studio] To LOVE-Ru Darkness [01][Ma10p_1080p][x265_flac_aac].mkv
      [VCB-Studio] To LOVE-Ru Darkness [01][Ma10p_1080p][x265_flac_aac].ass       main subtitle track
      zh-hant/[VCB-Studio] To LOVE-Ru Darkness [01][Ma10p_1080p][x265_flac_aac].ass   another track
    music/momo-single/ music/rakuen-project/                  the franchise's songs (CD rips, cue sheets)
    archives/                               subtitle packs as you got them (.rar)
    .tokimeki/
      work/        ★ your work: small text files, a git repository; back it up
        franchise.toml                      series, how files map to episode keys, subtitle tracks
        cast.toml                           the character registry (§3)
        labels/darkness/01.json …           voice labels, one file per episode (§4.4)
        faces.json                          face cluster bindings (by exemplar faces) and per-shot fixes
        plans/momo-ep1.json                 MAD plans (and their notes, e.g. momo-ep1.prompt.md)
        banks/momo.toml                     a voice bank's policy and style reference lines
      voices/      ♪ voice data per character id: never shared, back it up locally
        momo/bank/ momo/models/gpt-sovits/ momo/samples/
      out/         renders: mads/<name>/, report/; rebuilt from work/ + media
      cache/       safe to delete: library.db, frames/, voice/ (stems), proxy/, peaks/, motion/, clips/, assist/
      logs/
  music/ngnl-zero/                          songs that belong to no franchise
  .tokimeki/                                shared by every franchise
    work/songs/<id>.toml                    song recipes: source file, track or window, lyrics, excerpt
    cache/songs/<id>.json                   song analyses
~/.local/share/tokimeki/                    $TOKIMEKI_HOME: programs, not data
  engines/ bin/ fonts/                      TTS engines and their environments, net-guard, subtitle fonts (one copy)
```

- Media directories are only read. tokimeki writes only inside `.tokimeki/`. Labels are sidecars in that sense: kept apart from the video, never inside it and never beside it (**Q2**).
- Release filenames are kept: subtitle tracks match a video by basename, and `franchise.toml` maps each file to its key (**Q4**).
- `cache/` can be deleted at any time, and so can `out/` (renders come back from `work/plans/`).
- Paths inside work files are relative: to the franchise root, or for songs to the data root. Nothing stores an absolute path.
- In code, one place builds these paths (`library/paths.py`: `DataRoot`, `FranchisePaths` with `work`, `voices`, `out`, `cache`), as `SeriesPaths` does today.

### 2.4 `franchise.toml`

```toml
schema = "tokimeki.franchise/1"
id = "to-love-ru"
title = { ja = "To LOVEる -とらぶる-", en = "To LOVE-Ru" }

# One table per series, in viewing order. `dir` defaults to the id. `episodes` is a regular
# expression whose group is the episode number; VCB-style [01], [OAD01], [SP] and S01E01 work
# without it.
[[series]]
id = "darkness"
title = { ja = "To LOVEる -とらぶる- ダークネス" }
year = 2012

[[series]]
id = "darkness-2nd"
title = { ja = "To LOVEる -とらぶる- ダークネス 2nd" }
year = 2015
episodes = '\[(\d{2}|SP)\]'

[[series]]
id = "darkness-oad"
episodes = '\[OAD(\d{2})\]'

[[series]]
id = "motto"
title = { ja = "もっとTo LOVEる -とらぶる-" }
year = 2010

# Subtitle tracks beside the one next to the video, by an id that aliases can name (§3.1).
[subtitles.vcb-hans]
lang = "zh-Hans"
path = "{stem}.ass"
[subtitles.vcb-hant]
lang = "zh-Hant"
path = "zh-hant/{stem}.ass"
```

A video under the franchise root that no series claims shows up on the board as "unassigned", for example a new download, and nothing is ingested until it gets a key.

### 2.5 One library per franchise

Today's five episodes hold 19k faces and 15k sampled frames in 73 MB. A franchise of about 80 episodes would be about 1.2 GB, which is fine for SQLite (WAL, one writer). Clustering costs the same per new episode, since it runs against at most 48 exemplars per cluster.

The alternative is a library per series plus a shared registry. The files would be smaller, but cross-series work (Momo's bank, a MAD across Motto and Darkness) would need `ATTACH` everywhere, and clusters would again be named per library (**Q1**).

### 2.6 Stable references

- **An episode** is its key (`darkness/01`) plus a fingerprint: size, duration and a hash of the first 16 MB. A moved file with the same fingerprint keeps its analysis; only the stored path changes. A replaced file (another release) is flagged and its labels re-timed. The offset comes from correlating the labelled speech with the new file's voice activity, the same check the lines stage makes for subtitles.
- **A time** is seconds in the episode, 3 decimals, rounded to the 10 ms voice frame.
- **A shot or a line** is its episode plus its time range. Row ids stay inside the database and in short-lived API responses.

| Today | Becomes |
|---|---|
| Plan: `series` (absolute path), `character` (a name), slot `shot` and voice `line` (row ids) | Plan v4: `franchise` and `character` ids; slots and voices by episode key + `in`/`out`, which plans already carry; ids become optional hints |
| Song: `source.path` and `inputs` (absolute) | `work/songs/<id>.toml`, a recipe with paths relative to the data root; the analysis in `cache/songs/` |
| `cast_fixes(shot_id, from_cluster, to_cluster)` | `work/faces.json`: per shot time range, from character to character |
| `speaker_fixes(line_id)`, `scratch/labels_ep01.txt` | `work/labels/<series>/<nn>.json` |
| Cluster and speaker names | `cast.toml` plus bindings (§3.4) |

### 2.7 Backup and export

- `work/` is a git repository of its own. The server commits to it when an episode is marked done, and at most every 10 minutes while you label ("labels: darkness/01, 34 changed"), so a bad edit can always be undone. Whether it is pushed anywhere is up to you (**Q3**). It holds ids, names and times: no media, no subtitle text, no audio.
- Every work file names its format (`schema = "tokimeki.labels/1"`), each format has a JSON Schema in `docs/formats/`, and readers accept older versions. `tokimeki export to-love-ru` writes a zip of `work/`.
- `library.db` is snapshotted to `cache/snapshots/` before each schema migration, which replaces the hand-made `.bak-v4` and `.bak-v5`.
- Voice models: copy `voices/*/models/` to another local disk. They take hours to train, and they never go to a cloud drive.

### 2.8 The hard rules in this layout

- Media and voice data never enter a git repository. `work/` holds neither, and a test rejects any audio, video or image file under it.
- The content filter stays first. Dropped shots get no frames, no picture in the proxy (black), no voice (the stem is silent there), no segments, and nothing from them reaches the agent.
- The server listens on 127.0.0.1 only, so voice clips never leave the machine. The screenshot service for the phone gets a listener of its own, with only its own route.

## 3. Names: the character registry

### 3.1 `cast.toml`

```toml
schema = "tokimeki.cast/1"
language = "zh-Hans"            # the language of display names; the UI shows Japanese beside them

[characters.momo]               # the id: ASCII, never changes, used in paths and files
display = "梦梦"                 # what you call her, shown everywhere
color = "#e85d9c"
key = 1                         # her hotkey on the labeling page
names = { ja = "モモ・ベリア・デビルーク", ja-short = "モモ", en = "Momo Belia Deviluke" }
voice = { ja = "豊崎愛生", en = "Aki Toyosaki" }
series = ["tv", "motto", "darkness", "darkness-2nd", "darkness-oad"]   # as you listed; the UI also shows where she was found
note = "Deviluke's third princess, Nana's twin"                        # context for the agent
aliases.subtitles = ["茉茉"]                     # in any subtitle track
aliases.wd14 = ["momo_velia_deviluke"]
aliases.speech = ["モモ"]                        # what the Japanese audio calls her (transcripts)

[characters.yami]
display = "小暗"
names = { ja = "金色の闇", en = "Golden Darkness" }
voice = { ja = "福圓美里", en = "Misato Fukuen" }
aliases.subtitles = ["小暗"]
aliases."subtitles:vcb-hant" = ["小闇"]           # only in that group's track
aliases.wd14 = ["konjiki_no_yami"]

[characters.rito]
display = "梨斗"
names = { ja = "結城リト" }
voice = { ja = "渡辺明乃", en = "Akeno Watanabe" }

[names]
ignore = []                     # strings that look like names but name nobody
```

- The **id** is what every file and table stores. Renaming 梦梦 changes one line in this file, nothing else.
- **Aliases are per source.** `subtitles` applies to every subtitle group; `subtitles:<track id>` applies to one group, for when two groups disagree (zh-Hans 小暗, zh-Hant 小闇 in the two VCB packs). `wd14` holds tagger tags, `speech` names in the Japanese audio, and `lyrics` names in song credits and lyric files. A cue sheet's `PERFORMER "モモ・ベリア・デビルーク(CV:豊崎愛生)"` already matches through `names.ja` once the CV part is stripped.
- `other` is reserved as an id: it marks a minor character not worth naming.
- **Q8:** romaji ids, your Chinese names for display, Japanese beside them.

### 3.2 Where names come from

| Source | What it says | How much it is worth | Seen here |
|---|---|---|---|
| You: CLI, the picker, plans, comic requests | any name or alias | exact | 梦梦, momo, 茉茉, モモ |
| Subtitle text | who is addressed or talked about | evidence about the listener, not the speaker | 茉茉 appears in 69 lines over six episodes |
| A subtitle's Actor field | the speaker, when a group fills it in | a strong pre-fill for speakers | VCB writes the placeholder `NTP`, so not here |
| WD14 character tags | a face's name | a hint for a face cluster | `momo_velia_deviluke` on 2,099 frames, `konjiki_no_yami` on 1,562; Nana has no tag; noise such as `hinata_hajime` on 55 |
| Japanese transcripts | names in the audio | as subtitle text | モモ |
| Song credits, lyric files | who sings | binds a song to a character | cue `PERFORMER` |
| Comic scripts (vision model) | speaker names as the translation writes them | resolved like your input | |

### 3.3 Resolving a name

Names are compared after folding: NFKC, case folding, and `・ · . _ -` and spaces dropped. They are stored as written.

1. **Exact.** The folded string is looked up among every character's id, display name, names and aliases (scoped to the source when one is known). One hit resolves it.
2. **Honorifics.** If nothing matched, Japanese さん, ちゃん, 様, 先輩 and similar endings are stripped and the lookup runs again.
3. **Several hits** make the name ambiguous, for example two characters both called 会长 in different series. Context decides when there is some (the series, who is on screen); otherwise you are asked.
4. **No hit** leaves the name unresolved. It goes to an inbox with suggestions and is never applied on its own.

The suggestions for an unknown name, best first:

- **It is written in a Japanese name:** 美柑 is part of 結城美柑, and 春菜 of 西連寺春菜.
- **It sounds like one:** the pinyin of a Chinese name against the romaji of a Japanese name or the English name. 茉茉 is mòmò and モモ is momo; 娜娜 is nana, like ナナ; 菈菈 is lala, like Lala. (`pypinyin`, and a kana table.)
- **Who is on screen when it is said**, against their base rate: the person addressed is often in the shot. This is the lift `voice/speakers.py: matches` already computes for speakers.

Measured on eight cast names as the Darkness zh-Hans subtitles write them, the first two settle five of them (茉茉, 娜娜, 菈菈, 美柑, 春菜). The on-screen lift puts the right character in the top three for all eight but first for only three: for 茉茉 it ranks 娜娜 first, because the twins share the screen. That is why the lift only suggests and never decides.

How 茉茉 gets resolved, step by step:

| When | What happens |
|---|---|
| Before | The line shows 茉茉 as text; nothing resolves and nothing breaks. The resolver notes unknown names it is asked about: you type 茉茉 in the picker or pass `--character 茉茉`, or the agent reports it as a mention. |
| Suggestion | The inbox shows "茉茉 (zh-Hans subtitles, 69 lines): sounds like モモ → 梦梦 (momo); on screen: 娜娜, 梦梦, 梨斗". |
| You accept | One key. `cast.toml` gets `aliases.subtitles += "茉茉"` (written by `tomlkit`, which keeps comments and layout), and `work/` gets a commit. |
| After | `plan auto --character 茉茉` works; the picker finds her by 茉, mo or モ; subtitle lines carry `mentions: ["momo"]`, which the agent and `plan context` read as "a line said to or about her". |

WD14 tags reach the same inbox from the other side: "`momo_velia_deviluke` is on 82% of 梦梦's solo faces. Add it as her alias?"

### 3.4 What binds to a character

```
                        cast.toml  ·  momo  (梦梦 · モモ · 茉茉 · momo_velia_deviluke)
                                         ▲  character id
   ┌──────────────────┬──────────────────┼──────────────────┬───────────────┬───────────────┐
 face clusters     speaker clusters    voice labels       voices/momo/     MAD plans       comic scripts
 #3 …              #136 …             who = "momo"       bank, models,    character =     speaker names,
 bound by you,     bound by you, or    (work/labels)      samples, one     "momo"          resolved by
 kept by exemplar  by your labels'                        per franchise                    alias
 faces             majority
```

- **Face clusters → character**, many to one: Nana with her hair down can stay its own cluster, bound to `nana`, with no merge needed. The binding is kept in `work/faces.json` by a few exemplar faces (episode, sampled frame, box): the faces you saw when you bound the cluster. After any re-clustering, the cluster holding an exemplar face is bound again; when one cluster holds exemplars of two characters, the cast page shows the conflict.
- **Speaker clusters → character**, many to one. The `char` embedding splits Momo by manner (her scheming monologue, her polite voice). Those clusters stay apart and both bind to `momo`, which keeps them as style groups for the bank. A binding is kept by the labels it rests on (§4.3), so it survives a rebuild.
- **Labels, banks, plans and comics** store the id.
- **The database** mirrors the registry (`characters`, `character_names`) and gets `character_id` on clusters and speakers. `clusters.name` and `speakers.name` go away after the migration.

### 3.5 Where the registry lives and how you edit it

It lives in `work/cast.toml` of the franchise: versioned with your other work, readable in any editor, small. There are three ways to edit it, all writing the same file:

- **The cast page in the web UI:** names, aliases, colours and hotkeys; the inbox of unresolved names and WD14 hints; the face and speaker clusters with their bindings.
- **The CLI:** `tokimeki cast alias to-love-ru momo 茉茉 --source subtitles`, `tokimeki cast resolve to-love-ru 茉茉`, `tokimeki cast bind to-love-ru 3 梦梦`. Any alias works where an id is expected.
- **A text editor.** The server notices the change (by hash), validates the file and re-imports it. Validation requires unique ids, no alias claimed by two characters in the same source, and slug-safe ids; `tokimeki cast check` does the same.

**Q9** is about the series list itself.

## 4. Voice labeling

### 4.1 Segments and labels

- **An AI segment** is today's `line_voices` row: one dialogue line inside kept shots and outside the OP/ED, trimmed to where its voice is in the stem. Its voice status (`clean`, `overlap`, `short`, `long`, `mixed`) comes along; `silent` lines get no segment. For a `mixed` line the AI also proposes a split point, where the two halves' embeddings differ most.
- **A label** is yours: a time range plus a verdict.

| `kind` | `who` | Meaning | Used for |
|---|---|---|---|
| `speech` | a character id | she says it | banks, MADs, clustering |
| `speech` | `other` | someone minor, not worth naming | kept out of banks |
| `noise` | | not speech: a breath, an effect, singing, a mistaken cut | excluded |
| `overlap` | | two voices at once | out of banks; MADs may still use it |
| `unsure` | | you will come back to it | stays in the queue |

- **Labels cover AI segments.** A label covers an AI segment when they overlap by at least half of the shorter one. An episode's segments, as every view shows them, are your labels plus the AI segments no label covers. Splitting, merging or moving an edge writes labels. When the AI segments change (a redo, a new trimming), your labels stay where they are; newly uncovered AI segments show up as new work.

### 4.2 What the AI fills in

| On the page | From | Shown as |
|---|---|---|
| Segments and their bounds | `line_voices` (the voice trimmed inside the subtitle times) | blocks on the timeline |
| The speaker | the speaker cluster's character | an outlined chip; dashed when the AI is unsure |
| How sure it is | distance to the cluster's centre, margin to the next character, agreement of the nearest labelled voices | solid or dashed; the reason when it needs you |
| Who is on screen | face clusters per shot (framing, presence) | the on-screen lane, and close-ups called out |
| A close-up seed | one named character alone in a close-up | "picture: 梦梦" next to the speaker |
| Text | the subtitle line (zh), and the Japanese transcript once ASR runs | the panel and the lines lane |
| The agent's guess | §4.9, only for lines the clustering left | a third chip, "agent: 美柑 · Rito's sister answers" |
| Mentions | the registry (§3.3) | "to: 梦梦" when a line names her |

A line **needs you** when:

- it has no speaker (a lone voice, or an unbound cluster);
- it is doubtful: far from its cluster's centre, or nearer another character's labelled voices;
- the picture disagrees: another character is alone in a close-up. That is often inner monologue, which is still worth a look;
- its status is short, mixed or overlap;
- the AI now disagrees with your label (a conflict);
- the agent made a suggestion;
- its episode's media changed and the labels were re-timed.

### 4.3 AI and you

- **You win.** The AI never edits or removes a label. If it later disagrees with one (after a recluster, say), the line goes to the queue as a conflict, showing both answers.
- **Your labels teach the clustering:**
  - **Pins.** A labelled voice is fixed to its character and no clustering moves it. This replaces `speaker_fixes`.
  - **Naming.** A cluster is bound to the character most of its labelled voices belong to (at least 2 labels and 60%). That evidence ranks above close-up seeds and the on-screen lift.
  - **Neighbours.** A voice left out by clustering is suggested the character of its labelled neighbours when 4 of its 5 nearest agree. This is today's `_fill`, weighted toward your labels.
- **Binding a whole cluster** by ear (the speaker view, §4.6) labels only the clips you heard, as `by: "cluster"`, and binds the cluster in the database. If the library is rebuilt, those labels name the new cluster again. The rest of the cluster's lines stay AI answers, which is the honest state.
- **Accuracy.** Every `key` label records what the AI said at the time (`ai`). Per episode, `tokimeki voice check` and the board report the agreement. The ep01 figure (145 of 146 named lines right) becomes a number that updates as you label, and it tells you whether the next season needs watching or only the queue.

### 4.4 The label file

`work/labels/darkness/01.json` holds one segment per line, so git diffs stay readable:

```json
{
  "schema": "tokimeki.labels/1",
  "episode": "darkness/01",
  "media": {"bytes": 1288490188, "duration": 1422.62, "head": "sha256:9f2c41d0"},
  "watched_to": 812.4,
  "done": false,
  "segments": [
    {"start": 70.41, "end": 72.08, "kind": "speech", "who": "rito", "by": "watch"},
    {"start": 75.12, "end": 75.60, "kind": "speech", "who": "momo", "by": "key", "ai": null},
    {"start": 77.57, "end": 78.31, "kind": "speech", "who": "rito", "by": "key", "ai": "rito"},
    {"start": 78.31, "end": 79.10, "kind": "speech", "who": "momo", "by": "key", "ai": "rito"},
    {"start": 80.02, "end": 80.61, "kind": "noise", "by": "key"},
    {"start": 92.10, "end": 94.00, "kind": "speech", "who": "other", "by": "cluster"}
  ]
}
```

- `by` records how you gave the label:
  - `key`: you pressed a key;
  - `watch`: you watched a line the AI was sure of go past without objecting (**Q5**);
  - `cluster`: you bound its cluster after hearing this clip.
- `ai` is what the AI said at that moment (the third and fourth segments are a split: Rito, then Momo).
- The file holds no subtitle text: it is derivable, and it belongs to the subtitle authors, not to you.
- All writes go through one module that writes the file atomically and then the database mirror. If the file changes underneath (an editor, `git checkout`), its new hash makes the mirror re-import it. Two tabs editing one episode are kept apart by a revision check (HTTP 409).

### 4.5 Progress

Each segment is in one of these states:

- **ai**: you have not seen it;
- **watched**: it went past in Watch mode, the AI was sure, and you did not object;
- **confirmed**: you agreed with the AI;
- **corrected**: you chose differently;
- **excluded**: noise or overlap;
- **unsure**.

An episode is *untouched*, then *in progress* (watched to mm:ss, n lines need you), then *done*, which you mark with Ctrl+Enter. You can mark it done with unsure lines left; they stay in the queue. `watched_to` and `done` are kept in the label file, and the rest is counted from the segments.

```
 To LOVE-Ru · progress                       reviewed   need you   AI agreed   梦梦    娜娜    梨斗
 darkness       01  ██████████  done          234/234       0         97%      2.3m   1.1m   1.5m
                02  ██████▌░░░  12:40         182/284      31         95%      1.9m   0.8m   1.2m
                03  ░░░░░░░░░░  AI only         0/186      44          —        —      —      —
 darkness-2nd   01  ░░░░░░░░░░  not ingested
 motto          01  ░░░░░░░░░░  not ingested
```

The board's character columns count minutes of reviewed speech: the voice-bank budget at a glance.

### 4.6 Views

**Episode page (along the timeline).** This is the main view; see the [mockup](design/labeling-episode.svg). The video is on the left, with a speaker tag before the subtitle in the speaker's colour, so a wrong speaker stands out as the line plays. The panel on the right shows the selected line: times, text, the AI's answer and why, who is on screen, your label, and the hotbar. Under them is the timeline with one lane each for:

- the shots, as thumbnails, with dropped ones hatched;
- who is on screen, as coloured bars per character;
- the voice stem's waveform;
- the AI's speakers, as outlined chips;
- your labels, as filled chips;
- the subtitle lines.

It has two modes:

- **Watch:** the video plays and you label as it goes. The selected line switches to a new line 0.5 s after it starts, so a key pressed as a line ends still lands on that line. Lines that need you pause the video at their end. Lines the AI is sure of go past and are recorded as `watch` if you do not object. OP/ED, dropped stretches and gaps over 3 s are skipped.
- **Step:** you go line by line, and each line plays when you reach it. A key labels it and moves on. Best for the queue and for fixing bounds.

The lines quoted in these mockups are made up.

**Queue (what needs you).** The lines from §4.2, for one episode, a series or the franchise, grouped by reason. It runs in Step mode, with a small picture of the shot beside each line.

```
 Needs you · darkness/04 · 38 lines                                   [reason ▾] [series ▾]
 ▸ 03:12.40  1.1s  「……」                no speaker        on screen 美柑, 梨斗    agent: 美柑
   03:15.02  0.5s  「欸？」               short             on screen 梨斗
   07:48.91  2.6s  「你们两个都给我……」     mixed (split 1.3s?)  voice 娜娜 → ?
   11:02.10  1.8s  「我才没有……」          picture disagrees voice 梨斗 · close-up 梦梦
   …
```

**Speaker clusters (by uncertain cluster).** This replaces the static listening page (`voice page` and `voice serve`). The unbound clusters are sorted by minutes, each with its six most typical and three least typical clips and its evidence (seeds, on-screen lift, neighbours). A digit binds the cluster: the clips you heard become `cluster` labels and the rest of its lines follow. S splits a cluster more tightly; selecting clips and pressing a digit moves just those.

```
 Unbound speakers · 86 clusters · 17.6 min                    ▸ #164  43 lines · 1.8 min
   suggestion: 美柑 (on screen 41%, lift 3.1) · seeds: none
   ▶ 「梨斗，饭好了哦」      ▶ 「真是的」      ▶ 「……」        typical
   ▶ 「欸——？」             ▶ 「不知道啦」                    least typical
   [8] 美柑   [/] other name   [0] someone minor   [S] split   [→] next cluster
```

**Character (per character).** All of one character's segments across the franchise, least typical first: the place to check a bank's purity before training. Its filters are series, source (yours, AI only, agent) and status. X sends a line back to `unsure`, and a digit moves it to another character. The header shows her minutes (total, reviewed, bank-eligible) and whether her bank is stale.

**Board** (§4.5) and **cast page** (§3.5) complete the set.

### 4.7 Keys

Keyboard first; the mouse works too: click to select, drag an edge, double-click to play, the wheel to zoom, drag the ruler to scrub.

| Key | Action |
|---|---|
| **Playback** | |
| Space | play / pause |
| R | replay the selected line |
| Shift+R | replay it with 2 s of context on each side |
| ← / → | previous / next line (→ is also "skip") |
| ↑ / ↓ | previous / next line that needs you |
| Shift+← / → | back / forward 2 s |
| − / = | slower / faster (0.75, 1, 1.25, 1.5, 2×) |
| B | hear the voice stem or the full mix |
| A | auto-pause at lines that need you, on / off |
| Tab | Watch / Step |
| **Labels** | |
| 1 – 9 | this character (the hotbar), then move on |
| Enter | accept the AI's (or the agent's) answer, then move on |
| / | any character, by any name or alias: 梦梦, 茉茉, momo, モモ |
| 0 | someone minor (`other`) |
| N | not speech |
| O | two voices at once |
| U | unsure |
| Backspace | remove my label (back to the AI) |
| Ctrl+Z / Ctrl+Shift+Z | undo / redo |
| Ctrl+Enter | mark the episode done |
| **Segments** | |
| S | split at the playhead |
| M | merge with the next line |
| Z / X | start 20 ms earlier / later (Shift: 100 ms); plays the 0.6 s around the edge |
| C / V | end 20 ms earlier / later (Shift: 100 ms) |
| + | new segment at the playhead, from the voice activity there (shouts, laughs, untranslated lines) |
| ? | this table |

The hotbar holds the nine characters with the most voice in the series, unless `key` in `cast.toml` pins them. Keys are read at the page level. While a Chinese IME is composing they are ignored, and a banner asks for English input; the picker (/) is the one text field, and it takes Chinese.

### 4.8 How long it takes

Take ep01: 295 dialogue lines, 234 voices, 10.4 min of dialogue in a 23:42 episode.

- **Watch** at 1.25×, skipping the OP/ED, dropped shots and gaps, pausing at 20–60 lines: about 10–12 min an episode.
- **Speaker view, then queue:** bind the 5–10 unbound clusters that matter (about a minute), then 20–60 queued lines at about 4 s each: 3–5 min an episode.

A 12-episode season is about an hour through the queue, or 2–2.5 h watched in full. The agreement numbers on the first fully watched episodes tell you which of the two the rest needs.

### 4.9 Agent assist for the leftover short lines

Clustering leaves 4–18 lone voices per episode, about a second each, plus the `short` ones. Too little voice to embed reliably. Cheap local evidence comes first, with no cloud and no GPU:

- **Neighbours** (§4.3).
- **Turn-taking:** in an A, ?, A exchange where only A and B are on screen, the gap is probably B.
- **Mentions:** a line that names 梨斗 is rarely said by Rito.

What is still open goes to an agent, **one scene at a time**. It gets:

- the scene as a script with the known speakers filled in and the gaps marked, with times;
- for each gap: who is on screen and how (close-up, wide), two keyframes at 512 px from **kept frames only**, the voice's three nearest characters with distances, and its mentions;
- the registry's names and notes (relationships help: Mikan is Rito's sister).

It answers in a schema-checked JSON: per gap, `who` (an id, `other` or `unknown`), `confidence` (low, medium or high) and a reason of at most 15 words. The answers are suggestions shown as agent chips; Enter accepts one, which writes your label. They are never labels by themselves.

| Rule | How |
|---|---|
| No bursts | one Message Batches job per episode (asynchronous, half price), never parallel sessions, a daily request cap |
| Resumable | each scene's request has a `custom_id` and an input hash; results are cached in `cache/assist/`, and a rerun sends only what changed |
| The content filter first | only kept frames and lines in kept shots can be sent; the request builder checks shot status itself |
| Your call | `tokimeki voice assist darkness/04 --dry-run` shows the count of scenes and lines and the estimated cost before sending |
| Measured | before it is turned on, run it on ep01 against your 170 checked lines; keep only answers whose confidence held up there |

Cost: about 20–30k input tokens an episode (two 512×288 keyframes are about 400 tokens per line, plus the script). Through the batch API that is about 2 cents an episode on Haiku 4.5, 4 on Sonnet 5 or 10 on Opus 5, so well under a dollar a season (**Q7**).

### 4.10 Into voice banks and MADs

- **Banks** read resolved segments, not raw clusters. A bank's policy goes in `work/banks/momo.toml`:
  - `strict`: only lines you labelled by key;
  - `reviewed` (the default): also `watch` and `cluster` labels;
  - `all`: also confident AI segments that conflict with nothing.

  Clips are cut at your bounds, and split segments become separate clips; `overlap`, `noise`, `unsure` and `other` are left out. `bank.json` lists each clip by episode key and time with its label's source, so a later label change marks the bank stale on the board ("梦梦: 12 labels changed since export") and a re-export cuts only those clips. Banks can be limited by series (`series = ["darkness", "darkness-2nd"]`) where a voice changed over the years.
- **MADs:**
  - `plan context` lists each voice line with its speaker (an id and its source);
  - `plan auto` lays only lines resolved as hers over the song;
  - validation warns `voice.speaker` when a plan's voice line now resolves to someone else;
  - the MAD report says who speaks.

  Scenes gain their speakers too, so the library can find "scenes where 梦梦 talks to 梨斗".
- **Speaker clustering** improves as you label (§4.3), and the next ingest of a new episode starts from the better clusters.

## 5. How it lands in the code

### 5.1 Packages

| Package | Adds |
|---|---|
| `cast/` | `registry.py`: `cast.toml` read, validated and written with `tomlkit`, and the ids. `names.py`: folding, resolving, suggestions (Japanese names, pinyin and romaji, on-screen lift) and the unresolved inbox. Face cluster bindings by exemplar faces, and face fixes by time. |
| `library/` | `paths.py`: `DataRoot` and `FranchisePaths`. `franchise.py`: `franchise.toml`, series, episode keys and fingerprints. `work.py`: atomic writes of work files, the SQLite mirror (with file hashes), git commits. Schema v7–v9 (§5.2). |
| `media/` | `proxy.py`: the browser proxy (NVDEC, `scale_cuda`, `h264_nvenc`, AAC), black and silent over dropped shots. `peaks.py`: waveform peaks. |
| `anime/` | a `proxy` stage after `voice`: proxy video, voice track, peaks. NVDEC and NVENC only, no model, a few minutes an episode. |
| `voice/` | `segments.py`: AI segments plus labels give the resolved segments and their needs-you reasons. `labels.py`: label operations (set, clear, split, merge, bounds) on the episode file. Speakers named and pinned by labels. `assist.py`: the agent (batch, cache, schema). The bank reads resolved segments. |
| `mad/` | plan v4 (ids and keys), voice lines with speakers, the `voice.speaker` check |
| `server/` | a Starlette app on 127.0.0.1: `/api` routes, each a thin wrapper over an `api.py` function (so the CLI and agents get the same operations); `/media` files with Range; the built `web/dist`. `types.py`: the API's dataclasses. |
| `web/` | React + Vite + TypeScript strict. Pages `Episode`, `Queue`, `Speakers`, `Character`, `Board`, `Cast`; `timeline/` (canvas lanes); `keys.ts` (one keymap, which also draws the `?` sheet); `api/types.gen.ts` (generated). TanStack Query for server data, a reducer for the page's state and undo. |

The labeling page never touches the GPU. Labeling can go on while a voice model trains.

### 5.2 Schema sketch

```sql
-- V7: a library holds one franchise; episodes get stable keys; the registry is mirrored here.
CREATE TABLE series (
    id TEXT PRIMARY KEY,                       -- 'darkness-2nd'
    ord INTEGER NOT NULL,
    dir TEXT NOT NULL UNIQUE                   -- relative to the franchise root
) STRICT;

ALTER TABLE episodes ADD COLUMN series_id TEXT REFERENCES series(id);
ALTER TABLE episodes ADD COLUMN key TEXT;      -- 'darkness-2nd/03'
ALTER TABLE episodes ADD COLUMN fingerprint TEXT;
CREATE UNIQUE INDEX episodes_key ON episodes(key);

CREATE TABLE characters (
    id TEXT PRIMARY KEY CHECK (id GLOB '[a-z]*' AND id NOT GLOB '*[^a-z0-9-]*'),
    display TEXT NOT NULL,
    color TEXT
) STRICT;

CREATE TABLE character_names (
    character_id TEXT NOT NULL REFERENCES characters(id) ON DELETE CASCADE,
    source TEXT NOT NULL,                      -- 'name', 'subtitles', 'subtitles:vcb-hant', 'wd14', 'speech', 'lyrics'
    name TEXT NOT NULL,                        -- as written
    folded TEXT NOT NULL,                      -- NFKC, case-folded, separators dropped
    PRIMARY KEY (source, folded, character_id)
) STRICT;
CREATE INDEX character_names_folded ON character_names(folded);

ALTER TABLE clusters ADD COLUMN character_id TEXT REFERENCES characters(id) ON DELETE SET NULL;
ALTER TABLE speakers ADD COLUMN character_id TEXT REFERENCES characters(id) ON DELETE SET NULL;

-- V8: your work, mirrored from .tokimeki/work/ (the files win; re-imported when their hash changes)
CREATE TABLE work_files (
    path TEXT PRIMARY KEY,                     -- relative to work/
    sha256 TEXT NOT NULL,
    imported_at TEXT NOT NULL
) STRICT;

CREATE TABLE voice_labels (
    id INTEGER PRIMARY KEY,
    episode_id INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    start REAL NOT NULL,
    end REAL NOT NULL CHECK (end > start),
    kind TEXT NOT NULL CHECK (kind IN ('speech', 'noise', 'overlap', 'unsure')),
    character_id TEXT REFERENCES characters(id),   -- NULL with 'speech': someone minor ('other')
    by TEXT NOT NULL CHECK (by IN ('key', 'watch', 'cluster')),
    ai TEXT,                                       -- what the AI said when you labelled
    CHECK (kind = 'speech' OR character_id IS NULL)
) STRICT;
CREATE INDEX voice_labels_episode ON voice_labels(episode_id, start);

CREATE TABLE face_fixes (                          -- replaces cast_fixes
    episode_id INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    start REAL NOT NULL,                           -- the shot's range when you fixed it
    end REAL NOT NULL,
    from_character TEXT NOT NULL REFERENCES characters(id),
    to_character TEXT REFERENCES characters(id),   -- NULL: nobody
    PRIMARY KEY (episode_id, start, from_character)
) STRICT;

-- what the AI suggests beside the clustering; regenerable
CREATE TABLE voice_suggestions (
    line_id INTEGER NOT NULL REFERENCES lines(id) ON DELETE CASCADE,
    source TEXT NOT NULL CHECK (source IN ('neighbours', 'turns', 'agent')),
    character_id TEXT REFERENCES characters(id),
    confidence REAL NOT NULL,
    reason TEXT NOT NULL,
    PRIMARY KEY (line_id, source)
) STRICT;

-- V9, once the migration is verified: drop clusters.name, speakers.name, cast_fixes, speaker_fixes.
```

Resolving segments (labels over AI segments, §4.1) is plain Python in `voice/segments.py`. It is simpler and better tested there than as an SQL view.

### 5.3 API

All JSON, under `/api/<franchise>`, with episodes as `<series>/<nn>`:

| Method and path | For |
|---|---|
| `GET /api/franchises` | franchises, their series and progress |
| `GET /api/{f}/episodes/{s}/{e}` | duration, fps, media URLs, OP/ED, shots (kept and dropped ranges), on-screen cast per shot |
| `GET /api/{f}/episodes/{s}/{e}/segments` | the resolved segments with the AI's answer, its evidence, suggestions, your label, text and mentions; the label file's revision |
| `POST /api/{f}/episodes/{s}/{e}/labels` | a batch of operations (`set`, `clear`, `split`, `merge`, `bounds`) against a `base` revision; returns the new revision and the segments that changed, or 409 |
| `PUT /api/{f}/episodes/{s}/{e}/progress` | `watched_to`, `done` |
| `GET /api/{f}/queue` | lines that need you; filters: series, episode, reason |
| `GET /api/{f}/speakers` · `POST …/speakers/{id}/bind` · `POST …/speakers/{id}/split` | the speaker cluster view (what `api.voice_speakers` returns today) |
| `GET /api/{f}/characters/{id}/segments` | the character view: sorting, filters, paging |
| `GET /api/{f}/cast` · `PATCH /api/{f}/cast/{id}` · `POST /api/{f}/cast` | the registry |
| `GET /api/{f}/names?q=茉茉` | resolve a name: the character, or suggestions |
| `GET /api/{f}/names/inbox` · `POST /api/{f}/names/inbox/{name}` | unresolved names and WD14 hints; accept, ignore, or make a new character |
| `GET /api/{f}/clusters` · `POST …/clusters/{id}/bind` | face clusters |
| `POST /api/{f}/assist` · `GET /api/jobs/{id}` | queue an agent run (`dry_run` returns the count and the cost); job status |
| `GET /api/schema` | every JSON Schema, for development |

Media, all with Range:

| Path | What |
|---|---|
| `GET /media/{f}/{s}/{e}/video.mp4` | the proxy |
| `GET /media/{f}/{s}/{e}/voice.m4a` | the voice stem |
| `GET /media/{f}/{s}/{e}/peaks.bin` | waveform peaks |
| `GET /media/{f}/{s}/{e}/clip.wav?start=&end=&track=voice\|mix` | one segment's exact audio |
| `GET /media/{f}/{s}/{e}/frames/{n}.jpg?w=160` | a sampled frame (kept shots only; 404 otherwise), resized and cached |

### 5.4 Serving video and audio for scrubbing

- **The proxy.** Browsers cannot be counted on for HEVC Main10 in MKV with FLAC, so the `proxy` stage writes `cache/proxy/<key>.mp4`:
  - H.264 at 540p (NVENC), with a keyframe every 0.5 s so a seek lands at once;
  - AAC stereo of the main track;
  - `+faststart`, so the index comes first;
  - dropped shots black and silent.

  That is about 150–200 MB an episode (**Q10**). Beside it, `voice.m4a` (the stem as AAC) and `peaks.bin`: min/max per 10 ms as int8, about 280 KB an episode.
- **HTTP Range.** Starlette's `FileResponse` answers `Range: bytes=…` with 206 and `Content-Range` (and 416 when the range is invalid) since 0.39. Pin 0.49.1 or later for the multi-range fix. The video element then seeks by fetching only the bytes it needs. Tests check 200, 206, 416 and `Accept-Ranges` on a small fixture.
- **The voice track.** Pressing B mutes the video and plays `voice.m4a` in an `<audio>` element kept on the video's clock: re-synced on play, seek and rate change, and whenever drift exceeds 80 ms.
- **Exact segment playback.** R plays from the Web Audio API. `clip.wav` returns the segment plus 2 s on each side, cut from the cached stem (or the proxy's audio for the mix) and kept in a small in-memory cache. `AudioBufferSourceNode.start(when, offset, duration)` then plays the exact bounds to the sample, which `timeupdate` (about 4 Hz) cannot. Nudging an edge replays the 0.6 s around it the same way.
- **Thumbnails** come from the frame cache that already exists (kept frames every 0.5 s), so nothing new is extracted.

### 5.5 Shared types: JSON Schema → TypeScript

- The Python side keeps frozen dataclasses: the API's in `server/types.py`, the work file formats in `library/work.py`, and the plan as it is. One registry lists them.
- `pixi run schema` turns each into a JSON Schema with Pydantic's `TypeAdapter`, which works on plain dataclasses. The same adapter validates request bodies at the boundary only; the core never sees Pydantic. The hand-written plan schema joins the same registry. The schemas go to `web/src/api/schema/`, and `json-schema-to-typescript` writes `web/src/api/types.gen.ts`.
- `pixi run check` regenerates both into a temporary directory and fails if they differ from what is committed, then runs `tsc --noEmit` and the web lint. So the two sides cannot drift.
- The work file schemas are also published in `docs/formats/`: that is the stable export format.

### 5.6 Tests

Following the project's rule (small fixtures, never episodes):

- A franchise fixture: two 10 s synthetic clips (`testsrc`, tones), two series, a `cast.toml`.
- **Resolver:** 茉茉, 小闇 (scoped to one group), `momo_velia_deviluke`, honorifics, ambiguity, the pinyin and romaji suggestions.
- **Labels:** set, split, merge and bounds round trips; re-attachment when AI segments move; conflicts; 409 on a stale revision; the file stays line-per-segment.
- **Migration:** a v6 fixture library becomes v7 with keys, names become bindings, fixes are exported. The counts before and after are equal.
- **Proxy:** black and silent over a dropped shot. **Range:** 206, 416.
- **Rules:** no media file under `work/`; the assist request builder refuses a dropped shot.
- **Web:** vitest on the label reducer and the keymap.

## 6. Migration from today, step by step

The migration is a command, `tokimeki migrate`. It runs as a dry run that prints every move and rewrite until you pass `--apply`. Each step checks itself, and nothing is deleted until you say so at the end.

0. **Wait** until no ingest, download, training or server touches `~/anime`. Downloads are writing `.partial` files into `to-love-ru-darkness/s1/` and `~/anime/music/rakuen-project/` right now, and moving a directory under a downloader breaks its final rename.
1. **Snapshot** to `~/anime/.migration-2026-10-xx/`: `sqlite3 library.db ".backup …"` plus copies of `mads/`, `scratch/`, `voice/`, `subs/`, `logs/` and `~/.local/share/tokimeki/songs/` (about 250 MB, with no cache).
2. **Move the media** with `mv`, a rename within one filesystem, so nothing is copied:

   | From | To |
   |---|---|
   | `to-love-ru-darkness/s1` | `to-love-ru/darkness` |
   | `to-love-ru-darkness/s2` | `to-love-ru/darkness-2nd` |
   | `to-love-ru-darkness/oad` | `to-love-ru/darkness-oad` |
   | `to-love-ru-motto/tv` | `to-love-ru/motto` |
   | `to-love-ru-darkness/music/*` and `~/anime/music/rakuen-project` | `to-love-ru/music/` (rakuen-project merged: identical `.cue`, the FLAC from `~/anime/music/`) |
   | `to-love-ru-darkness/_subs/*.rar` | `to-love-ru/archives/` |
   | `.tokimeki/subs/zh-hant/*.ass` | `to-love-ru/darkness/zh-hant/` (another track; it was an input, not an output); `subs/zh-hans/` is a byte-identical copy of the track beside the video and stays only in the snapshot |
   | `.tokimeki/` | `to-love-ru/.tokimeki/`, reorganised as in §2.3 |
   | `~/anime/music/ngnl-zero` | stays |

   Check: every file is present at its new path, with the same count and bytes.
3. **Upgrade the library** to v7 and relink: each `episodes.path` is rewritten (`s1/…` becomes `darkness/…`) and given its key and fingerprint. Check: every episode resolves to an existing file with that fingerprint. The counts of shots, faces, clusters, lines, voices and speakers are unchanged.
4. **Write `franchise.toml` and `cast.toml`** from today's names. The 13 cluster names and 7 speaker names become 13 characters with ids (`momo`, `nana`, `rito`, `lala`, `yami`, `mea`, `yui`, `mikan`, `haruna`, `celine`, `risa`, `saruyama`, `principal`) whose display names are your current names. Aliases are seeded as suggestions in the inbox: 茉茉, 小闇 and 芽亞 from the subtitles, and the four WD14 tags. You confirm them on the cast page or in the CLI. Check: every named cluster and speaker is bound to a character.
5. **Export your work** into `work/`:
   - the 170 checked ep01 lines (`scratch/labels_ep01.txt`, line ids turned into times) → `labels/darkness/01.json` as `key` labels;
   - the 3 face fixes → `faces.json`, by shot time, plus exemplar faces for every bound cluster;
   - the three plans → `plans/` as plan v4 (franchise and character ids, episode keys), and `momo-ep1/prompt.md` beside them; their renders → `out/mads/`;
   - the two song analyses → `~/anime/.tokimeki/work/songs/<id>.toml` recipes plus `cache/songs/<id>.json`, with data-root-relative paths.

   Check: `voice check darkness/01` reproduces 145 of 146; every plan validates.
6. **Move the shared and generated files:** fonts to `$TOKIMEKI_HOME/fonts/` (one copy), the voice page to `out/`, logs to `logs/`. `scratch/` stays only in the snapshot.
7. **Start versioning:** `git init` in `work/`, and the first commit.
8. **Verify:** `tokimeki status to-love-ru` matches the numbers from before. Render `momo-ep1` as a preview and compare it with the old preview.mp4 (same cuts; the clip cache misses once because paths are part of its key).
9. **Clean up, only after you confirm:** remove the now-empty `to-love-ru-darkness/` and `to-love-ru-motto/`, `library.db.bak-v4` and `-v5`, and the old song directory. Keep the snapshot for a month.

Steps 2 and 3 are the only ones that change anything existing. Each is undone by moving the directories back and restoring the snapshot.

## 7. Build order

Each step is usable on its own:

| # | Step | Then you can |
|---|---|---|
| 1 | Franchise layout, episode keys, `tokimeki migrate`, `work/` as git, plan v4, song recipes | find everything in one place, back up one directory, and redo any stage without losing a fix |
| 2 | Registry: `cast.toml`, the resolver, clusters and speakers bound by id, `cast alias / resolve / bind`, the inbox in the CLI | use 梦梦, 茉茉, momo or モモ anywhere; Momo is one character across series |
| 3 | Labels as files, CLI first: import ep01's 170 lines, `voice check`, labels pin speakers and name clusters | measure the AI on every labelled episode; corrections survive every redo |
| 4 | Server, the `proxy` stage, the types pipeline, and the episode page read-only (Watch with speaker tags) | watch an episode and see who the AI thinks is speaking |
| 5 | Labeling on the episode page: keys, Watch and Step, progress, auto-commit | label an episode in minutes |
| 6 | Queue, speaker cluster and character views; the board; retire `voice page` and `voice serve` | clear a season's doubtful lines; bind clusters by ear |
| 7 | Feedback: neighbours from labels; banks from resolved segments with policies; MAD voice lines by speaker | banks and MADs use your labels |
| 8 | Agent assist for the leftover short lines, measured on ep01 first | settle the last few percent without listening to each |
| 9 | The cast page: names, aliases, colours, hotkeys, the inbox, face cluster binding | manage names without an editor |

Steps 1–3 touch `library/`, `cast/` and `voice/`. They start after `refactor/layout` and `voice/bank` merge. The bank's `voice/<name>/` directories become `voices/<character id>/` in step 1.

## 8. Questions for you

- **Q1 · One library per franchise?** Recommended: To LOVE-Ru's series share one cast, one set of clusters and one bank per character. The alternative is a library per series with a shared registry.
- **Q2 · Where do labels live?** Recommended: `.tokimeki/work/labels/`, so media directories stay read-only and all your work is one directory to back up. The alternative is a `.json` beside each video, a sidecar in the literal sense.
- **Q3 · Backup of `work/`:** a local git repository with automatic commits, plus a copy to another disk. Should it also go to a private remote (a GitHub private repository)? It holds ids, names, times and plans: no media, no subtitle text, no audio.
- **Q4 · File names:** keep the release names (recommended; subtitle tracks match them), or rename to `darkness/01.mkv`?
- **Q5 · Watch mode:** may a confident line that goes past without objection count as reviewed (`watch`)? Should banks use such lines by default (the `reviewed` policy), or only lines you pressed a key on (`strict`)? Recommended: yes and yes.
- **Q6 · Dropped shots on the labeling page:** black and silent, consistent with the content filter (recommended). Is that all right while labeling?
- **Q7 · The agent:** may it send kept keyframes and the subtitle text around the leftover lines to a cloud model through the batch API? Which model (Haiku 4.5 is cheapest, then Sonnet 5, then Opus 5)? Recommended: try two of them on ep01 and keep the one that holds up, with a cap of a few dollars a month.
- **Q8 · Names:** romaji ids (`momo`, `rito`, `yami`), your Chinese names for display, and Japanese beside them. Is that right? Which other names do you want shown?
- **Q9 · Series:** `tv`, `ova`, `motto`, `darkness`, `darkness-2nd`, `darkness-oad`, with the Darkness 2nd SP as `darkness-2nd/sp`. Does "To LOVE-Ru" in your list mean the 2008 TV series, the OVAs, or both?
- **Q10 · Proxy size:** 540p is about 150–200 MB an episode, roughly 15 GB for 80 episodes. Is that fine, or do you want 720p (about twice that)?
- **Q11 · Other devices:** should the labeling page ever be reachable from an iPad or the phone over Tailscale? Recommended: no, localhost only, since voice data never leaves the machine; the screenshot service stays a separate listener.
- **Q12 · Style labels** (whispering, shouting, the scheming voice): now, while labeling, or later? Recommended: later; the label format has room for a `tags` field.
