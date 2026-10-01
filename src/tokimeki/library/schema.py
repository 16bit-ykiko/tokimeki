"""Library schema, one script per version; `PRAGMA user_version` records the applied count."""

V1 = """
CREATE TABLE episodes (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    fps_num INTEGER NOT NULL,
    fps_den INTEGER NOT NULL,
    frame_count INTEGER NOT NULL
) STRICT;

CREATE TABLE stage_runs (
    episode_id INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    params TEXT NOT NULL,
    PRIMARY KEY (episode_id, stage)
) STRICT;

CREATE TABLE shots (
    id INTEGER PRIMARY KEY,
    episode_id INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    start_frame INTEGER NOT NULL,
    end_frame INTEGER NOT NULL CHECK (end_frame > start_frame),
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'kept', 'dropped')),
    UNIQUE (episode_id, idx)
) STRICT;

CREATE TABLE frames (
    id INTEGER PRIMARY KEY,
    shot_id INTEGER NOT NULL REFERENCES shots(id) ON DELETE CASCADE,
    frame_index INTEGER NOT NULL,
    general REAL NOT NULL,
    sensitive REAL NOT NULL,
    questionable REAL NOT NULL,
    explicit REAL NOT NULL,
    UNIQUE (shot_id, frame_index)
) STRICT;

CREATE TABLE frame_tags (
    frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    category TEXT NOT NULL CHECK (category IN ('general', 'character')),
    score REAL NOT NULL,
    PRIMARY KEY (frame_id, tag)
) STRICT, WITHOUT ROWID;

CREATE TABLE clusters (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE
) STRICT;

CREATE TABLE faces (
    id INTEGER PRIMARY KEY,
    frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    x0 REAL NOT NULL,
    y0 REAL NOT NULL,
    x1 REAL NOT NULL,
    y1 REAL NOT NULL,
    score REAL NOT NULL,
    embedding BLOB NOT NULL,
    cluster_id INTEGER REFERENCES clusters(id) ON DELETE SET NULL
) STRICT;

CREATE INDEX faces_frame ON faces(frame_id);
CREATE INDEX faces_cluster ON faces(cluster_id);

-- The content filter is enforced here too: only kept shots may own frames, and a shot
-- cannot leave the kept state while it still owns any.
CREATE TRIGGER frames_only_for_kept_shots
BEFORE INSERT ON frames
WHEN (SELECT status FROM shots WHERE id = NEW.shot_id) IS NOT 'kept'
BEGIN
    SELECT RAISE(ABORT, 'frames are stored only for kept shots');
END;

CREATE TRIGGER shot_with_frames_stays_kept
BEFORE UPDATE OF status ON shots
WHEN NEW.status != 'kept' AND EXISTS (SELECT 1 FROM frames WHERE shot_id = NEW.id)
BEGIN
    SELECT RAISE(ABORT, 'delete a shot''s frames before it leaves the kept state');
END;

CREATE VIEW shot_cast AS
SELECT
    fr.shot_id AS shot_id,
    fa.cluster_id AS cluster_id,
    CAST(COUNT(DISTINCT fa.frame_id) AS REAL)
        / (SELECT COUNT(*) FROM frames AS f2 WHERE f2.shot_id = fr.shot_id) AS presence,
    MAX(fa.y1 - fa.y0) AS face_height,
    AVG((fa.x1 - fa.x0) * (fa.y1 - fa.y0)) AS face_area
FROM faces AS fa
JOIN frames AS fr ON fr.id = fa.frame_id
WHERE fa.cluster_id IS NOT NULL
GROUP BY fr.shot_id, fa.cluster_id;
"""

# Cluster ids are what the user types to name clusters, so a deleted id is never handed out again.
V2 = """
CREATE TABLE clusters_v2 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE
) STRICT;
INSERT INTO clusters_v2 (id, name) SELECT id, name FROM clusters;
DROP TABLE clusters;
ALTER TABLE clusters_v2 RENAME TO clusters;
"""

MIGRATIONS: tuple[str, ...] = (V1, V2)
