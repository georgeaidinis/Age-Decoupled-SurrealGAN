from __future__ import annotations

import pandas as pd

from age_decoupled_surrealgan.config import ProjectConfig
from age_decoupled_surrealgan.data.prepare import canonicalize_dataframe, create_split_manifest


def test_canonicalize_dataframe_marks_ref_and_tar_rows():
    cfg = ProjectConfig()
    frame = pd.DataFrame(
        {
            "PTID": ["s1", "s2", "s3", "s4"],
            "Study": ["A", "A", "B", "HANDLS"],
            "Age": [25, 63, 38, 70],
            "Sex": ["F", "M", "F", "M"],
            "Diagnosis": ["CN", "AD", "CN", "CN"],
            "MUSE_Volume_101": [1.0, 2.0, 3.0, 4.0],
            "MUSE_Volume_102": [2.0, 3.0, 4.0, 5.0],
            "MUSE_Volume_345": [5.0, 6.0, 7.0, 8.0],
        }
    )
    canonical, features = canonicalize_dataframe(frame, cfg)
    assert features == ["MUSE_Volume_101", "MUSE_Volume_102"]
    assert canonical["cohort_bucket"].tolist() == ["ref", "tar", "ref", "tar"]


def test_create_split_manifest_includes_holdout_study_as_ood():
    cfg = ProjectConfig()
    frame = pd.DataFrame(
        {
            "subject_id": [f"s{i}" for i in range(12)],
            "study": ["A"] * 10 + ["HANDLS"] * 2,
            "age": [25, 27, 29, 31, 55, 57, 59, 61, 63, 65, 45, 75],
            "sex": ["F"] * 12,
            "diagnosis_raw": ["CN", "CN", "CN", "CN", "AD", "AD", "AD", "AD", "AD", "AD", "CN", "CN"],
            "diagnosis_group": ["normal"] * 4 + ["ad"] * 6 + ["normal", "normal"],
            "cohort_bucket": ["ref"] * 4 + ["tar"] * 8,
            "eligible": [True] * 12,
            "is_holdout_study": [False] * 10 + [True] * 2,
        }
    )
    manifest = create_split_manifest(frame, cfg)
    assert set(manifest["ood_test"]) == {"s10", "s11"}
    assert set(manifest["application"]) == {"s10", "s11"}
