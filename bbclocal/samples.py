"""Bias-correction samples (IDSAMPLE) from surveygroup_biascor and fieldGroup_biascor, as in prepare_IDSAMPLE_biasCor,
match_fieldGroup and match_surveyGroup of SALT2mu.c.

Rules reproduced: an event whose FIELD (split on '+') shares a substring, either way, with a field group belongs to the first
such field group (in input order); events in the same field group form one sample regardless of survey; every other event
belongs to the survey group that lists its IDSURVEY. A survey that is not in any group forms its own group (SALT2mu then has
no z bin size for it, which bbclocal refuses).

Not reproduced (refused): SALT2mu keeps the field group of the previous event when an event's FIELD matches no group, so an
unmatched event that follows a matched one in file order would be assigned to the previous event's field group.
"""
from dataclasses import dataclass

import numpy as np

from .config import UnsupportedOptionError


@dataclass
class Sample:
    key: tuple
    name: str
    zbin: float | None


def _resolve_members(cfg, survey_ids):
    out = []
    for g in cfg.surveygroups:
        ids = []
        for mname in g.members:
            if mname.lstrip("-").isdigit():
                ids.append(int(mname))
            elif survey_ids and mname in survey_ids:
                ids.append(int(survey_ids[mname]))
            else:
                raise UnsupportedOptionError(
                    f"survey '{mname}' of surveygroup_biascor has no IDSURVEY: give a SURVEY.DEF file, a name->IDSURVEY map, "
                    "or write the group with IDSURVEY integers")
        out.append(ids)
    return out


def _match_field(field, fieldgroups):
    """Index of the matching field group, or -1. Raises on an ambiguous match of a single (no '+') field, as SALT2mu."""
    if not fieldgroups or field == "NONE":
        return -1
    parts = field.split("+") if "+" in field else [field]
    first, nmatch = -1, 0
    for i, g in enumerate(fieldgroups):
        for p in parts:
            if (p in g.name) or (g.name in p):
                if first < 0:
                    first = i
                nmatch += 1
    if nmatch > 1 and len(parts) == 1:
        raise UnsupportedOptionError(f"FIELD '{field}' matches {nmatch} field groups (SALT2mu aborts on this)")
    return first


class SampleSet:
    def __init__(self, cfg, survey_ids=None):
        self.cfg = cfg
        self.members = _resolve_members(cfg, survey_ids)
        self.id2group = {}
        for gi, ids in enumerate(self.members):
            for i in ids:
                if i in self.id2group and self.id2group[i] != gi:
                    raise UnsupportedOptionError(f"IDSURVEY {i} is listed in two survey groups")
                self.id2group[i] = gi
        self.names_by_id = {v: k for k, v in (survey_ids or {}).items()}
        self.samples = []
        self._index = {}

    def _key(self, idsurvey, fg):
        cfg = self.cfg
        sg = self.id2group.get(int(idsurvey), None)
        if fg >= 0:
            fgrp = cfg.fieldgroups[fg]
            if fgrp.zbin is not None:
                sgname = self.names_by_id.get(int(idsurvey), str(idsurvey))
                zbin = fgrp.zbin
            elif sg is not None:
                sgname, zbin = cfg.surveygroups[sg].name, cfg.surveygroups[sg].zbin
            else:
                sgname, zbin = self.names_by_id.get(int(idsurvey), str(idsurvey)), None
            return ("F", fg), f"{sgname}({fgrp.name})", zbin
        if sg is not None:
            return ("S", sg), cfg.surveygroups[sg].name, cfg.surveygroups[sg].zbin
        return ("A", int(idsurvey)), self.names_by_id.get(int(idsurvey), str(idsurvey)), None

    def _row_keys(self, idsurvey, field_code, field_names, what):
        """Per-row (key, name, zbin) via the unique (IDSURVEY, FIELD) pairs; checks the stale-field-group condition.
        field_code: per-row index into field_names."""
        idsurvey = np.asarray(idsurvey).astype(np.int64)
        field_code = np.asarray(field_code).astype(np.int64)
        field_names = np.asarray(field_names).astype(str)
        uid, id_code = np.unique(idsurvey, return_inverse=True)
        nf = max(len(field_names), 1)
        upair, inv = np.unique(id_code.ravel() * nf + field_code, return_inverse=True)
        inv = np.asarray(inv).ravel()
        pair_id = uid[upair // nf]
        pair_field = field_names[upair % nf]
        fgs = np.array([_match_field(f, self.cfg.fieldgroups) for f in pair_field], dtype=int)
        if self.cfg.fieldgroups:
            matched = fgs[inv] >= 0
            if matched.any():
                first = int(np.argmax(matched))
                if (~matched[first:]).any():
                    raise UnsupportedOptionError(
                        f"{what}: an event whose FIELD matches no field group follows (in file order) an event that matches one; "
                        "SALT2mu would assign it to the previous event's field group. Reorder the file(s) so that such events "
                        "come first.")
        keys = [self._key(int(i), int(fg)) for i, fg in zip(pair_id, fgs)]
        return keys, inv

    def build_from_data(self, idsurvey, field):
        """Define the samples from the data rows (all rows, before cuts); returns the per-row sample index."""
        names, code = np.unique(np.asarray(field).astype(str), return_inverse=True)
        keys, inv = self._row_keys(idsurvey, code.ravel(), names, "data")
        _, first = np.unique(inv, return_index=True)
        for j in np.argsort(first, kind="stable"):   # samples in order of first appearance
            key, name, zbin = keys[j]
            if key not in self._index:
                self._index[key] = len(self.samples)
                self.samples.append(Sample(key, name, zbin))
        lut = np.array([self._index[k[0]] for k in keys])
        return lut[inv]

    def assign(self, idsurvey, field_code, field_names, what="bias-correction simulation"):
        """Sample index of other rows (bias-correction simulations); -1 if the row's sample is not in the data."""
        keys, inv = self._row_keys(idsurvey, field_code, field_names, what)
        lut = np.array([self._index.get(k[0], -1) for k in keys])
        return lut[inv]
