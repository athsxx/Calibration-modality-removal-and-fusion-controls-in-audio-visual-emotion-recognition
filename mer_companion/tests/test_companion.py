from __future__ import annotations
import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from merkit.common import (expected_runs, metadata_eligibility, validate_splits, sha256,
    write_json, write_csv, within, SPECS, PRIMARY_PAIRS)
from merkit.export import keep_mask, validate_export_selection, compare_reference_predictions
from merkit.analysis import (softmax, score, paired, analyze_frames, validate_table,
    fit_temperature, holm, primary_tests, run_analysis)
from merkit.support import run_ravdess_manifest, run_pack

def frame(ids, labels, probabilities, condition="clean", split="test", model="audio_w2v", seed=0, fold=1):
    p = np.asarray(probabilities, dtype=float)
    z = np.log(p)
    result = pd.DataFrame(dict(model=model, seed=seed, fold=fold, utterance_id=ids,
        condition=condition, split=split, label=labels))
    for k in range(4):
        result[f"p_{k}"] = p[:, k]; result[f"z_{k}"] = z[:, k]
    return result

def simple_frames():
    p = [[.97,.01,.01,.01], [.97,.01,.01,.01], [.01,.97,.01,.01], [.01,.01,.97,.01]]
    dev = frame(["v1","v2","v3","v4"], [0,1,2,3], p, split="validation")
    test = frame(["t1","t2","t3","t4"], [0,1,2,3], p)
    changed = frame(["t1","t2","t3","t4"], [0,1,2,3], p, condition="zero_audio")
    return dev, pd.concat([test, changed], ignore_index=True)

class AuditTests(unittest.TestCase):
    def record(self):
        s = next(r for r in expected_runs() if r["model"] == "essence_v2")
        c = dict(model_name="essence", audio_dir="audio_w2v", video_dir="video_v2", epoch=22, stage=2)
        m = dict(seed=0, best_epoch=22, stage2_reached=True, used_stage1_fallback=False,
                 aborted_nonfinite=False, nonfinite_batch_count=0, test={"macro_f1": .5})
        return s,c,m
    def test_75_unique_expected_identities(self):
        rows=expected_runs(); self.assertEqual(len(rows),75)
        self.assertEqual(len({r["run_id"] for r in rows}),75)
    def test_valid_metadata(self):
        self.assertEqual(metadata_eligibility(*self.record())[0],"eligible")
    def test_stage1_rejected_even_when_flag_says_stage2(self):
        s,c,m=self.record(); c["stage"]=1
        self.assertEqual(metadata_eligibility(s,c,m)[0],"rejected")
    def test_fallback_and_nonfinite_rejected(self):
        for field,value in (("used_stage1_fallback",True),("nonfinite_batch_count",1),("aborted_nonfinite",True)):
            s,c,m=self.record();m[field]=value
            self.assertEqual(metadata_eligibility(s,c,m)[0],"rejected")
    def test_missing_history_is_not_zero(self):
        s,c,m=self.record();del m["nonfinite_batch_count"]
        self.assertEqual(metadata_eligibility(s,c,m)[0],"history_unknown")
    def test_stale_epoch_rejected(self):
        s,c,m=self.record();c["epoch"]=99
        self.assertEqual(metadata_eligibility(s,c,m)[0],"rejected")
    def test_nan_reference_score_rejected(self):
        s,c,m=self.record();m["test"]["macro_f1"]=float("nan")
        self.assertEqual(metadata_eligibility(s,c,m)[0],"rejected")
    def test_rejected_cannot_be_allowed_as_unknown(self):
        a={"inputs_valid":True,"runs":[dict(model="m",run_id="r",status="rejected")]}
        with self.assertRaisesRegex(ValueError,"Rejected"):
            validate_export_selection(a,None,None,True)
    def test_unknown_requires_explicit_flag(self):
        a={"inputs_valid":True,"runs":[dict(model="m",run_id="r",status="history_unknown")]}
        with self.assertRaises(ValueError): validate_export_selection(a,None,None,False)
        self.assertEqual(len(validate_export_selection(a,None,None,True)),1)
    def splits(self):
        ids=[f"Ses{i:02d}F_impro01_F000" for i in range(1,6)]
        data=pd.DataFrame(dict(utterance_id=ids,emotion4=["angry"]*5))
        splits={}
        for i in range(5):
            splits[f"fold_{i+1}"]={"test":[ids[i]],"dev":[ids[(i+1)%5]],
                "train":[u for j,u in enumerate(ids) if j not in (i,(i+1)%5)]}
        return data,splits
    def test_splits_cover_5_outer_tests(self):
        self.assertEqual(len(validate_splits(*self.splits())),25)
    def test_speaker_leakage_rejected(self):
        data,splits=self.splits(); extra="Ses01F_impro02_F000"
        data.loc[len(data)]=[extra,"angry"]
        for fold in splits.values(): fold["train"].append(extra)
        with self.assertRaisesRegex(ValueError,"Speaker leakage"):validate_splits(data,splits)
    def test_paths_cannot_escape(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):within(Path(d),"../outside")

class AnalysisTests(unittest.TestCase):
    def test_saved_predictions_checked_by_id(self):
        reference=pd.DataFrame(dict(utterance_id=["b","a"],label=[1,0],prediction=[0,1]))
        compare_reference_predictions(["a","b"],[0,1],[1,0],reference)
        with self.assertRaisesRegex(ValueError,"predicted classes differ"):
            compare_reference_predictions(["a","b"],[0,1],[0,1],reference)
    def test_uniform_metrics(self):
        s=score(np.arange(4),np.zeros((4,4)))
        self.assertAlmostEqual(s["nll"],np.log(4));self.assertAlmostEqual(s["brier"],.75)
    def test_extreme_logits_nll_has_no_hidden_clip(self):
        self.assertAlmostEqual(score(np.array([1]),np.array([[1000,-1000,-1000,-1000]]))["nll"],2000)
    def test_argmax_flip_vrr_uses_new_maximum(self):
        clean=frame(["a","b"],[0,1],[[.6,.2,.1,.1],[.1,.7,.1,.1]])
        alt=frame(["a","b"],[0,1],[[.1,.8,.05,.05],[.2,.6,.1,.1]],condition="zero_audio")
        p=paired(clean,alt.iloc[::-1]);self.assertEqual(p["vrr"],.5)
        self.assertEqual(p["clean_correct_altered_wrong_fraction"],.5)
    def test_identity_mismatch_and_duplicates_rejected(self):
        _,test=simple_frames();a=test[test.condition=="clean"]
        with self.assertRaises(ValueError):paired(a,a.iloc[:3])
        with self.assertRaises(ValueError):paired(a,pd.concat([a,a.iloc[:1]]))
    def test_label_disagreement_rejected(self):
        _,test=simple_frames();a=test[test.condition=="clean"];b=a.copy();b.loc[b.index[0],"label"]=3
        with self.assertRaisesRegex(ValueError,"Labels disagree"):paired(a,b)
    def test_temperature_never_uses_test_labels(self):
        dev,test=simple_frames();first=analyze_frames(dev,test,["clean","zero_audio"])[2]
        test["label"]=3-test.label
        second=analyze_frames(dev,test,["clean","zero_audio"])[2]
        self.assertEqual(first,second)
        self.assertLessEqual(first["validation_nll_after"],first["validation_nll_before"])
    def test_development_test_overlap_rejected(self):
        dev,test=simple_frames();dev.loc[0,"utterance_id"]="t1"
        with self.assertRaisesRegex(ValueError,"leakage"):analyze_frames(dev,test,["clean","zero_audio"])
    def test_wrong_split_rejected(self):
        dev,test=simple_frames();dev["split"]="test"
        with self.assertRaisesRegex(ValueError,"split=validation"):analyze_frames(dev,test,["clean","zero_audio"])
    def test_probability_logit_mismatch_rejected(self):
        _,test=simple_frames();test.loc[0,["p_0","p_1"]]=[.01,.97]
        with self.assertRaisesRegex(ValueError,"disagree"):validate_table(test,"test")
    def test_holm_keeps_declared_family_size(self):
        np.testing.assert_allclose(holm([.01,.03,.04]),[.03,.06,.06])
        self.assertAlmostEqual(holm([.01,np.nan,.04])[0],.03)
    def test_partial_loss_deterministic_nested_nonempty(self):
        a=keep_mask(12,.25,"id");b=keep_mask(12,.5,"id")
        np.testing.assert_array_equal(a,keep_mask(12,.25,"id"))
        self.assertTrue(np.all(~a <= ~b));self.assertTrue(keep_mask(1,.75,"id").all())
        self.assertEqual(a.sum(),9);self.assertEqual(b.sum(),6)
    def test_primary_tests_require_complete_grid(self):
        rows=[]
        for i,spec in enumerate(expected_runs()):
            rows.append(dict(model=spec["model"],seed=spec["seed"],fold=spec["fold"],
                condition="clean",calibration="raw",macro_f1=.4 + list(SPECS).index(spec["model"])*.01*spec["fold"]))
        scored=pd.DataFrame(rows);tests,diffs,_=primary_tests(scored)
        self.assertEqual(len(tests),7);self.assertEqual(len(diffs),35)
        self.assertEqual(primary_tests(scored.iloc[1:])[0],[])

class IntegrationTests(unittest.TestCase):
    def test_actual_analysis_cli_on_synthetic_export(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);exports=root/"exports";run_dir=exports/"runs/r";run_dir.mkdir(parents=True)
            dev,test=simple_frames();records={};splits=[]
            for part,f in (("validation",dev),("test",test)):
                file=run_dir/f"{part}_predictions.csv.gz";f.to_csv(file,index=False)
                records[part]=dict(file=file.name,rows=len(f),sha256=sha256(file))
                for uid,y in f[["utterance_id","label"]].drop_duplicates().itertuples(index=False,name=None):
                    splits.append(dict(utterance_id=uid,label=y,split=part,fold=1))
            write_csv(exports/"split_manifest.csv",splits)
            payload=dict(complete=True,selected_count=1,history_unknown_count=0,
                settings=dict(selected=["r"],split_manifest_sha256=sha256(exports/"split_manifest.csv")),
                runs=[dict(run_id="r",model="audio_w2v",seed=0,fold=1,conditions=["clean","zero_audio"],
                           history_status="eligible",files=records)])
            write_json(exports/"export_manifest.json",payload)
            cli=Path(__file__).resolve().parents[1]/"mer_companion.py"
            p=subprocess.run([sys.executable,str(cli),"analyze","--exports",str(exports),
                "--out",str(root/"analysis"),"--allow-partial"],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            scores=pd.read_csv(root/"analysis/fold_scores.csv")
            self.assertEqual(len(scores),4)
            with self.assertRaises(FileExistsError):
                run_analysis(SimpleNamespace(exports=exports,out=root/"analysis",allow_partial=True))
    def test_ravdess_filter_and_actor_disjointness(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw=root/"raw";raw.mkdir()
            for actor in range(1,25):
                for emotion in ("01","03","04","05"):
                    for intensity in (["01"] if emotion=="01" else ["01","02"]):
                        for statement in ("01","02"):
                            for repetition in ("01","02"):
                                (raw/f"01-01-{emotion}-{intensity}-{statement}-{repetition}-{actor:02d}.mp4").write_bytes(b"fixture")
            (raw/"02-01-01-01-01-01-01.mp4").write_bytes(b"video-only duplicate")
            run_ravdess_manifest(SimpleNamespace(raw=raw,out=root/"ravdess"))
            m=pd.read_csv(root/"ravdess/ravdess_manifest.csv");s=pd.read_csv(root/"ravdess/split_manifest.csv")
            self.assertEqual(len(m),672)
            for _,g in s.groupby("fold"):
                sets=[set(g[g.split==part].speaker_id) for part in ("train","validation","test")]
                self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    def test_pack_omits_model_and_feature_files(self):
        import zipfile
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out=root/"mtap_outputs";out.mkdir()
            (out/"scores.csv").write_text("x\n1\n");(out/"weights.pt").write_bytes(b"not included")
            run_pack(SimpleNamespace(outputs=out,out=root/"return.zip"))
            with zipfile.ZipFile(root/"return.zip") as z:
                self.assertIn("mtap_outputs/scores.csv",z.namelist())
                self.assertNotIn("mtap_outputs/weights.pt",z.namelist())

if __name__ == "__main__":
    unittest.main()
