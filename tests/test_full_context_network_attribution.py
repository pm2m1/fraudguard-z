import json
import numpy as np
import pytest

from src.full_context import FULL_FEATURE_NAMES
from src.full_context_journal import FEATURE_SCHEMA_SHA256
from src.full_context_materialize import file_sha256
from src.full_context_network_attribution import BASE, NETWORK, EXCLUDED, variants, run


def test_predeclared_variants_exclude_current_amount_balance():
    groups = variants()
    assert len(groups) == 12
    assert len(NETWORK) == 5
    assert len(BASE) == 11
    assert [len(indices) for _, indices in groups] == [11]+[12]*5+[15]*5+[16]
    assert all(not (set(indices) & EXCLUDED) for _, indices in groups)
    assert len({name for name, _ in groups}) == 12


def test_diagnostic_run_isolated(tmp_path):
    source = tmp_path/"features"
    source.mkdir()
    split = tmp_path/"manifest.json"
    output = tmp_path/"attribution"
    n=40
    X=np.zeros((n,21),dtype=np.float32)
    X[:,1]=1
    X[:,9]=np.arange(n)
    X[:,16]=np.arange(n)%3
    y=np.tile(np.array([0,1,0,0],dtype=np.uint8),10)
    np.save(source/"features.npy",X)
    np.save(source/"labels.npy",y)
    partitions={name:{"rows":10,"frauds":int(y[i*10:(i+1)*10].sum())}
                for i,name in enumerate(("train","validation","calibration","policy"))}
    manifest={"dev_max_step":372,"final_holdout_start_step":373,
              "split_end_steps":[10,20,30],
              "split_end_rows":[10,20,30],
              "development_row_count":40,"partitions":partitions}
    split.write_text(json.dumps(manifest))
    (source/"materialization.json").write_text(json.dumps({
        "status":"DEVELOPMENT_ONLY_FEATURE_MATERIALIZATION",
        "schema":list(FULL_FEATURE_NAMES),"schema_sha256":FEATURE_SCHEMA_SHA256,
        "split_manifest_sha256":file_sha256(split),
        "rows":40,"partitions":partitions}))
    result=run(source,split,output,families=("logistic",))
    assert len(result["candidates"])==12
    assert result["validation_rows"]==10
    assert (output/"comparison.json").is_file()
    assert len(list(output.glob("*.joblib")))==12
    with pytest.raises(FileExistsError):
        run(source,split,output,families=("logistic",))
